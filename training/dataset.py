import os

import h5py
import numpy as np
import torch
import xarray as xr


class AbstractSDADataset(torch.utils.data.Dataset):
    def __init__(self, order):
        self._order = order
        self._window = 2 * order + 1

    @property
    def window(self):
        return self._window

    @property
    def order(self):
        return self._order


class ERADataset(AbstractSDADataset):
    def __init__(
        self,
        data_path,
        stats_dir=None,
        cached=True,
        norm_mode="minmax",
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)

        # SET UP DATASET
        self._data_path = os.path.abspath(data_path)
        self._dataset_path = data_path
        if stats_dir is None:
            stats_dir = os.path.join(os.path.dirname(data_path), "../stats")
        print(stats_dir)
        assert os.path.exists(stats_dir)
        assert os.path.isdir(stats_dir)
        self._cached = cached
        self.norm_mode = norm_mode

        self._norm_stats = dict()
        if norm_mode == "minmax":
            min_ds = xr.load_dataset(os.path.join(stats_dir, "min.nc"))
            max_ds = xr.load_dataset(os.path.join(stats_dir, "max.nc"))
            range_ds = max_ds - min_ds
            self._norm_stats["offset"] = min_ds
            self._norm_stats["scale"] = range_ds
        elif norm_mode == "zscore":
            mean_ds = xr.load_dataset(os.path.join(stats_dir, "mean.nc"))
            std_ds = xr.load_dataset(os.path.join(stats_dir, "std.nc"))
            self._norm_stats["offset"] = mean_ds
            self._norm_stats["scale"] = std_ds
        elif norm_mode == "mixed":
            shift_ds = xr.load_dataset(os.path.join(stats_dir, "norm_shift.nc"))
            scale_ds = xr.load_dataset(os.path.join(stats_dir, "norm_scale.nc"))
            self._norm_stats["offset"] = shift_ds
            self._norm_stats["scale"] = scale_ds
        else:
            raise ValueError(f"Unknown norm_mode: {norm_mode}")

        self.ORDERED_FEATURE_NAMES = (
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "surface_solar_radiation_downwards",
        )

        if cached:
            self.xr_dataset = xr.load_dataset(self._dataset_path).transpose(
                "time", "lat", "lon"
            )
        else:
            self.xr_dataset = xr.open_dataset(self._dataset_path).transpose(
                "time", "lat", "lon"
            )

    def __len__(self) -> int:
        return self.xr_dataset.sizes["time"] - self.window + 1

    @property
    def spatial_res(self):
        return (self.xr_dataset.sizes["lat"], self.xr_dataset.sizes["lon"])

    @property
    def feature_names(self):
        return self.ORDERED_FEATURE_NAMES

    @property
    def num_features(self):
        return len(self.feature_names)

    @property
    def norm_stats(self):
        return self._norm_stats

    def _stack_to_np(self, ds, feature_names=None):
        if feature_names is None:
            feature_names = self.feature_names
        return np.stack(
            [ds[var].values for var in feature_names],
            axis=1,
        )

    def normalize_ds(self, ds):
        norm_ds = (ds - self._norm_stats["offset"]) / self._norm_stats["scale"]
        if self.norm_mode == "zscore":
            return norm_ds

        return norm_ds * 2 - 1

    def denormalize_ds(self, ds):
        if self.norm_mode == "zscore":
            return ds * self._norm_stats["scale"] + self._norm_stats["offset"]
        denorm_ds = (ds + 1) / 2
        return denorm_ds * self._norm_stats["scale"] + self._norm_stats["offset"]

    def _norm_stats_to_tensor(self):
        norm_offset_tensor = torch.stack(
            [
                torch.from_numpy(self._norm_stats["offset"][v].values.reshape(1, 1, 1))
                for v in self.feature_names
            ],
            dim=1,
        )
        norm_scale_tensor = torch.stack(
            [
                torch.from_numpy(self._norm_stats["scale"][v].values.reshape(1, 1, 1))
                for v in self.feature_names
            ],
            dim=1,
        )
        return {"tensor_offset": norm_offset_tensor, "tensor_scale": norm_scale_tensor}

    def normalize_tensor(self, *, x_LCHW):
        *leading_dims, L, C, H, W = x_LCHW.shape
        assert C == len(self.feature_names)

        tensor_norm_stats = self._norm_stats_to_tensor()
        norm_offset_tensor = tensor_norm_stats["tensor_offset"]
        norm_scale_tensor = tensor_norm_stats["tensor_scale"]

        assert norm_offset_tensor.shape == (1, C, 1, 1)
        assert norm_scale_tensor.shape == (1, C, 1, 1)

        # add back leading dimensions
        norm_offset_tensor = norm_offset_tensor.reshape(
            tuple([1] * len(leading_dims)) + tuple(norm_offset_tensor.shape)
        ).to(device=x_LCHW.device, dtype=x_LCHW.dtype)
        norm_scale_tensor = norm_scale_tensor.reshape(
            tuple([1] * len(leading_dims)) + tuple(norm_scale_tensor.shape)
        ).to(device=x_LCHW.device, dtype=x_LCHW.dtype)

        return (x_LCHW - norm_offset_tensor) / norm_scale_tensor

    def denormalize_tensor(self, *, x_LCHW):
        *leading_dims, L, C, H, W = x_LCHW.shape
        assert C == len(self.feature_names)

        tensor_norm_stats = self._norm_stats_to_tensor()
        norm_offset_tensor = tensor_norm_stats["tensor_offset"]
        norm_scale_tensor = tensor_norm_stats["tensor_scale"]

        assert norm_offset_tensor.shape == (1, C, 1, 1)
        assert norm_scale_tensor.shape == (1, C, 1, 1)

        # add back leading dimensions
        norm_offset_tensor = norm_offset_tensor.reshape(
            tuple([1] * len(leading_dims)) + tuple(norm_offset_tensor.shape)
        ).to(device=x_LCHW.device, dtype=x_LCHW.dtype)
        norm_scale_tensor = norm_scale_tensor.reshape(
            tuple([1] * len(leading_dims)) + tuple(norm_scale_tensor.shape)
        ).to(device=x_LCHW.device, dtype=x_LCHW.dtype)

        return x_LCHW * norm_scale_tensor + norm_offset_tensor

    def load_window(self, i: int):  # -> [L, C, H, W]
        return self.xr_dataset.isel(time=slice(i, i + self.window))

    def __getitem__(self, i):
        window_ds = self.load_window(i)
        month_idcs = torch.from_numpy(window_ds.time.dt.month.values)  # [L, ]
        hour_idcs = torch.from_numpy(window_ds.time.dt.hour.values)  # [L, ]
        # Normalize
        window_ds = self.normalize_ds(window_ds)
        x = torch.from_numpy(self._stack_to_np(window_ds))  # [L, C, H, W]
        x = x.flatten(0, 1)  # [L * C, H, W]
        month_idcs = torch.LongTensor(month_idcs) - 1
        hour_idcs = torch.LongTensor(hour_idcs) // 6
        return x, month_idcs, hour_idcs


class COSMODataset(AbstractSDADataset):
    def __init__(
        self,
        data_path,
        num_features=4,
        spatial_res=128,
        cached=False,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)

        # SET UP DATASET
        self._data_path = os.path.abspath(data_path)
        self._h5_data_var = "x"
        assert os.path.exists(self._data_path)
        assert os.path.isfile(self._data_path)
        assert os.path.splitext(self._data_path)[-1] == ".h5"

        self.ORDERED_FEATURE_NAMES = ("psl", "tas", "uas", "vas")

        self._cached = cached
        if self._cached:
            with h5py.File(self._data_path, mode="r") as f:
                self.dataset = f[self._h5_data_var][:]  # [[N], L, C, H, W]
                self._h5_ds_shape = self.dataset.shape
        else:
            self.dataset = None
            with h5py.File(self._data_path, mode="r") as f:
                self._h5_ds_shape = f[self._h5_data_var].shape

        assert self._h5_ds_shape[-1] == self._h5_ds_shape[-2] == spatial_res
        self.spatial_res = spatial_res

        assert num_features == self.num_features, (
            f"The number of specified features ({num_features}) does not match the number of features in the data ({self.num_features})."
        )

    def __len__(self) -> int:
        return self._h5_ds_shape[0] - self.window + 1

    @property
    def raw_data_shape(self):
        return self._h5_ds_shape

    @property
    def raw_spatial_res(self):
        return self.spatial_res

    @property
    def num_features(self):
        return self._h5_ds_shape[-3]

    @property
    def data_path(self):
        return self._data_path

    def load_window(self, i: int):  # -> [L, C, H, W]
        if (not self._cached) and (self.dataset is None):
            self.dataset = h5py.File(self._data_path, "r")[self._h5_data_var]

        traj = torch.from_numpy(self.dataset[i : i + self.window, ...])
        return traj

    def __getitem__(self, i):
        x = self.load_window(i)  # [L, C, H, W]
        return x.flatten(0, 1)  # [L * C, H, W]
