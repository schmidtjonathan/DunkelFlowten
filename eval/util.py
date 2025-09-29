import random
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import inspect
import os


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior


def create_unique_directory(base_path, unique=True) -> Path:
    """
    Creates a unique directory based on the given base_path.
    If the directory already exists, appends a suffix `_i` where `i` is an index.
    """

    if isinstance(base_path, str):
        base_path = Path(base_path)

    if not base_path.exists():
        base_path.mkdir(parents=True)
        return base_path

    if not unique:
        return base_path
    else:
        index = 1
        while True:
            new_path = base_path.parent / f"{base_path.name}_{index}"
            if not new_path.exists():
                new_path.mkdir(parents=True)
                return new_path
            index += 1


def np_to_ds(np_arr, data_vars, coords=None):

    time = coords["time"]
    lat = coords["lat"]
    lon = coords["lon"]
    # print(f"np_arr shape: {np_arr.shape}")
    assert np_arr.shape[0] == len(time)
    assert np_arr.shape[1] == len(data_vars)
    assert np_arr.shape[2] == len(lat)
    assert np_arr.shape[3] == len(lon)
    if isinstance(np_arr, torch.Tensor):
        np_arr = np_arr.detach().numpy()

    data_dict = {
        v: (("time", "lat", "lon"), np_arr[:, i]) for i, v in enumerate(data_vars)
    }
    ds = xr.Dataset(data_dict, coords=dict(coords))
    return ds


def get_checkpoint_path(extension='pth'):
    """
    Get the checkpoint path from the environment variable or use a default path.

    Returns:
        str: The checkpoint path.
    """
    # Check if the environment variable is set
    checkpoint_path = os.getenv("CHECKPOINT_PATH")
    if checkpoint_path is None:
        # Get the file path of the current function
        frame = inspect.currentframe()
        file_path = inspect.getfile(frame)

        # Get the absolute path
        abs_path = Path(os.path.abspath(file_path))
        # Set a default path if the environment variable is not set
        checkpoint_path = Path(abs_path.parent.parent) / "checkpoints/"
    else:
        # Convert to Path object if it's a string
        checkpoint_path = Path(checkpoint_path)
        # Ensure the path is absolute
        checkpoint_path = checkpoint_path.resolve()
    # Check if the path exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {checkpoint_path}")
    # Check if it's a directory
    if not checkpoint_path.is_dir():
        raise NotADirectoryError(
            f"Checkpoint path is not a directory: {checkpoint_path}")
        # Check if it contains checkpoint files
    if not any(checkpoint_path.rglob(f"*.{extension}")):
        raise FileNotFoundError(
            f"No checkpoint files found in: {checkpoint_path}")

    return checkpoint_path
