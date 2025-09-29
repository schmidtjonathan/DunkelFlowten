def spatial_window_averaging(X, win_size):
    # print("spat", X.shape)
    X = X.unfold(2, win_size, win_size).unfold(3, win_size, win_size)
    # print("spat", X.shape)
    X = X.mean(dim=(4, 5))
    # print("spat", X.shape)
    return X


def nearest_neighor_ds(X, factor):
    # return torch.nn.functional.interpolate(
    #     X,
    #     scale_factor=(1 / factor, 1 / factor),
    #     mode="nearest",
    #     # align_corners=True,
    #     # antialias=True,
    # )
    return X[..., ::factor, ::factor]


def temporal_window_averaging(X, win_size):
    X = X.unfold(0, win_size, win_size)
    # print("temp", X.shape)
    X = X.mean(dim=-1)
    # print("temp", X.shape)
    return X
