import numpy as np


def fit_straight_line(xs: np.ndarray, ys: np.ndarray, step: int=1) -> list:
    N = xs.shape[0] - 1
    dx = (xs[-1] - xs[0]) / N
    dy = (ys[-1] - ys[0]) / N
    ds = np.hypot(dx, dy) / N
    yaw = np.arctan2(dy, dx)
    k = 999999.99

    curve = [(xs[0]+dx*i, ys[0]+dy*i, yaw, k, ds*i) for i in np.arange(0, N+1, step)]
    return curve