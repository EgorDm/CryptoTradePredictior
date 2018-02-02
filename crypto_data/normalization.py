import numpy as np


def normalize_windows(windows: np.ndarray) -> np.ndarray:
    ret = np.zeros_like(windows)
    ret[:, 1:, :] = windows[:, 1:, :] / windows[:, 0:1, :] - 1
    return ret
