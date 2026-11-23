import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # Forward pass
        z = x @ w + b
        y_hat = 1 / (1 + np.exp(-z))

        # Chain rule
        dL_dy = y_hat - y_true
        dy_dz = y_hat * (1 - y_hat)
        dz = dL_dy * dy_dz

        dL_dw = dz * x
        dL_db = dz

        return np.round(dL_dw, 5), round(float(dL_db), 5)