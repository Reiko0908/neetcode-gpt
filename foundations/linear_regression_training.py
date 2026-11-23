import numpy as np
from numpy.typing import NDArray


class Solution:
    lr = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Perform Linear Regression

        for _ in range(num_iterations):
            y_pred = X @ initial_weights
            loss = Y - y_pred
            grad = -2 * X.T / len(X)
            initial_weights -= self.lr * grad @ loss
        return np.round(initial_weights, 5)
