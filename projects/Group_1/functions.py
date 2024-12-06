from typing import List, Union
import numpy as np

from baseloss import LossFunction


class Function1(LossFunction):
    def __init__(self, A: np.ndarray, b: np.ndarray):
        super().__init__()
        self.A = A
        self.b = b

    def value_oracle(self, theta: np.ndarray) -> float:
        return float(.5 * theta.T @ self.A @ theta + self.b @ theta)

    def grad_oracle(self, theta: np.ndarray) -> np.ndarray:
        return self.A @ theta + self.b

    def hessian_oracle(self, _theta: np.ndarray) -> np.ndarray:
        return self.A


class Function2(LossFunction):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__()
        self.X = X
        self.y = y
        assert X.shape[0] == y.shape[0]

    def value_oracle(self, theta: np.ndarray) -> float:
        pred = self.X @ theta
        error = pred - self.y
        return .5 * np.dot(error, error)

    def grad_oracle(self, theta: np.ndarray) -> np.ndarray:
        assert self.X.shape[1] == theta.shape[0]
        random_index = np.random.randint(self.X.shape[0])
        random_point = self.X[random_index, :]
        random_label = self.y[random_index]
        return random_point * (random_point @ theta - random_label)

    def batched_value_oracle(
        self,
        indexing: Union[int, List[int], np.ndarray],
        theta: np.ndarray
    ):
        assert np.max(indexing) < self.X.shape[0]
        assert np.min(indexing) >= 0
        pred = self.X[indexing, :] @ theta
        error = pred - self.y[indexing]
        return .5 * np.dot(error, error)

    def batched_grad_oracle(
        self,
        indexing: Union[int, List[int], np.ndarray],
        theta: np.ndarray
    ):
        assert np.max(indexing) < self.X.shape[0]
        assert np.min(indexing) >= 0
        selected_points = self.X[indexing, :]
        pred = selected_points @ theta
        error = pred - self.y[indexing]
        return selected_points * error
