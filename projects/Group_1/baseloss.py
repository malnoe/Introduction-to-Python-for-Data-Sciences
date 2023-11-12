from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

class LossFunction(ABC):
    """
    Base class for loss functions
    """
    def __init__(self):
        self.value: Optional[float] = None
        self.grad: Optional[np.ndarray] = None

    def __call__(self, theta: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Loss functions should be callable, e.g.:
        ```
        >>> theta = np.array([1, 2, 3])
        >>> loss = MyLossFunction(my_parameters)
        >>> print(loss(theta))
        -49.3
        ```
        """
        self.value = self.value_oracle(theta)
        self.grad = self.grad_oracle(theta)
        return float(self.value), self.grad

    @abstractmethod
    def value_oracle(self, theta: np.ndarray) -> float:
        """
        Oracle for the value of the loss function
        """
        pass

    @abstractmethod
    def grad_oracle(self, theta: np.ndarray) -> np.ndarray:
        """
        Oracle for the gradient of the loss function
        """
        pass
