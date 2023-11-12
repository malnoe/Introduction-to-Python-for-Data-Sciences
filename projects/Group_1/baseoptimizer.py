from abc import ABC, abstractmethod

import numpy as np

from baseloss import LossFunction

class Optimizer(ABC):
    """
    Base class for optimizers.
    You should implement on your inheriting classes the step() method.
    Optimizers are callable and perform one step on their own function to optimize it.
    """
    def __init__(self, function: LossFunction, theta_0: np.ndarray) -> None:
        # You can use this state of the optimizer to store any kind of information you need
        # (maybe on past gradients ? on past values ?)
        self.state = {}

        # Here is the function that you want to optimize
        self.function = function

        # Remember the curent parameters
        # Start at:
        self.theta = theta_0

    def __call__(self, *args):
        self.step(*args)

    @abstractmethod
    def step(self, *args):
        pass
