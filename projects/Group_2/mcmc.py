from abc import ABC, abstractmethod
from typing import Callable, Tuple
import numpy as np

class Mcmc(ABC):
    def __init__(
            self,
            first_distrib: Callable[[], np.ndarray],
            target_distrib: Callable[[np.ndarray], float]
        ) -> None:
        """
        Constructor

        Parameters
        ----------
        first_distrib : Callable[[], np.ndarray]
            The first distribution to sample from, as a function of no arguments
            returns a numpy array for theta_0
        target_distrib : Callable[[np.ndarray], float]
            This function should return the posterior
            up to the scaling constant (the "partition function", here the evidence),
            as a float
        """
        self.theta = first_distrib()
        self.target = target_distrib

    def step(self) -> Tuple[np.ndarray, int]:
        """
        A stepping funcion to evaluate candidate states until a good one is reached
        """
        accepted = False
        n_tries = 1
        while not accepted:
            proposal = self.proposal()
            acceptance_proba = self.acceptance(proposal)
            if acceptance_proba > np.random.rand():
                self.theta = proposal
                accepted = True
            else:
                accepted = False
                n_tries += 1
        return self.theta, n_tries

    @abstractmethod
    def proposal(self) -> np.ndarray:
        """
        This you should write yourselves in your child classes:
        it is the proposal function, g, producing candidate states,
        depending on the current state
        """
        pass

    @abstractmethod
    def acceptance(self, proposal) -> float:
        """
        This you should write yourselves in your child classes:
        it is the acceptance function, alpha, producing the acceptance probability,
        depending on the current state and the proposal state.
        """
        pass
