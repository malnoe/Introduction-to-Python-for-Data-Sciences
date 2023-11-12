from typing import Dict, Any
import numpy as np
import scipy.stats as scs
from scipy.stats._distn_infrastructure import rv_continuous_frozen

COMPANY_SIZE = 10000
AC_POLICY = 23
MAX_TEMP_INCREASE_PER_UNIT_TIME = 10

def _assert_distribs(distribs: Dict[str, Any]) -> Dict[str, rv_continuous_frozen]:
    for d in distribs.values():
        assert isinstance(d, rv_continuous_frozen)
    return distribs

class RoomTempDistrib:
    def __init__(self, seed: int=666, **params) -> None:
        self.rng = np.random.default_rng(seed)
        sigma_star = np.random.rand() * 1e-2
        lam_star = np.random.rand() * 1e-3
        p_star = np.random.rand() * .5
        kappa_star = np.random.rand() * 30
        # DO NOT TOUCH, IT IS SUPPOSED TO BE A SECRET
        self._true_distribs = {
            "t": scs.norm(loc=AC_POLICY, scale=sigma_star),
            "n": scs.binom(n=COMPANY_SIZE, p=lam_star),
            "w": scs.uniform(loc=0, scale=p_star),
            "epsilon": scs.expon(scale=1. / kappa_star)
        }
        self.distribs = _assert_distribs({
            "t": scs.norm(loc=AC_POLICY, scale=params["sigma"]),
            "n": scs.norm(
                loc=params["lam"]*COMPANY_SIZE,
                scale=np.sqrt(params["lam"] * (1. - params["lam"]) * COMPANY_SIZE)
                ),
            "w": scs.uniform(loc=0, scale=params["p"]),
            "epsilon": scs.expon(scale=1. / params["kappa"])
        })

    def _sample_true_prior(self):
        t = self._true_distribs["t"].rvs(random_state=self.rng)
        n = self._true_distribs["n"].rvs(random_state=self.rng)
        w = self._true_distribs["w"].rvs(random_state=self.rng)
        epsilon = self._true_distribs["epsilon"].rvs(random_state=self.rng)
        return np.array([t, n, w, epsilon])

    def _sample_true_posterior(self, time: float) -> float:
        return self.transform(self._sample_true_prior(), time)

    def transform(self, theta, time: float) -> float:
        return theta[0] + .1 * time * min(MAX_TEMP_INCREASE_PER_UNIT_TIME, theta[3] * (.1 * theta[1] - theta[2]))

    def likelihood(self, theta, measures) -> float:
        """
        The likelihood up to a scaling constant
        """
        T_hat = np.array([self.transform(theta, tau) for tau in range(len(measures))])
        return (-1e4 * (T_hat - measures) ** 2).exp()

    def prior(self, theta):
        return np.prod([
            self.distribs["t"].pdf(theta[0]),
            self.distribs["n"].pdf(theta[1]),
            self.distribs["w"].pdf(theta[2]),
            self.distribs["epsilon"].pdf(theta[3])
        ])


