import numpy as np


DEFAULT_PRIOR_VALUE = 5.0


class OccupancyPrior:
    """Defines the prior Beta parameters of an occupancy field."""


    def __init__(self,
                 alpha : float = DEFAULT_PRIOR_VALUE,
                 beta  : float = DEFAULT_PRIOR_VALUE):

        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be finite and positive")

        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be finite and positive")

        self.alpha = float(alpha)
        self.beta  = float(beta)


    def evaluate(self,
                 points : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates the prior Beta parameters at the supplied points.

        Args:
            points: Query locations with shape `(n, dimension)`.

        Returns:
            Prior alpha and beta parameters at each query location.
        """

        count = len(points)

        alpha = np.full(count, self.alpha, dtype = float)
        beta  = np.full(count, self.beta,  dtype = float)

        return alpha, beta