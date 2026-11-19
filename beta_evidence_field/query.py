"""Defines query results from an occupancy field."""

from dataclasses import dataclass

import numpy as np


@dataclass
class OccupancyQuery:
    """Represents the Beta distribution returned by an occupancy query.

    Attributes:
        alpha: Occupancy evidence parameter of the Beta distribution.
        beta: Free-space evidence parameter of the Beta distribution.
        support: Whether the queried location is supported by observations.
    """

    alpha   : np.ndarray | float
    beta    : np.ndarray | float
    support : np.ndarray | bool


    @property
    def concentration(self) -> np.ndarray | float:
        """Returns the total evidence concentration."""

        return self.alpha + self.beta


    @property
    def mean(self) -> np.ndarray | float:
        """Returns the expected occupancy probability."""

        return self.alpha/self.concentration


    @property
    def variance(self) -> np.ndarray | float:
        """Returns the variance of the occupancy probability."""

        concentration = self.concentration

        return ((self.alpha*self.beta) / (concentration**2*(concentration + 1.0)))