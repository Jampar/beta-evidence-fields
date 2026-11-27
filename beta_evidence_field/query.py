"""Defines query results from an occupancy field."""

from dataclasses import dataclass

import numpy as np


@dataclass
class OccupancyQuery:
    """Represents the Beta distribution returned by an occupancy query."""

    alpha          : np.ndarray | float
    beta           : np.ndarray | float
    alpha_gradient : np.ndarray 
    beta_gradient  : np.ndarray
    support        : np.ndarray | bool


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

        return ((self.alpha*self.beta)
                / (concentration**2*(concentration + 1.0)))


    @property
    def gradient(self) -> np.ndarray:
        """Returns the spatial gradient of the expected occupancy."""

        concentration = np.asarray(self.concentration)

        return (np.asarray(self.beta)[..., None]*self.alpha_gradient  - 
                np.asarray(self.alpha)[..., None]*self.beta_gradient) / concentration[..., None]**2