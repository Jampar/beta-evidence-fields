"""Defines an individual range observation for an occupancy field."""

from dataclasses import dataclass

import numpy as np


@dataclass
class OccupancyObservation:
    """Represents a range observation contributing occupancy evidence."""

    origin          : np.ndarray
    direction       : np.ndarray
    range           : float
    returned        : bool
    time            : float
    evidence_weight : float = 1.0


    def __post_init__(self) -> None:
        """Validates the observation and normalizes its direction."""

        self.origin          = np.asarray(self.origin,    dtype = float)
        self.direction       = np.asarray(self.direction, dtype = float)
        self.range           = float(self.range)
        self.returned        = bool(self.returned)
        self.time            = float(self.time)
        self.evidence_weight = float(self.evidence_weight)

        if self.origin.ndim != 1:
            raise ValueError("origin must be one-dimensional")

        if self.direction.shape != self.origin.shape:
            raise ValueError("direction must have the same shape as origin")

        direction_norm = np.linalg.norm(self.direction)

        if direction_norm == 0.0:
            raise ValueError("direction must be non-zero")

        if self.range < 0.0:
            raise ValueError("range must be non-negative")

        self.direction = self.direction / direction_norm


    @property
    def state(self) -> np.ndarray:
        """Returns the lifted spatiotemporal observation state."""

        return np.concatenate((self.origin,
                               self.direction,
                               np.array([self.range, self.time])))


    @property
    def weight(self) -> float:
        """Returns the observation evidence weight."""

        return self.evidence_weight


    @property
    def endpoint(self) -> np.ndarray:
        """Returns the endpoint of the observation ray."""

        return self.origin + self.range*self.direction


    @property
    def dimension(self) -> int:
        """Returns the spatial dimension of the observation."""

        return len(self.origin)