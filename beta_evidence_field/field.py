import numpy as np

from .kernels import GaussianKernel
from .observation import OccupancyObservation
from .prior import OccupancyPrior
from .query import OccupancyQuery


DEFAULT_SUPPORT_SIGMA          = 3.0
DEFAULT_TEMPORAL_SUPPORT_SIGMA = 3.0


class OccupancyField:
    """Continuous Beta occupancy field constructed from range observations."""


    def __init__(self,
                 surface_covariance     : np.ndarray,
                 free_covariance        : np.ndarray,
                 temporal_variance      : float,
                 prior                  : OccupancyPrior,
                 support_sigma          : float = DEFAULT_SUPPORT_SIGMA,
                 temporal_support_sigma : float = DEFAULT_TEMPORAL_SUPPORT_SIGMA):

        surface_covariance = np.asarray(surface_covariance, dtype = float)
        free_covariance    = np.asarray(free_covariance,    dtype = float)

        if surface_covariance.ndim != 2:
            raise ValueError("surface_covariance must be a matrix")

        if surface_covariance.shape[0] != surface_covariance.shape[1]:
            raise ValueError("surface_covariance must be square")

        if free_covariance.shape != surface_covariance.shape:
            raise ValueError("free_covariance must match surface_covariance")

        if not np.all(np.isfinite(surface_covariance)):
            raise ValueError("surface_covariance must be finite")

        if not np.all(np.isfinite(free_covariance)):
            raise ValueError("free_covariance must be finite")

        if not np.isfinite(temporal_variance) or temporal_variance <= 0.0:
            raise ValueError("temporal_variance must be finite and positive")

        if not np.isfinite(support_sigma) or support_sigma <= 0.0:
            raise ValueError("support_sigma must be finite and positive")

        if (not np.isfinite(temporal_support_sigma) or temporal_support_sigma <= 0.0):
            raise ValueError("temporal_support_sigma must be finite and positive")

        self.dimension              = surface_covariance.shape[0]

        self.surface_kernel         = GaussianKernel(covariance = surface_covariance)
        self.free_kernel            = GaussianKernel(covariance = free_covariance)
        self.temporal_kernel        = GaussianKernel(covariance = np.array([[temporal_variance]]))

        self.prior                  = prior
        self.support_sigma          = float(support_sigma)
        self.temporal_support_sigma = float(temporal_support_sigma)

        self.observations           : list[OccupancyObservation] = []


    def add(self,
            observations : list[OccupancyObservation]) -> "list[OccupancyObservation]":
        """Adds observations acquired at the same time."""

        observations = list(observations)

        if not observations:
            return observations

        time = observations[0].time

        if not np.isfinite(time):
            raise ValueError("observation time must be finite")

        if self.observations and time < self.observations[-1].time:
            raise ValueError("observations must be added in chronological order")

        for observation in observations:

            if observation.dimension != self.dimension:
                raise ValueError(f"observation must be {self.dimension}-dimensional")

            if not np.isclose(observation.time, time):
                raise ValueError("observations added together must have the same time")

        self.observations.extend(observations)
        self._prune(time)

        return observations


    def query(self,
              points : np.ndarray,
              time   : float | None = None) -> OccupancyQuery:
        """Queries the occupancy field at one or more locations."""

        points, scalar  = self._prepare(points)

        alpha, beta     = self.prior.evaluate(points)
        alpha           = np.asarray(alpha, dtype = float)
        beta            = np.asarray(beta,  dtype = float)
        support         = np.zeros(len(points), dtype = bool)

        if alpha.shape != (len(points),):
            raise ValueError("prior alpha must have shape (n,)")

        if beta.shape != (len(points),):
            raise ValueError("prior beta must have shape (n,)")

        if not self.observations:
            if scalar:
                return OccupancyQuery(alpha   = float(alpha[0]),
                                      beta    = float(beta[0]),
                                      support = False)

            return OccupancyQuery(alpha   = alpha,
                                  beta    = beta,
                                  support = support)

        time = (self.observations[-1].time if time is None else float(time))

        if not np.isfinite(time):
            raise ValueError("time must be finite")

        observations = [observation for observation in self.observations
                        if observation.time <= time]

        if not observations:
            if scalar:
                return OccupancyQuery(alpha   = float(alpha[0]),
                                      beta    = float(beta[0]),
                                      support = False)

            return OccupancyQuery(alpha   = alpha,
                                  beta    = beta,
                                  support = support)

        origins    = np.asarray([o.origin          for o in observations], dtype = float)
        directions = np.asarray([o.direction       for o in observations], dtype = float)
        endpoints  = np.asarray([o.endpoint        for o in observations], dtype = float)
        returned   = np.asarray([o.returned        for o in observations], dtype = bool)
        times      = np.asarray([o.time            for o in observations], dtype = float)
        weights    = np.asarray([o.evidence_weight for o in observations], dtype = float)


        weights        *= self.temporal_kernel.evaluate(points = times[:, None],   
                                                        centre = np.array([time]))

        free_evidence   = self.free_kernel.convolve_line(points = points[None, :, :],
                                                         start  = origins[:, None, :],
                                                         end    = endpoints[:, None, :])

        precision       = np.einsum("ri,ij,rj->r",
                                    directions,
                                    self.free_kernel.precision,
                                    directions,
                                    optimize = True)

        beta           += np.sum((weights * np.sqrt(precision / (2.0*np.pi)))[:, None] * free_evidence, axis = 0)

        if np.any(returned):
            endpoints        = endpoints[returned]
            weights          = weights[returned]

            surface_evidence = self.surface_kernel.evaluate(points = points[None, :, :],
                                                            centre = endpoints[:, None, :])
            
            surface_distance = self.surface_kernel.distance(points = points[None, :, :], 
                                                            centre = endpoints[:, None, :])
            
            alpha   += np.sum(weights[:, None] * surface_evidence,          axis = 0)
            support  = np.any(surface_distance <= self.support_sigma**2,    axis = 0)


        if scalar:
            return OccupancyQuery(alpha   = float(alpha[0]),
                                  beta    = float(beta[0]),
                                  support = bool(support[0]))

        return OccupancyQuery(alpha   = alpha,
                              beta    = beta,
                              support = support)


    def _prepare(self,
                 points : np.ndarray) -> tuple[np.ndarray, bool]:

        points = np.asarray(points, dtype = float)
        scalar = points.ndim == 1

        if scalar:
            points = points[None, :]

        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError(
                f"points must have shape ({self.dimension},) "
                f"or (n, {self.dimension})")

        if not np.all(np.isfinite(points)):
            raise ValueError("points must be finite")

        return points, scalar


    def _prune(self,
               time : float) -> None:

        times    = np.asarray([observation.time
                               for observation in self.observations],
                               dtype = float)

        distance = self.temporal_kernel.distance(points = times[:, None],
                                                 centre = np.array([time]))

        retained = distance <= self.temporal_support_sigma**2

        self.observations = [observation 
                             for observation, keep in zip(self.observations, retained)
                             if keep]


    def clear(self) -> None:
        """Removes all observations from the field."""

        self.observations.clear()


    def __len__(self) -> int:
        """Returns the number of currently retained observations."""

        return len(self.observations)