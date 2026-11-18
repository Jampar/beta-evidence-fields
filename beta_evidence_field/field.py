import numpy as np

from .kernels import GaussianKernel
from .observation import OccupancyObservation
from .query import OccupancyQuery


DEFAULT_PRIOR_VALUE   = 1.0
DEFAULT_SUPPORT_SIGMA = 3.0


class OccupancyField:
    """Continuous Beta occupancy field constructed from range observations."""


    def __init__(self,
                 surface_covariance : np.ndarray,
                 free_covariance    : np.ndarray,
                 temporal_variance  : float,
                 occupancy_prior    : float = DEFAULT_PRIOR_VALUE,
                 free_prior         : float = DEFAULT_PRIOR_VALUE,
                 support_sigma      : float = DEFAULT_SUPPORT_SIGMA):

        surface_covariance = np.asarray(surface_covariance, dtype = float)
        free_covariance    = np.asarray(free_covariance,    dtype = float)

        if surface_covariance.ndim != 2:
            raise ValueError("surface_covariance must be a matrix")

        if surface_covariance.shape[0] != surface_covariance.shape[1]:
            raise ValueError("surface_covariance must be square")

        if free_covariance.shape != surface_covariance.shape:
            raise ValueError("free_covariance must match surface_covariance")

        if temporal_variance <= 0.0:
            raise ValueError("temporal_variance must be positive")

        if occupancy_prior <= 0.0:
            raise ValueError("occupancy_prior must be positive")

        if free_prior <= 0.0:
            raise ValueError("free_prior must be positive")

        if support_sigma <= 0.0:
            raise ValueError("support_sigma must be positive")

        self.dimension = surface_covariance.shape[0]

        self.surface_kernel  = GaussianKernel(covariance = surface_covariance)
        self.free_kernel     = GaussianKernel(covariance = free_covariance)
        self.temporal_kernel = GaussianKernel(
            covariance = np.array([[temporal_variance]]))

        self.occupancy_prior = float(occupancy_prior)
        self.free_prior      = float(free_prior)
        self.support_sigma   = float(support_sigma)

        self.batches : list[list[OccupancyObservation]] = []


    @property
    def observations(self) -> list[OccupancyObservation]:
        """Returns all observations in acquisition order."""

        return [observation
                for batch in self.batches
                for observation in batch]


    @property
    def time(self) -> float | None:
        """Returns the latest observation time."""

        return self.batches[-1][0].time if self.batches else None


    @property
    def prior(self) -> OccupancyQuery:
        """Returns the unsupported scalar prior."""

        return OccupancyQuery(alpha   = self.occupancy_prior,
                              beta    = self.free_prior,
                              support = False)


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

        return points, scalar


    def _prior(self,
               count  : int,
               scalar : bool) -> OccupancyQuery:

        if scalar:
            return self.prior

        return OccupancyQuery(
            alpha   = np.full(count,
                              self.occupancy_prior,
                              dtype = float),
            beta    = np.full(count,
                              self.free_prior,
                              dtype = float),
            support = np.zeros(count,
                               dtype = bool))


    def _observations(self,
                      time : float) -> tuple[np.ndarray,
                                            np.ndarray,
                                            np.ndarray,
                                            np.ndarray,
                                            np.ndarray,
                                            np.ndarray]:

        observations = [observation
                        for batch in self.batches
                        if batch[0].time <= time
                        for observation in batch]

        origins = np.asarray(
            [observation.origin
             for observation in observations],
            dtype = float)

        directions = np.asarray(
            [observation.direction
             for observation in observations],
            dtype = float)

        ranges = np.asarray(
            [observation.range
             for observation in observations],
            dtype = float)

        returned = np.asarray(
            [observation.returned
             for observation in observations],
            dtype = bool)

        times = np.asarray(
            [observation.time
             for observation in observations],
            dtype = float)

        weights = np.asarray(
            [observation.evidence_weight
             for observation in observations],
            dtype = float)

        return origins, directions, ranges, returned, times, weights


    def add(self,
            observations : OccupancyObservation |
                           list[OccupancyObservation]
            ) -> list[OccupancyObservation]:
        """Adds one observation or one simultaneous observation batch."""

        observations = ([observations]
                        if isinstance(observations, OccupancyObservation)
                        else list(observations))

        if not observations:
            return observations

        time = observations[0].time

        for observation in observations:

            if observation.dimension != self.dimension:
                raise ValueError(
                    f"observation must be {self.dimension}-dimensional")

            if not np.isclose(observation.time, time):
                raise ValueError(
                    "observations added together must have the same time")

        self.batches.append(observations)

        return observations


    def query(self,
              points : np.ndarray,
              time   : float | None = None) -> OccupancyQuery:
        """Queries the occupancy field at one or more spatial points.

        Args:
            points: One point with shape `(dimension,)`, or multiple points
                with shape `(n, dimension)`.
            time: Query time. The latest observation time is used when omitted.

        Returns:
            Beta occupancy parameters and observation support at the query
            points.
        """

        points, scalar = self._prepare(points)
        time           = self.time if time is None else float(time)

        if time is None:
            return self._prior(len(points), scalar)

        origins, directions, ranges, returned, times, weights = (
            self._observations(time))

        if not len(origins):
            return self._prior(len(points), scalar)

        endpoints = origins + ranges[:, None]*directions


        temporal_weights = self.temporal_kernel.evaluate(
            points = times[:, None],
            centre = np.array([time]))

        weights = weights*temporal_weights


        precision = np.einsum("ri,ij,rj->r",
                            directions,
                            self.free_kernel.precision,
                            directions,
                            optimize = True)

        free_evidence = self.free_kernel.convolve_line(
            points = points[None, :, :],
            start  = origins[:, None, :],
            end    = endpoints[:, None, :])

        free_weights = (
            weights*np.sqrt(precision/(2.0*np.pi)))

        beta = (
            self.free_prior
            + np.sum(free_weights[:, None]*free_evidence,
                    axis = 0))


        surface_evidence = self.surface_kernel.evaluate(
            points = points[None, :, :],
            centre = endpoints[:, None, :])

        surface_weights = weights*returned

        alpha = (
            self.occupancy_prior
            + np.sum(surface_weights[:, None]*surface_evidence,
                    axis = 0))


        surface_distance = self.surface_kernel.distance(
            points = points[None, :, :],
            centre = endpoints[:, None, :])

        support = np.any(
            returned[:, None]
            & (surface_distance <= self.support_sigma**2),
            axis = 0)


        if scalar:

            return OccupancyQuery(alpha   = float(alpha[0]),
                                beta    = float(beta[0]),
                                support = bool(support[0]))

        return OccupancyQuery(alpha   = alpha,
                            beta    = beta,
                            support = support)


    def clear(self) -> None:
        """Removes all observations from the field."""

        self.batches.clear()


    def __len__(self) -> int:
        """Returns the number of observations in the field."""

        return sum(len(batch) for batch in self.batches)