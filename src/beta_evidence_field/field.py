import numpy as np

from .kernels import GaussianKernel
from .observation import OccupancyObservation
from .query import OccupancyQuery


DEFAULT_PRIOR_VALUE   = 1.0
DEFAULT_SUPPORT_SIGMA = 3.0


class OccupancyField:


    def __init__(self,
                 surface_covariance : np.ndarray,
                 free_covariance    : np.ndarray,
                 temporal_variance  : float,
                 occupancy_prior    : float = DEFAULT_PRIOR_VALUE,
                 free_prior         : float = DEFAULT_PRIOR_VALUE):

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

        self.dimension = surface_covariance.shape[0]

        self.surface_kernel  = GaussianKernel(covariance = surface_covariance)
        self.free_kernel     = GaussianKernel(covariance = free_covariance)
        self.temporal_kernel = GaussianKernel(covariance = np.array([[temporal_variance]]))

        self.occupancy_prior = float(occupancy_prior)
        self.free_prior      = float(free_prior)

        self.batches : list[list[OccupancyObservation]] = []


    @property
    def observations(self) -> list[OccupancyObservation]:
        return [observation
                for batch in self.batches
                for observation in batch]


    @property
    def time(self) -> float | None:
        return self.batches[-1][0].time if self.batches else None


    @property
    def prior(self) -> OccupancyQuery:
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


    def add(self,
            observations : OccupancyObservation | 
                           list[OccupancyObservation]) -> list[OccupancyObservation]:

        observations = ([observations]
                        if isinstance(observations, OccupancyObservation)
                        else list(observations))

        if not observations:
            return observations

        time = observations[0].time

        for observation in observations:

            if observation.dimension != self.dimension:
                raise ValueError(f"observation must be {self.dimension}-dimensional")

            if observation.time != time:
                raise ValueError("observations added together must have the same time")

        self.batches.append(observations)

        return observations


    def evidence(self,
                 points       : np.ndarray,
                 observations : list[OccupancyObservation]
                 ) -> tuple[np.ndarray | float,
                            np.ndarray | float]:

        points, scalar = self._prepare(points)

        surface = np.zeros(len(points), dtype = float)
        free    = np.zeros(len(points), dtype = float)

        for observation in observations:

            precision = (observation.direction
                         @ self.free_kernel.precision
                         @ observation.direction)

            free += (
                observation.evidence_weight
                * np.sqrt(precision/(2.0*np.pi))
                * self.free_kernel.convolve_line(
                    points = points,
                    start  = observation.origin,
                    end    = observation.endpoint))

            if observation.returned:

                surface += (
                    observation.evidence_weight
                    * self.surface_kernel.evaluate(
                        points = points,
                        centre = observation.endpoint))

        if scalar:
            return surface[0], free[0]

        return surface, free


    def support(self,
                points       : np.ndarray,
                observations : list[OccupancyObservation],
                sigma        : float = DEFAULT_SUPPORT_SIGMA
                ) -> np.ndarray | bool:

        points, scalar = self._prepare(points)
        support        = np.zeros(len(points), dtype = bool)

        for observation in observations:

            if observation.returned:

                support |= (
                    self.surface_kernel.distance(
                        points = points,
                        centre = observation.endpoint)
                    <= sigma**2)

        return bool(support[0]) if scalar else support


    def parameters(self,
                   surface_history : np.ndarray,
                   free_history    : np.ndarray,
                   support_history : np.ndarray,
                   batches         : list[list[OccupancyObservation]],
                   time            : float) -> OccupancyQuery:

        times = np.asarray([batch[0].time for batch in batches],
                           dtype = float)

        temporal_weights = self.temporal_kernel.evaluate(
            points = times[:, None],
            centre = np.array([float(time)]))

        alpha = (
            self.occupancy_prior
            + np.tensordot(temporal_weights,
                           surface_history,
                           axes = (0, 0)))

        beta = (
            self.free_prior
            + np.tensordot(temporal_weights,
                           free_history,
                           axes = (0, 0)))

        return OccupancyQuery(
            alpha   = alpha,
            beta    = beta,
            support = np.any(support_history, axis = 0))


    def query(self,
              point : np.ndarray,
              time  : float | None = None) -> OccupancyQuery:

        point, scalar = self._prepare(point)

        if not scalar:
            raise ValueError("query expects a single point")

        time = self.time if time is None else float(time)

        if time is None:
            return self.prior

        batches = [batch
                   for batch in self.batches
                   if batch[0].time <= time]

        if not batches:
            return self.prior

        evidence = [self.evidence(points       = point[0],
                                  observations = batch)
                    for batch in batches]

        support = [self.support(points       = point[0],
                                observations = batch)
                   for batch in batches]

        surface_history, free_history = map(np.asarray, zip(*evidence))

        return self.parameters(
            surface_history = surface_history,
            free_history    = free_history,
            support_history = np.asarray(support),
            batches         = batches,
            time            = time)


    def clear(self) -> None:
        self.batches.clear()


    def __len__(self) -> int:
        return sum(len(batch) for batch in self.batches)

