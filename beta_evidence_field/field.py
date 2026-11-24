import numpy as np
import yaml

from .kernels       import GaussianKernel
from .observation   import OccupancyObservation
from .prior         import OccupancyPrior
from .query         import OccupancyQuery
from ._validation   import require


DEFAULT_SUPPORT_SIGMA           = 3.0
DEFAULT_TEMPORAL_SUPPORT_SIGMA  = 3.0


class OccupancyField:
    """Continuous Beta occupancy field constructed from range observations."""


    @classmethod
    def from_yaml(cls,
                  path : str) -> "OccupancyField":
        """Constructs an occupancy field from a YAML configuration."""

        with open(path, "r", encoding = "utf-8") as file:
            config = yaml.safe_load(file)

        surface  = config["surface"]
        free     = config["free"]
        state    = config["state"]
        prior    = config["prior"]
        support  = config.get("support", {})

        return cls(surface_covariance           = np.asarray(surface["covariance"], dtype = float),
                   free_covariance              = np.asarray(free["covariance"],    dtype = float),
                   state_covariance             = np.asarray(state["covariance"],   dtype = float),

                   prior                        = OccupancyPrior(alpha = float(prior["alpha"]),
                                                                 beta  = float(prior["beta"])),

                   support_sigma                = float(support.get("sigma",                DEFAULT_SUPPORT_SIGMA)),
                   temporal_support_sigma       = float(state.get("temporal_support_sigma", DEFAULT_TEMPORAL_SUPPORT_SIGMA)))


    def __init__(self,
                 surface_covariance         : np.ndarray,
                 free_covariance            : np.ndarray,
                 state_covariance           : np.ndarray,
                 prior                      : OccupancyPrior,
                 support_sigma              : float = DEFAULT_SUPPORT_SIGMA,
                 temporal_support_sigma     : float = DEFAULT_TEMPORAL_SUPPORT_SIGMA):

        surface_covariance = np.asarray(surface_covariance, dtype = float)
        free_covariance    = np.asarray(free_covariance,    dtype = float)
        state_covariance   = np.asarray(state_covariance,   dtype = float)

        require(surface_covariance.ndim == 2,                               
                ValueError, 
                "surface_covariance must be a matrix")
        
        require(surface_covariance.shape[0] == surface_covariance.shape[1], 
                ValueError, 
                "surface_covariance must be square")

        require(free_covariance.shape == surface_covariance.shape,
                ValueError,
                "free_covariance must match surface_covariance")

        require(state_covariance.ndim == 2,
                ValueError,
                "state_covariance must be a matrix")

        dimension       = surface_covariance.shape[0]

        
        require(state_covariance[-1, -1] > 0.0,
                ValueError,
                "state temporal variance must be positive")

        require(state_covariance.shape[0] == state_covariance.shape[1],
                ValueError,
                "state_covariance must be square")
        
        require(np.all(np.isfinite(surface_covariance)),
                ValueError,
                "surface_covariance must be finite")

        require(np.all(np.isfinite(free_covariance)),
                ValueError,
                "free_covariance must be finite")

        require(np.all(np.isfinite(state_covariance)),
                ValueError,
                "state_covariance must be finite")

        require(np.allclose(state_covariance[-1, :-1], 0.0) and np.allclose(state_covariance[:-1, -1], 0.0),
                ValueError,
                "state_covariance must not couple time with origin or direction")

        require(np.isfinite(support_sigma) and support_sigma > 0.0,
                ValueError,
                "support_sigma must be finite and positive")

        require(np.isfinite(temporal_support_sigma ) and temporal_support_sigma  > 0.0,
                ValueError,
                "temporal_support_sigma must be finite and positive")

        self.dimension                  = dimension

        self.surface_kernel             = GaussianKernel(covariance = surface_covariance)
        self.free_kernel                = GaussianKernel(covariance = free_covariance)
        self.state_kernel               = GaussianKernel(covariance = state_covariance)

        self.prior                      = prior

        self.support_sigma              = float(support_sigma)
        self.temporal_support_sigma     = float(temporal_support_sigma )

        self.temporal_variance          = float(state_covariance[-1, -1])

        self.observations               : list[OccupancyObservation] = []



    def add(self,
            observations : list[OccupancyObservation]) -> "list[OccupancyObservation]":
        """Adds observations acquired at the same time."""

        observations = list(observations)

        if not observations:
            return observations

        time = observations[0].time

        require(np.isfinite(time),
                ValueError,
                "observation time must be finite")

        require(not self.observations or time >= self.observations[-1].time,
                ValueError,
                "observations must be added in chronological order")

        for observation in observations:

                require(observation.dimension == self.dimension,
                        ValueError,
                        f"observation must be {self.dimension}-dimensional")

                require(len(observation.state) == self.state_kernel.dimension,
                        ValueError,
                        "observation state must match state covariance")

                require(np.isclose(observation.time, time),
                        ValueError,
                        "observations added together must have the same time")
                        
        self._prune(time)
        self.observations.extend(observations)

        return observations

    
    def evidence(self,
                 observations : list[OccupancyObservation],
                 points       : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluates occupied and free evidence from the supplied observations."""

        origins             = np.asarray([o.origin    for o in observations], dtype = float)
        directions          = np.asarray([o.direction for o in observations], dtype = float)
        endpoints           = np.asarray([o.endpoint  for o in observations], dtype = float)
        returned            = np.asarray([o.returned  for o in observations], dtype = bool)
        observation_weights = np.asarray([o.weight for o in observations],    dtype = float)

        redundancy_weights  = self._redundancy_weights(observations) 
        weights             = observation_weights * redundancy_weights

        surface_evidence    = self.surface_kernel.evaluate(points = points[None, :, :],
                                                           centre = endpoints[:, None, :])

        surface_evidence   *= returned[:, None]
        free_evidence       = self.free_kernel.convolve_line(points = points[None, :, :],
                                                             start  = origins[:, None, :],
                                                             end    = endpoints[:, None, :])

        precision           = np.einsum("ri,ij,rj->r",
                                        directions,
                                        self.free_kernel.precision,
                                        directions,
                                        optimize = True)

        free_evidence      *= np.sqrt(precision/(2.0*np.pi))[:, None]
        free_evidence      *= 1.0 - surface_evidence

        alpha               = np.sum(weights[:, None] * surface_evidence, axis = 0)
        beta                = np.sum(weights[:, None] * free_evidence, axis = 0)

        if np.any(returned):
            surface_distance = self.surface_kernel.distance(points = points[None, :, :],
                                                            centre = endpoints[:, None, :])

            support = np.any(returned[:, None] & (surface_distance <= self.support_sigma**2), axis = 0)

        else:
            support = np.zeros(len(points), dtype = bool)

        return alpha, beta, support


    def query(self,
              points : np.ndarray,
              time   : float | None = None) -> OccupancyQuery:
        """Queries the occupancy field at one or more locations."""

        points, scalar = self._prepare(points)

        alpha, beta = self.prior.evaluate(points)

        alpha = np.asarray(alpha, dtype = float)
        beta  = np.asarray(beta,  dtype = float)

        require(alpha.shape == (len(points),),
                ValueError,
                "prior alpha must have shape (n,)")

        require(beta.shape == (len(points),),
                ValueError,
                "prior beta must have shape (n,)")

        support = np.zeros(len(points), dtype = bool)

        if self.observations:

            time = self.observations[-1].time if time is None else float(time)

            require(np.isfinite(time),
                    ValueError,
                    "time must be finite")

            observations = [observation
                            for observation in self.observations
                            if observation.time <= time]

            if observations:
                alpha_evidence, beta_evidence, support = self.evidence(observations = observations,
                                                                       points       = points)

                alpha += alpha_evidence
                beta  += beta_evidence

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

        require(points.ndim == 2 and points.shape[1] == self.dimension,
                ValueError,
                f"points must have shape ({self.dimension},) or (n, {self.dimension})")

        require(np.all(np.isfinite(points)),
                ValueError,
                "points must be finite")

        return points, scalar


    def _redundancy_weights(self,
                            observations : list[OccupancyObservation]) -> np.ndarray:

        if not observations:
                return np.empty(0, dtype = float)

        states     = np.asarray([observation.state
                                for observation in observations],
                                dtype = float)

        distance   = self.state_kernel.distance(points = states[:, None, :],
                                                centre = states[None, :, :])

        similarity = np.exp(-0.5 * distance)
        density    = np.sum(similarity, axis = 1)

        return 1.0 / density


    def _prune(self,
               time : float) -> None:

        times             = np.asarray([observation.time 
                                        for observation in self.observations], 
                                        dtype = float)
        
        temporal_distance = (times - time)**2 / self.temporal_variance
        retained          = temporal_distance <= self.temporal_support_sigma**2

        self.observations = [observation
                             for observation, keep in zip(self.observations, retained)
                             if keep]


    def clear(self) -> None:
        """Removes all observations from the field."""

        self.observations.clear()


    def __len__(self) -> int:
        """Returns the number of currently retained observations."""

        return len(self.observations)