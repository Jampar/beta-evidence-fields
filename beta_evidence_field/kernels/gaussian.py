"""Gaussian kernel operations for continuous evidence fields."""

from scipy.linalg  import solve_triangular
from scipy.spatial import cKDTree
from scipy.special import erf

from .._validation import require, require_all

import numpy as np


class GaussianKernel:


    def __init__(self,
                 covariance : np.ndarray):

        covariance = np.asarray(covariance, dtype = float)

        require(covariance.ndim == 2, ValueError, "Covariance must be a matrix.")
        require_all([(covariance.shape[0] == covariance.shape[1], ValueError, "Covariance must be square."),
                    (np.allclose(covariance, covariance.T),      ValueError, "Covariance must be symmetric.")])

        try:
            cholesky = np.linalg.cholesky(covariance)

        except np.linalg.LinAlgError as error:
            raise ValueError(
                "Covariance must be positive definite.") from error

        self.covariance = covariance
        self.precision  = np.linalg.inv(covariance)
        self.cholesky   = cholesky


    @property
    def dimension(self) -> int:
        return self.covariance.shape[0]


    def sparse_distance(self,
                        points        : np.ndarray,
                        support_sigma : float):

        points = np.asarray(points, dtype = float)

        require(points.ndim == 2,                  ValueError, "Points must be a matrix.")
        require(points.shape[1] == self.dimension, ValueError, "Point dimension does not match covariance.")

        normalised = solve_triangular(self.cholesky,
                                      points.T,
                                      lower = True).T

        tree     = cKDTree(normalised)
        distance = tree.sparse_distance_matrix(tree,
                                               max_distance = support_sigma,
                                               output_type  = "coo_matrix")

        distance.data **= 2

        return distance


    def distance(self,
                 points : np.ndarray,
                 centre : np.ndarray) -> np.ndarray:

        points = np.asarray(points, dtype = float)
        centre = np.asarray(centre, dtype = float)

        require_all([(points.shape[-1] == self.dimension, ValueError, "Point dimension does not match covariance."),
                     (centre.shape[-1] == self.dimension, ValueError, "Centre dimension does not match covariance.")])

        difference = points - centre

        return np.einsum("...i,ij,...j->...",
                         difference,
                         self.precision,
                         difference,
                         optimize = True)


    def evaluate(self,
                 points : np.ndarray,
                 centre : np.ndarray) -> np.ndarray:

        return np.exp(-0.5 * self.distance(points = points, centre = centre))


    def gradient(self,
                 points : np.ndarray,
                 centre : np.ndarray) -> np.ndarray:

        points = np.asarray(points, dtype = float)
        centre = np.asarray(centre, dtype = float)

        require_all([(points.shape[-1] == self.dimension, ValueError, "Point dimension does not match covariance."),
                     (centre.shape[-1] == self.dimension, ValueError, "Centre dimension does not match covariance.")])

        difference = points - centre
        values     = self.evaluate(points = points,
                                centre = centre)

        return -values[..., None] * np.einsum("ij,...j->...i",
                                              self.precision,
                                              difference,
                                              optimize = True)


    def convolve_line(self,
                      points : np.ndarray,
                      start  : np.ndarray,
                      end    : np.ndarray) -> np.ndarray:
        
        points = np.asarray(points, dtype = float)
        start  = np.asarray(start,  dtype = float)
        end    = np.asarray(end,    dtype = float)

        require_all([(points.shape[-1] == self.dimension, ValueError, "Point dimension does not match covariance."),
                     (start.shape[-1]  == self.dimension, ValueError, "Line start dimension does not match covariance."),
                     (end.shape[-1]    == self.dimension, ValueError, "Line end dimension does not match covariance.")])

        segment = end - start
        length  = np.linalg.norm(segment, axis = -1)

        valid       = ~np.isclose(length, 0.0)
        safe_length = np.where(valid, length, 1.0)

        direction  = segment/safe_length[..., None]
        difference = points - start

        direction_precision = np.einsum("...i,ij,...j->...",
                                direction,
                                self.precision,
                                direction,
                                optimize = True)

        projection = np.einsum("...i,ij,...j->...",
                            direction,
                            self.precision,
                            difference,
                            optimize = True)

        distance = np.einsum("...i,ij,...j->...",
                            difference,
                            self.precision,
                            difference,
                            optimize = True)

        safe_direction_precision = np.where(valid,
                                            direction_precision,
                                            1.0)

        exponent = distance - projection**2/safe_direction_precision
        scale    = np.sqrt(2.0*safe_direction_precision)

        longitudinal = (erf((safe_direction_precision*safe_length - projection)/scale) + erf(projection/scale))
        values = (np.sqrt(np.pi/(2.0*safe_direction_precision)) * np.exp(-0.5*exponent) * longitudinal)

        return np.where(valid,
                        values,
                        0.0)


    def convolve_line_gradient(self,
                               points : np.ndarray,
                               start  : np.ndarray,
                               end    : np.ndarray) -> np.ndarray:

        points = np.asarray(points, dtype = float)
        start  = np.asarray(start,  dtype = float)
        end    = np.asarray(end,    dtype = float)

        require_all([(points.shape[-1] == self.dimension, ValueError, "Point dimension does not match covariance."),
                    (start.shape[-1]  == self.dimension, ValueError, "Line start dimension does not match covariance."),
                    (end.shape[-1]    == self.dimension, ValueError, "Line end dimension does not match covariance.")])

        segment = end - start
        length  = np.linalg.norm(segment, axis = -1)

        valid       = ~np.isclose(length, 0.0)
        safe_length = np.where(valid, length, 1.0)

        direction  = segment/safe_length[..., None]
        difference = points - start

        direction_precision = np.einsum("...i,ij,...j->...",
                                        direction,
                                        self.precision,
                                        direction,
                                        optimize = True)

        projection = np.einsum("...i,ij,...j->...",
                            direction,
                            self.precision,
                            difference,
                            optimize = True)

        distance = np.einsum("...i,ij,...j->...",
                            difference,
                            self.precision,
                            difference,
                            optimize = True)

        safe_direction_precision = np.where(valid,
                                            direction_precision,
                                            1.0)

        integral = self.convolve_line(points = points,
                                    start  = start,
                                    end    = end)

        start_value = np.exp(-0.5*distance)
        end_value   = np.exp(-0.5*(distance - 2.0*projection*safe_length + direction_precision*safe_length**2))

        first_moment = (projection*integral + start_value - end_value)/safe_direction_precision

        displacement = (difference*integral[..., None]
                        - direction*first_moment[..., None])

        gradient = -np.einsum("ij,...j->...i",
                            self.precision,
                            displacement,
                            optimize = True)

        return np.where(valid[..., None],
                        gradient,
                        0.0)