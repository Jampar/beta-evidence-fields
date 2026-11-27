"""Gaussian kernel operations for continuous evidence fields."""

from scipy.linalg  import solve_triangular
from scipy.spatial import cKDTree
from scipy.special import erf

import numpy as np


class GaussianKernel:
    """Represents an unnormalised multivariate Gaussian kernel.

    The kernel is parameterised by a symmetric positive-definite covariance
    matrix and supports evaluation, squared Mahalanobis distance, and analytic
    convolution along finite line segments.

    Attributes:
        covariance: Covariance matrix defining the spatial shape of the kernel.
        precision: Inverse covariance matrix of the kernel.
    """


    def __init__(self,
                 covariance : np.ndarray):
        """Initialises a Gaussian kernel from a covariance matrix.

        Args:
            covariance: Symmetric positive-definite covariance matrix.

        Raises:
            ValueError: If the covariance is not a square, symmetric,
                positive-definite matrix.
        """

        covariance = np.asarray(covariance, dtype = float)

        if covariance.ndim != 2:
            raise ValueError("Covariance must be a matrix.")

        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("Covariance must be square.")

        if not np.allclose(covariance, covariance.T):
            raise ValueError("Covariance must be symmetric.")

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
        """Returns the spatial dimension of the kernel."""

        return self.covariance.shape[0]


    def sparse_distance(self,
                        points        : np.ndarray,
                        support_sigma : float):
        """Calculates sparse squared Mahalanobis distances within support."""

        points = np.asarray(points, dtype = float)

        if points.ndim != 2:
            raise ValueError("Points must be a matrix.")

        if points.shape[1] != self.dimension:
            raise ValueError("Point dimension does not match covariance.")

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
        """Calculates squared Mahalanobis distance from a kernel centre.

        Leading dimensions are broadcast according to NumPy broadcasting
        rules. The final axis represents spatial dimension.

        Args:
            points: Points at which to evaluate the distance.
            centre: Centre of the Gaussian kernel.

        Returns:
            Squared Mahalanobis distance over the broadcast dimensions.

        Raises:
            ValueError: If points or centre do not match the spatial dimension
                of the kernel.
        """

        points = np.asarray(points, dtype = float)
        centre = np.asarray(centre, dtype = float)

        if points.shape[-1] != self.dimension:
            raise ValueError("Point dimension does not match covariance.")

        if centre.shape[-1] != self.dimension:
            raise ValueError("Centre dimension does not match covariance.")

        difference = points - centre

        return np.einsum("...i,ij,...j->...",
                         difference,
                         self.precision,
                         difference,
                         optimize = True)


    def evaluate(self,
                 points : np.ndarray,
                 centre : np.ndarray) -> np.ndarray:
        """Evaluates the unnormalised Gaussian kernel.

        Leading dimensions are broadcast according to NumPy broadcasting
        rules. The final axis represents spatial dimension.

        Args:
            points: Points at which to evaluate the kernel.
            centre: Centre of the Gaussian kernel.

        Returns:
            Kernel values over the broadcast dimensions.
        """

        return np.exp(-0.5*self.distance(points = points,
                                         centre = centre))


    def gradient(self,
                 points : np.ndarray,
                 centre : np.ndarray) -> np.ndarray:
        """Evaluates the spatial gradient of the Gaussian kernel."""

        points = np.asarray(points, dtype = float)
        centre = np.asarray(centre, dtype = float)

        if points.shape[-1] != self.dimension:
            raise ValueError("Point dimension does not match covariance.")

        if centre.shape[-1] != self.dimension:
            raise ValueError("Centre dimension does not match covariance.")

        difference = points - centre
        values     = self.evaluate(points = points,
                                centre = centre)

        return -values[..., None]*np.einsum("ij,...j->...i",
                                            self.precision,
                                            difference,
                                            optimize = True)


    def convolve_line(self,
                      points : np.ndarray,
                      start  : np.ndarray,
                      end    : np.ndarray) -> np.ndarray:
        """Evaluates the Gaussian kernel integrated along finite line segments.

        Leading dimensions are broadcast according to NumPy broadcasting
        rules. The final axis represents spatial dimension.

        Args:
            points: Points at which to evaluate the line convolution.
            start: Start of each finite line segment.
            end: End of each finite line segment.

        Returns:
            Integrated Gaussian kernel values over the broadcast dimensions.

        Raises:
            ValueError: If points or line endpoints do not match the spatial
                dimension of the kernel.
        """

        points = np.asarray(points, dtype = float)
        start  = np.asarray(start,  dtype = float)
        end    = np.asarray(end,    dtype = float)

        if points.shape[-1] != self.dimension:
            raise ValueError("Point dimension does not match covariance.")

        if start.shape[-1] != self.dimension:
            raise ValueError("Line start dimension does not match covariance.")

        if end.shape[-1] != self.dimension:
            raise ValueError("Line end dimension does not match covariance.")

        segment = end - start
        length  = np.linalg.norm(segment, axis = -1)

        valid       = ~np.isclose(length, 0.0)
        safe_length = np.where(valid, length, 1.0)

        direction  = segment/safe_length[..., None]
        difference = points - start

        alpha = np.einsum("...i,ij,...j->...",
                          direction,
                          self.precision,
                          direction,
                          optimize = True)

        beta = np.einsum("...i,ij,...j->...",
                         direction,
                         self.precision,
                         difference,
                         optimize = True)

        gamma = np.einsum("...i,ij,...j->...",
                          difference,
                          self.precision,
                          difference,
                          optimize = True)

        safe_alpha = np.where(valid, alpha, 1.0)

        exponent = gamma - beta**2/safe_alpha
        scale    = np.sqrt(2.0*safe_alpha)

        longitudinal = (erf((safe_alpha*safe_length - beta)/scale)
                        + erf(beta/scale))

        values = (np.sqrt(np.pi/(2.0*safe_alpha))
                  * np.exp(-0.5*exponent)
                  * longitudinal)

        return np.where(valid,
                        values,
                        0.0)