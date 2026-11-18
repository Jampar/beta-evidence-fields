"""Gaussian kernel operations for continuous evidence fields."""

import numpy as np

from scipy.special import erf


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
            np.linalg.cholesky(covariance)

        except np.linalg.LinAlgError as error:
            raise ValueError("Covariance must be positive definite.") from error

        self.covariance = covariance
        self.precision  = np.linalg.inv(covariance)


    @property
    def dimension(self) -> int:
        """Returns the spatial dimension of the kernel."""

        return self.covariance.shape[0]


    def distance(self,
                 points : np.ndarray,
                 centre : np.ndarray) -> np.ndarray:
        """Calculates squared Mahalanobis distance from a kernel centre.

        Args:
            points: Points at which to evaluate the distance.
            centre: Centre of the Gaussian kernel.

        Returns:
            Squared Mahalanobis distance of each point from the centre.
        """

        difference = np.asarray(points, dtype = float) - centre

        return np.einsum("...i,ij,...j->...",
                         difference,
                         self.precision,
                         difference)


    def evaluate(self,
                 points : np.ndarray,
                 centre : np.ndarray) -> np.ndarray:
        """Evaluates the unnormalised Gaussian kernel.

        Args:
            points: Points at which to evaluate the kernel.
            centre: Centre of the Gaussian kernel.

        Returns:
            Kernel value at each point.
        """

        return np.exp(-0.5 * self.distance(points = points,
                                           centre = centre))


    def convolve_line(self,
                      points : np.ndarray,
                      start  : np.ndarray,
                      end    : np.ndarray) -> np.ndarray:
        """Evaluates the Gaussian kernel integrated along a finite line segment.

        Computes the analytic line integral of the kernel between `start` and
        `end` for each query point. The formulation is dimension agnostic and
        uses the covariance-induced metric of the kernel.

        Args:
            points: Points at which to evaluate the line convolution.
            start: Start of the finite line segment.
            end: End of the finite line segment.

        Returns:
            Integrated Gaussian kernel value at each query point.

        Raises:
            ValueError: If the query points or line endpoints do not match the
                spatial dimension of the kernel.
        """

        points = np.asarray(points, dtype = float)
        start  = np.asarray(start,  dtype = float)
        end    = np.asarray(end,    dtype = float)

        scalar = points.ndim == 1

        if scalar:
            points = points[None, :]

        if points.shape[-1] != self.dimension:
            raise ValueError("Point dimension does not match covariance.")

        if start.shape != (self.dimension,):
            raise ValueError("Line start has incorrect dimension.")

        if end.shape != (self.dimension,):
            raise ValueError("Line end has incorrect dimension.")

        segment = end - start
        length  = np.linalg.norm(segment)

        if np.isclose(length, 0.0):
            values = np.zeros(len(points))

            return values[0] if scalar else values

        direction       = segment/length
        difference      = points - start
        alpha           = direction @ self.precision @ direction

        beta            = np.einsum("i,ij,nj->n",
                                    direction,
                                    self.precision,
                                    difference)
       
        gamma           = np.einsum("ni,ij,nj->n",
                                    difference,
                                    self.precision,
                                    difference)

        exponent        = gamma - beta**2/alpha
        scale           = np.sqrt(2.0*alpha)

        longitudinal    = (erf((alpha*length - beta)/scale) + erf(beta/scale))
        values          = (np.sqrt(np.pi/(2.0*alpha)) * np.exp(-0.5*exponent) * longitudinal)

        return values[0] if scalar else values