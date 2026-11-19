import numpy as np
import pytest

from scipy.integrate import quad

from beta_evidence_field.kernels.gaussian import GaussianKernel


def test_covariance_validation():

    with pytest.raises(ValueError):
        GaussianKernel(np.array([1.0, 1.0]))

    with pytest.raises(ValueError):
        GaussianKernel(np.ones((2, 3)))

    with pytest.raises(ValueError):
        GaussianKernel(np.array([[1.0, 1.0],
                                 [0.0, 1.0]]))

    with pytest.raises(ValueError):
        GaussianKernel(np.array([[1.0, 0.0],
                                 [0.0, 0.0]]))


def test_dimension():
    kernel = GaussianKernel(np.eye(3))
    
    assert kernel.dimension == 3


def test_distance_at_centre():
    kernel = GaussianKernel(np.eye(2))
    point  = np.array([1.0, 2.0])

    assert kernel.distance(point, point) == pytest.approx(0.0)


def test_distance_identity():

    kernel      = GaussianKernel(np.eye(2))
    points      = np.array([[1.0, 0.0],
                            [0.0, 2.0],
                            [1.0, 2.0]])

    centre      = np.zeros(2)
    expected    = np.array([1.0, 4.0, 5.0])

    np.testing.assert_allclose(kernel.distance(points, centre), expected)


def test_evaluate_at_centre():

    kernel = GaussianKernel(np.eye(2))
    centre = np.array([2.0, -1.0])

    assert kernel.evaluate(centre, centre) == pytest.approx(1.0)


def test_evaluate_known_distance():

    kernel = GaussianKernel(np.eye(2))

    point  = np.array([1.0, 0.0])
    centre = np.zeros(2)

    assert kernel.evaluate(point, centre) == pytest.approx(np.exp(-0.5))


def test_evaluate_vectorised():

    kernel      = GaussianKernel(np.eye(2))

    points      = np.array([[0.0, 0.0],
                            [1.0, 0.0],
                            [0.0, 2.0]])

    values      = kernel.evaluate(points, np.zeros(2))
    expected    = np.exp(-0.5 * np.array([0.0, 1.0, 4.0]))

    np.testing.assert_allclose(values, expected)


def test_convolve_line_against_numerical_integration():

    kernel          = GaussianKernel(np.array([[2.0, 0.3],
                                               [0.3, 1.0]]))

    point           = np.array([0.4, 0.7])
    start           = np.array([-1.0, 0.2])
    end             = np.array([2.0, 1.4])

    segment         = end - start
    length          = np.linalg.norm(segment)
    direction       = segment/length

    numerical, _    = quad(lambda s: kernel.evaluate(point, start + s * direction), 0.0, length)

    analytic = kernel.convolve_line(points = point,
                                    start  = start,
                                    end    = end)

    assert analytic == pytest.approx(numerical,
                                     rel = 1e-10,
                                     abs = 1e-12)


def test_convolve_line_zero_length():

    kernel  = GaussianKernel(np.eye(2))

    point   = np.array([1.0, 1.0])
    start   = np.array([0.0, 0.0])

    value   = kernel.convolve_line(points = point,
                                   start  = start,
                                   end    = start)

    assert value == pytest.approx(0.0)


def test_convolve_line_vectorised():

    kernel  = GaussianKernel(np.eye(2))

    points  = np.array([[0.0, 0.0],
                        [1.0, 1.0]])

    starts  = np.array([[-1.0, 0.0],
                        [0.0, 1.0]])

    ends    = np.array([[1.0, 0.0],
                        [2.0, 1.0]])

    values  = kernel.convolve_line(points = points,
                                   start  = starts,
                                   end    = ends)

    assert values.shape == (2,)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)


def test_three_dimensions():

    kernel  = GaussianKernel(np.eye(3))

    point   = np.array([1.0, 2.0, 2.0])
    centre  = np.zeros(3)

    assert kernel.distance(point, centre) == pytest.approx(9.0)
    assert kernel.evaluate(point, centre) == pytest.approx(np.exp(-4.5))