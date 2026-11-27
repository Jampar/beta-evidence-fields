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

    kernel   = GaussianKernel(np.eye(2))
    points   = np.array([[1.0, 0.0],
                         [0.0, 2.0],
                         [1.0, 2.0]])

    centre   = np.zeros(2)
    expected = np.array([1.0, 4.0, 5.0])

    np.testing.assert_allclose(kernel.distance(points, centre),
                               expected)


def test_distance_dimension_validation():

    kernel = GaussianKernel(np.eye(2))

    with pytest.raises(ValueError):
        kernel.distance(points = np.zeros(3),
                        centre = np.zeros(2))

    with pytest.raises(ValueError):
        kernel.distance(points = np.zeros(2),
                        centre = np.zeros(3))


def test_sparse_distance_matches_dense_distance():

    kernel = GaussianKernel(np.array([[2.0, 0.3],
                                      [0.3, 1.0]]))

    points = np.array([[0.0, 0.0],
                       [0.5, 0.2],
                       [1.0, 1.0],
                       [3.0, 2.0]])

    sparse = kernel.sparse_distance(points,
                                    support_sigma = 2.0)

    dense = kernel.distance(points = points[:, None, :],
                            centre = points[None, :, :])

    np.testing.assert_allclose(sparse.data,
                               dense[sparse.row, sparse.col])


def test_sparse_distance_respects_support():

    kernel = GaussianKernel(np.eye(2))

    points = np.array([[0.0, 0.0],
                       [1.0, 0.0],
                       [4.0, 0.0]])

    distance = kernel.sparse_distance(points,
                                      support_sigma = 1.5)

    pairs = set(zip(distance.row,
                    distance.col))

    assert (0, 1) in pairs
    assert (1, 0) in pairs
    assert (0, 2) not in pairs
    assert (2, 0) not in pairs


def test_sparse_distance_validation():

    kernel = GaussianKernel(np.eye(2))

    with pytest.raises(ValueError):
        kernel.sparse_distance(np.array([0.0, 0.0]),
                               support_sigma = 3.0)

    with pytest.raises(ValueError):
        kernel.sparse_distance(np.zeros((3, 3)),
                               support_sigma = 3.0)


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

    kernel = GaussianKernel(np.eye(2))

    points   = np.array([[0.0, 0.0],
                         [1.0, 0.0],
                         [0.0, 2.0]])

    values   = kernel.evaluate(points, np.zeros(2))
    expected = np.exp(-0.5 * np.array([0.0, 1.0, 4.0]))

    np.testing.assert_allclose(values,
                               expected)
    

def test_gradient_matches_finite_difference():

    kernel = GaussianKernel(np.array([
        [2.0, 0.3],
        [0.3, 1.0],
    ]))

    point  = np.array([1.2, -0.7])
    centre = np.array([0.3,  0.4])

    gradient = kernel.gradient(point, centre)

    epsilon = 1e-6
    finite  = np.empty(2)

    for i in range(2):
        offset    = np.zeros(2)
        offset[i] = epsilon
        finite[i] = (kernel.evaluate(point + offset, centre) - kernel.evaluate(point - offset, centre))/(2*epsilon)

    np.testing.assert_allclose(gradient, 
                               finite, 
                               rtol = 1e-5, 
                               atol = 1e-7)


def test_convolve_line_against_numerical_integration():

    kernel = GaussianKernel(np.array([[2.0, 0.3],
                                      [0.3, 1.0]]))

    point     = np.array([0.4, 0.7])
    start     = np.array([-1.0, 0.2])
    end       = np.array([2.0, 1.4])

    segment   = end - start
    length    = np.linalg.norm(segment)
    direction = segment/length

    numerical, _ = quad(lambda s: kernel.evaluate(point,
                                                   start + s*direction),
                        0.0,
                        length)

    analytic = kernel.convolve_line(points = point,
                                    start  = start,
                                    end    = end)

    assert analytic == pytest.approx(numerical,
                                     rel = 1e-10,
                                     abs = 1e-12)


def test_convolve_line_zero_length():

    kernel = GaussianKernel(np.eye(2))

    point = np.array([1.0, 1.0])
    start = np.array([0.0, 0.0])

    value = kernel.convolve_line(points = point,
                                 start  = start,
                                 end    = start)

    assert value == pytest.approx(0.0)


def test_convolve_line_vectorised():

    kernel = GaussianKernel(np.eye(2))

    points = np.array([[0.0, 0.0],
                       [1.0, 1.0]])

    starts = np.array([[-1.0, 0.0],
                       [0.0, 1.0]])

    ends   = np.array([[1.0, 0.0],
                       [2.0, 1.0]])

    values = kernel.convolve_line(points = points,
                                  start  = starts,
                                  end    = ends)

    assert values.shape == (2,)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)


def test_convolve_line_dimension_validation():

    kernel = GaussianKernel(np.eye(2))

    with pytest.raises(ValueError):
        kernel.convolve_line(points = np.zeros(3),
                             start  = np.zeros(2),
                             end    = np.zeros(2))

    with pytest.raises(ValueError):
        kernel.convolve_line(points = np.zeros(2),
                             start  = np.zeros(3),
                             end    = np.zeros(2))

    with pytest.raises(ValueError):
        kernel.convolve_line(points = np.zeros(2),
                             start  = np.zeros(2),
                             end    = np.zeros(3))


def test_three_dimensions():

    kernel = GaussianKernel(np.eye(3))

    point  = np.array([1.0, 2.0, 2.0])
    centre = np.zeros(3)

    assert kernel.distance(point, centre) == pytest.approx(9.0)
    assert kernel.evaluate(point, centre) == pytest.approx(np.exp(-4.5))

