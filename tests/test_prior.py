import numpy as np
import pytest

from beta_evidence_field.prior import OccupancyPrior, DEFAULT_PRIOR_VALUE


def test_default_prior():

    prior = OccupancyPrior()

    assert prior.alpha == pytest.approx(DEFAULT_PRIOR_VALUE)
    assert prior.beta  == pytest.approx(DEFAULT_PRIOR_VALUE)


def test_custom_prior():

    prior = OccupancyPrior(alpha = 2.0,
                           beta  = 3.0)

    assert prior.alpha == pytest.approx(2.0)
    assert prior.beta  == pytest.approx(3.0)


@pytest.mark.parametrize("alpha", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_alpha_must_be_finite_and_positive(alpha):

    with pytest.raises(ValueError):
        OccupancyPrior(alpha = alpha)


@pytest.mark.parametrize("beta", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_beta_must_be_finite_and_positive(beta):

    with pytest.raises(ValueError):
        OccupancyPrior(beta = beta)


def test_evaluate():

    prior = OccupancyPrior(alpha = 2.0,
                           beta  = 3.0)

    points = np.array([[0.0, 0.0],
                       [1.0, 2.0],
                       [3.0, 4.0]])

    alpha, beta = prior.evaluate(points)

    np.testing.assert_array_equal(alpha,    np.array([2.0, 2.0, 2.0]))
    np.testing.assert_array_equal(beta,     np.array([3.0, 3.0, 3.0]))


def test_evaluate_dimension_independence():

    prior = OccupancyPrior(alpha = 2.0,
                           beta  = 3.0)

    points = np.zeros((4, 3))

    alpha, beta = prior.evaluate(points)

    assert alpha.shape == (4,)
    assert beta.shape  == (4,)

    np.testing.assert_array_equal(alpha, np.full(4, 2.0))
    np.testing.assert_array_equal(beta,  np.full(4, 3.0))