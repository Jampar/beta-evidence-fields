import numpy as np
import pytest

from beta_evidence_field.query import OccupancyQuery


def test_concentration():

    query = OccupancyQuery(alpha   = 3.0,
                           beta    = 2.0,
                           support = True)

    assert query.concentration == pytest.approx(5.0)


def test_mean():

    query = OccupancyQuery(alpha   = 3.0,
                           beta    = 2.0,
                           support = True)

    assert query.mean == pytest.approx(0.6)


def test_variance():

    query = OccupancyQuery(alpha   = 3.0,
                           beta    = 2.0,
                           support = True)

    expected = (3.0*2.0) / (5.0**2*6.0)

    assert query.variance == pytest.approx(expected)


def test_vectorisation():

    query = OccupancyQuery(alpha   = np.array([1.0, 2.0, 3.0]),
                           beta    = np.array([1.0, 2.0, 1.0]),
                           support = np.array([True, True, False]))

    np.testing.assert_allclose(query.concentration, np.array([2.0, 4.0, 4.0]))
    np.testing.assert_allclose(query.mean,          np.array([0.5, 0.5, 0.75]))
    np.testing.assert_allclose(query.variance,      np.array([1.0/12.0, 
                                                              1.0/20.0, 
                                                              3.0/80.0]))
    np.testing.assert_array_equal(query.support,    np.array([True, True, False]))