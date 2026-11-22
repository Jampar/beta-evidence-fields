import numpy as np
import pytest

from beta_evidence_field.field import OccupancyField
from beta_evidence_field.observation import OccupancyObservation
from beta_evidence_field.prior import OccupancyPrior


def make_field():

    return OccupancyField(surface_covariance = np.eye(2),
                          free_covariance    = np.eye(2),
                          state_variance  = 1.0,
                          prior              = OccupancyPrior(alpha = 5.0, beta = 5.0))


def make_observation(time     : float = 0.0,
                     returned : bool  = True) -> OccupancyObservation:

    return OccupancyObservation(origin    = np.array([0.0, 0.0]),
                                direction = np.array([1.0, 0.0]),
                                range     = 2.0,
                                returned  = returned,
                                time      = time)


def test_empty_field_returns_prior():

    field = make_field()

    query = field.query(np.array([1.0, 0.0]))

    assert query.alpha   == pytest.approx(5.0)
    assert query.beta    == pytest.approx(5.0)
    assert query.support is False


def test_add():

    field       = make_field()
    observation = make_observation()

    added = field.add([observation])

    assert added == [observation]
    assert len(field) == 1


def test_clear():

    field = make_field()

    field.add([make_observation()])
    field.clear()

    assert len(field) == 0


def test_return_increases_occupancy_at_endpoint():

    field       = make_field()
    observation = make_observation(returned = True)

    field.add([observation])

    query = field.query(observation.endpoint)

    assert query.alpha > field.prior.alpha
    assert query.support is True


def test_ray_increases_free_space_evidence():

    field       = make_field()
    observation = make_observation(returned = False)

    field.add([observation])

    query = field.query(np.array([1.0, 0.0]))

    assert query.beta > field.prior.beta
    assert query.alpha == pytest.approx(field.prior.alpha)


def test_no_return_does_not_add_surface_evidence():

    field       = make_field()
    observation = make_observation(returned = False)

    field.add([observation])

    query = field.query(observation.endpoint)

    assert query.alpha == pytest.approx(field.prior.alpha)
    assert query.support is False


def test_vectorised_query_matches_scalar_queries():

    field = make_field()

    field.add([make_observation()])

    points = np.array([[0.0, 0.0],
                       [1.0, 0.0],
                       [2.0, 0.0],
                       [3.0, 0.0]])

    vector = field.query(points)
    scalar = [field.query(point) for point in points]

    np.testing.assert_allclose(vector.alpha,        np.asarray([query.alpha     for query in scalar], dtype = float))
    np.testing.assert_allclose(vector.beta,         np.asarray([query.beta      for query in scalar], dtype = float))
    np.testing.assert_array_equal(vector.support,   np.asarray([query.support   for query in scalar], dtype = bool))


def test_observations_must_be_chronological():

    field = make_field()

    field.add([make_observation(time = 2.0)])

    with pytest.raises(ValueError):
        field.add([make_observation(time = 1.0)])


def test_observations_added_together_must_have_same_time():

    field = make_field()

    observations = [make_observation(time = 0.0),
                    make_observation(time = 1.0)]

    with pytest.raises(ValueError):
        field.add(observations)


def test_observation_dimension_must_match_field():

    field = make_field()

    observation = OccupancyObservation(origin    = np.zeros(3),
                                       direction = np.array([1.0, 0.0, 0.0]),
                                       range     = 1.0,
                                       returned  = True,
                                       time      = 0.0)

    with pytest.raises(ValueError):
        field.add([observation])


def test_future_observations_do_not_affect_past_queries():

    field = make_field()

    field.add([make_observation(time = 1.0)])

    query = field.query(np.array([2.0, 0.0]),
                        time = 0.0)

    assert query.alpha   == pytest.approx(field.prior.alpha)
    assert query.beta    == pytest.approx(field.prior.beta)
    assert query.support is False