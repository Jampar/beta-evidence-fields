from pathlib import Path

import numpy as np
import pytest

from beta_evidence_field.field import OccupancyField
from beta_evidence_field.observation import OccupancyObservation


CONFIG = Path(__file__).parents[1] / "config" / "example_2d_config.yaml"


def make_field():

    return OccupancyField.from_yaml(CONFIG)


def make_observation(time     : float = 0.0,
                     range    : float = 1.0,
                     returned : bool  = True) -> OccupancyObservation:

    return OccupancyObservation(origin    = np.array([0.0, 0.0]),
                                direction = np.array([1.0, 0.0]),
                                range     = range,
                                returned  = returned,
                                time      = time)


def dense_redundancy_weights(field        : OccupancyField,
                             observations : list[OccupancyObservation]) -> np.ndarray:

    states = np.asarray([observation.state
                         for observation in observations],
                        dtype = float)

    distance = field.state_kernel.distance(points = states[:, None, :],
                                           centre = states[None, :, :])

    similarity = np.exp(-0.5 * distance)
    similarity[distance > field.state_support_sigma**2] = 0.0

    return 1.0 / np.sum(similarity, axis = 1)


def test_empty_field_returns_prior():

    field = make_field()

    query = field.query(np.array([1.0, 0.0]))

    assert query.alpha   == pytest.approx(1.0)
    assert query.beta    == pytest.approx(1.0)
    assert query.support is False


def test_add():

    field       = make_field()
    observation = make_observation()

    added = field.add([observation])

    assert added == [observation]
    assert len(field) == 1


def test_add_empty():

    field = make_field()

    added = field.add([])

    assert added == []
    assert len(field) == 0


def test_clear():

    field = make_field()

    field.add([make_observation()])
    field.clear()

    assert len(field) == 0
    assert len(field._redundancy) == 0


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

    np.testing.assert_allclose(vector.alpha,      [query.alpha   for query in scalar])
    np.testing.assert_allclose(vector.beta,       [query.beta    for query in scalar])
    np.testing.assert_array_equal(vector.support, [query.support for query in scalar])


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


def test_historical_query_uses_subset_redundancy():

    field = make_field()

    first  = make_observation(time = 0.0)
    second = make_observation(time = 1.0)

    field.add([first])
    field.add([second])

    query = field.query(first.endpoint,
                        time = 0.0)

    reference = make_field()
    reference.add([first])

    expected = reference.query(first.endpoint)

    assert query.alpha == pytest.approx(expected.alpha)
    assert query.beta  == pytest.approx(expected.beta)


def test_empty_redundancy_weights():

    field = make_field()

    weights = field._redundancy_weights([])

    assert weights.shape == (0,)


def test_identical_observations_share_effective_weight():

    field = make_field()

    observations = [make_observation(),
                    make_observation()]

    weights = field._redundancy_weights(observations)

    np.testing.assert_allclose(weights,
                               np.array([0.5, 0.5]))


def test_identical_observations_have_unit_total_effective_weight():

    field = make_field()

    observations = [make_observation()
                    for _ in range(10)]

    weights = field._redundancy_weights(observations)

    assert np.sum(weights) == pytest.approx(1.0)


def test_different_ranges_reduce_redundancy():

    field = make_field()

    same = [make_observation(range = 2.0),
            make_observation(range = 2.0)]

    different = [make_observation(range = 2.0),
                 make_observation(range = 4.0)]

    same_weights      = field._redundancy_weights(same)
    different_weights = field._redundancy_weights(different)

    assert np.all(different_weights > same_weights)


def test_observations_inside_state_support_are_redundant():

    field = make_field()

    sigma_t = np.sqrt(field.temporal_variance)

    observations = [make_observation(time = 0.0),
                    make_observation(time = sigma_t)]

    weights  = field._redundancy_weights(observations)
    expected = 1.0/(1.0 + np.exp(-0.5))

    np.testing.assert_allclose(weights,
                               np.full(2, expected))


def test_observations_outside_state_support_are_not_redundant():

    field = make_field()

    sigma_t = np.sqrt(field.temporal_variance)

    observations = [make_observation(time = 0.0),
                    make_observation(time = 4.0*sigma_t)]

    weights = field._redundancy_weights(observations)

    np.testing.assert_allclose(weights,
                               np.ones(2))


def test_sparse_redundancy_matches_dense_reference():

    field = make_field()

    observations = [make_observation(time = time,
                                     range = range)
                    for time, range in [(0.0, 1.0),
                                        (0.2, 1.1),
                                        (0.5, 1.5),
                                        (1.0, 2.0),
                                        (2.0, 3.0)]]

    expected = dense_redundancy_weights(field,
                                        observations)

    actual = field._redundancy_weights(observations)

    np.testing.assert_allclose(actual,
                               expected)


def test_redundancy_cache_updates_on_add():

    field = make_field()

    field.add([make_observation()])

    np.testing.assert_allclose(field._redundancy,
                               np.array([1.0]))

    field.add([make_observation()])

    np.testing.assert_allclose(field._redundancy,
                               np.array([0.5, 0.5]))


def test_redundancy_cache_matches_retained_observations():

    field = make_field()

    field.add([make_observation(),
               make_observation()])

    expected = field._redundancy_weights(field.observations)

    np.testing.assert_allclose(field._redundancy,
                               expected)


def test_redundancy_cache_matches_observations_after_pruning():

    field = make_field()

    sigma_t = np.sqrt(field.temporal_variance)

    field.add([make_observation(time = 0.0)])
    field.add([make_observation(time = 4.0*sigma_t)])

    assert len(field.observations) == 1

    expected = field._redundancy_weights(field.observations)

    np.testing.assert_allclose(field._redundancy,
                               expected)
