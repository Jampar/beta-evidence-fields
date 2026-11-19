import numpy as np
import pytest

from beta_evidence_field.observation import OccupancyObservation


def test_initialisation():

    observation = OccupancyObservation(origin          = [1.0, 2.0],
                                       direction       = [1.0, 0.0],
                                       range           = 3.0,
                                       returned        = True,
                                       time            = 4.0,
                                       evidence_weight = 0.5)

    np.testing.assert_array_equal(observation.origin, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(observation.direction,np.array([1.0, 0.0]))

    assert observation.range           == pytest.approx(3.0)
    assert observation.returned        is True
    assert observation.time            == pytest.approx(4.0)
    assert observation.evidence_weight == pytest.approx(0.5)


def test_default_evidence_weight():

    observation = OccupancyObservation(origin    = [0.0, 0.0],
                                       direction = [1.0, 0.0],
                                       range     = 1.0,
                                       returned  = True,
                                       time      = 0.0)

    assert observation.evidence_weight == pytest.approx(1.0)


def test_direction_is_normalised():

    observation = OccupancyObservation(origin    = [0.0, 0.0],
                                       direction = [3.0, 4.0],
                                       range     = 1.0,
                                       returned  = True,
                                       time      = 0.0)

    np.testing.assert_allclose(observation.direction, np.array([0.6, 0.8]))

    assert np.linalg.norm(observation.direction) == pytest.approx(1.0)


def test_endpoint():

    observation = OccupancyObservation(origin    = [1.0, 2.0],
                                       direction = [3.0, 4.0],
                                       range     = 5.0,
                                       returned  = True,
                                       time      = 0.0)

    np.testing.assert_allclose(observation.endpoint, np.array([4.0, 6.0]))


def test_dimension():

    observation = OccupancyObservation(origin    = [0.0, 0.0, 0.0],
                                       direction = [1.0, 0.0, 0.0],
                                       range     = 1.0,
                                       returned  = True,
                                       time      = 0.0)

    assert observation.dimension == 3


def test_origin_must_be_one_dimensional():

    with pytest.raises(ValueError):
        OccupancyObservation(origin    = [[0.0, 0.0]],
                             direction = [1.0, 0.0],
                             range     = 1.0,
                             returned  = True,
                             time      = 0.0)


def test_direction_must_match_origin():

    with pytest.raises(ValueError):
        OccupancyObservation(origin    = [0.0, 0.0],
                             direction = [1.0, 0.0, 0.0],
                             range     = 1.0,
                             returned  = True,
                             time      = 0.0)


def test_direction_must_be_non_zero():

    with pytest.raises(ValueError):
        OccupancyObservation(
            origin    = [0.0, 0.0],
            direction = [0.0, 0.0],
            range     = 1.0,
            returned  = True,
            time      = 0.0
        )


def test_range_must_be_non_negative():

    with pytest.raises(ValueError):
        OccupancyObservation(origin    = [0.0, 0.0],
                             direction = [1.0, 0.0],
                             range     = -1.0,
                             returned  = True,
                             time      = 0.0)


def test_zero_range():

    observation = OccupancyObservation(origin    = [1.0, 2.0],
                                       direction = [1.0, 0.0],
                                       range     = 0.0,
                                       returned  = True,
                                       time      = 0.0)

    np.testing.assert_array_equal(observation.endpoint, observation.origin)