"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

def test_daily_max_integers():
    """Test that max function works for an array of positive integers."""
    from inflammation.models import daily_max

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([5, 6])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)

def test_daily_max_nan():
    """Test that max function works for an array of positive integers."""
    from inflammation.models import daily_max

    test_input = np.array([[np.nan, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([np.nan, 6])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)



@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [np.nan, 2], [3, 4], [5, 6] ], [np.nan, 6]),
        ([ [5, 2], [3, 4], [5, 6] ], [5, 6]),
        ([ [22, 2], [3, 4], [5, 6]], [22, 6]),
    ])
def test_daily_max(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])


@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [1, 2], [1, 2], [1, 2] ], [0., 0.]),
        ([ [0, 0], [0, 0], [0, 0] ], [0., 0.]),
        ([ [-1, -2], [np.nan, -2], [-1, -2]], [np.nan, 0.]),
    ])
def test_daily_std(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_std_dev
    npt.assert_array_equal(daily_std_dev(np.array(test)), np.array(expected))
