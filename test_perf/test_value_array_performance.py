import unittest

from .perf_testing_util import (
    a_random_compatible_unit_array,
    a_random_compatible_unit_val,
    a_random_unit_array,
    a_random_unit_val,
    perf_goal,
)


@perf_goal(avg_micros=15, args=[a_random_compatible_unit_array] * 2)
def test_perf_array_add(a, b):
    return a + b


@perf_goal(avg_micros=15, args=[a_random_compatible_unit_array, a_random_compatible_unit_val])
def test_perf_array_shift(a, b):
    return a + b


@perf_goal(avg_micros=15, args=[a_random_unit_array] * 2)
def test_perf_array_multiply(a, b):
    return a * b


@perf_goal(avg_micros=15, args=[a_random_unit_array, a_random_unit_val])
def test_perf_array_scale(a, b):
    return a * b


@perf_goal(avg_micros=15, args=[a_random_unit_array] * 2)
def test_perf_array_divide(a, b):
    return a / b


@perf_goal(avg_micros=750, args=[a_random_unit_array])
def test_perf_array_str(a):
    return str(a)


@perf_goal(avg_micros=900, args=[a_random_unit_array])
def test_perf_array_repr(a):
    return repr(a)


if __name__ == "__main__":
    unittest.main()
