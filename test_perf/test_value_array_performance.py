import unittest
from .perf_testing_util import *


@perf_goal(avg_micros=15, args=[a_random_compatible_unit_array] * 2)
def test_perf_add_arrays(a, b):
    return a + b


@perf_goal(avg_micros=15, args=[a_random_compatible_unit_array,
                                a_random_compatible_unit_val])
def test_perf_add_value_array(a, b):
    return a + b


@perf_goal(avg_micros=15, args=[a_random_unit_array] * 2)
def test_perf_multiply_arrays(a, b):
    return a * b


@perf_goal(avg_micros=15, args=[a_random_compatible_unit_array,
                                a_random_compatible_unit_val])
def test_perf_multiply_value_array(a, b):
    return a * b


if __name__ == "__main__":
    unittest.main()
