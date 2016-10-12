import unittest
from .perf_testing_util import *
from pyfu import Value


@perf_goal(avg_micros=5, args=[a_random_compatible_unit_val] * 2)
def test_perf_add(a, b):
    return a + b


@perf_goal(avg_micros=5, args=[a_random_unit_val])
def test_perf_scale(a):
    return a * 3.14


@perf_goal(avg_micros=5, args=[a_random_unit_val] * 2)
def test_perf_multiply(a, b):
    return a * b


@perf_goal(avg_micros=5, args=[a_random_compatible_unit_val] * 2)
def test_perf_get_item(a, b):
    return a[b]


@perf_goal(avg_micros=5, args=[a_random_compatible_unit_val] * 2)
def test_perf_divmod(a, b):
    return divmod(a, b)


@perf_goal(avg_micros=20, args=[a_random_compatible_unit_val] * 2)
def test_perf_import_multiply_add_heterogeneous(a, b):
    from pyfu.units import kilometer, inch
    return a * kilometer + b * inch


@perf_goal(avg_micros=3, args=[a_random_unit_val])
def test_perf_abs(a):
    return abs(a)


@perf_goal(avg_micros=10, args=[a_random_unit_val])
def test_perf_pow(a):
    return abs(a)**(2 / 3.0)


@perf_goal(avg_micros=20, args=[a_random_unit_val])
def test_perf_str(a):
    return str(a)


@perf_goal(avg_micros=80, args=[a_random_unit_val])
def test_perf_repr(a):
    return repr(a)


@perf_goal(avg_micros=750)
def test_perf_parse_unit():
    return Value(1, 'm*s/kg^4')


@perf_goal(avg_micros=3, args=[a_random_unit_val] * 2)
def test_perf_is_compatible(a, b):
    return a.isCompatible(b)


@perf_goal(avg_micros=3, args=[a_random_unit_val] * 2)
def test_perf_equate(a, b):
    return a == b


@perf_goal(avg_micros=5, args=[a_random_compatible_unit_val] * 2)
def test_perf_compare(a, b):
    return a < b


if __name__ == "__main__":
    unittest.main()
