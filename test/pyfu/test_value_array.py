import numpy as np
import pytest
from pyfu._all_cythonized import raw_WithUnit, raw_UnitArray

from pyfu import ValueArray, UnitMismatchError


def array_equals(a, b):
    return np.asarray(a).shape == np.asarray(b).shape and np.all(a == b)


def test_construction():
    from pyfu.units import ns, ps
    assert isinstance([1, 2, 3] * ns, ValueArray)
    assert array_equals([1, 2, 3] * ns, [1000, 2000, 3000] * ps)


def test_slicing():
    from pyfu.units import ms, ns
    assert array_equals(([0, 1, 2, 3, 4] * ms)[3:], [3, 4] * ms)
    assert array_equals(([0, 1, 2, 3, 4] * ns)[::2], [0, 2, 4] * ns)


def test_set_item():
    from pyfu.units import km, m, s
    v = [1, 2, 3] * km

    with pytest.raises(UnitMismatchError):
        v[0] = 2 * s

    v[0] = 2 * km
    v[1] = 16 * m

    assert array_equals(v, [2000, 16, 3000] * m)


def test_addition():
    from pyfu.units import km, m
    assert array_equals([1, 2, 3] * km + [2, 3, 5] * m,
                        [1002, 2003, 3005] * m)

    assert array_equals([1, 2, 3] * km + 5 * m,
                        [1005, 2005, 3005] * m)

    with pytest.raises(UnitMismatchError):
        _ = 1.0 + [1, 2, 3] * km


def test_multiplication():
    from pyfu.units import km, m
    assert array_equals(([1, 2, 3] * km) * ([2, 3, 5] * m),
                        [2, 6, 15] * (km * m))

    assert array_equals(([1, 2, 3] * km) * 5j,
                        [5j, 10j, 15j] * km)

    u = [1, 2, 3] * km
    assert isinstance(u, ValueArray)
    assert u.display_units == km.display_units
    assert u[1] == 2 * km

    u = np.array([1., 2, 3]) * km
    assert isinstance(u, ValueArray)
    assert u.display_units == km.display_units
    assert u[1] == 2 * km


def test_power():
    from pyfu.units import s
    assert array_equals(([1, 2, 3] * s) ** 2,
                        [1, 4, 9] * s * s)


def test_repr():
    from pyfu.units import km, kg, s
    assert repr([] * s) == "ValueArray(array([], dtype=float64), 's')"
    assert repr([2, 3] * km) == "ValueArray(array([ 2.,  3.]), 'km')"
    assert repr([3j] * km * kg) == "ValueArray(array([ 0.+3.j]), 'kg*km')"
    assert (repr([-1] * km ** 2 / kg ** 3 * s) ==
            "ValueArray(array([-1.]), 'km^2*s/kg^3')")
    assert (repr([-1] * km ** (2 / 3.0) / kg ** 3 * s) ==
            "ValueArray(array([-1.]), 'km^(2/3)*s/kg^3')")

    # Numpy abbreviation is allowed.
    assert (repr(list(range(50000)) * km) ==
            "ValueArray(array([  0.00000000e+00,   1.00000000e+00,   "
            "2.00000000e+00, ...,\n         4.99970000e+04,   "
            "4.99980000e+04,   4.99990000e+04]), 'km')")

    # Fallback case.
    v = raw_WithUnit([1, 2, 3],
                     {
                         'factor': 3.0,
                         'ratio': {
                             'numer': 2,
                             'denom': 5,
                         },
                         'exp10': 10
                     },
                     raw_UnitArray([('muffin', 1, 1)]),
                     raw_UnitArray([('cookie', 1, 1)]))
    assert (repr(v) ==
            "raw_WithUnit(array([1, 2, 3]), "
            "{'exp10': 10, "
            "'ratio': {'numer': 2, 'denom': 5}, "
            "'factor': 3.0}, "
            "raw_UnitArray([('muffin', 1, 1)]), "
            "raw_UnitArray([('cookie', 1, 1)]))")


def test_str():
    from pyfu.units import mm
    assert str([] * mm ** 3) == '[] mm^3'
    assert str([2, 3, 5] * mm) == '[ 2.  3.  5.] mm'


def test_array_dtype():
    from pyfu.units import dekahertz, s
    a = np.array(s * [1, 2, 3] * dekahertz, dtype=np.complex)
    a += 1j
    assert array_equals(
        a,
        [10 + 1j, 20 + 1j, 30 + 1j])

    b = np.array(s * [1, 2, 3] * dekahertz, dtype=np.float64)
    with pytest.raises(TypeError):
        b += 1j

    c = np.array(s * [1, 2, 3] * dekahertz)  # infer not complex
    with pytest.raises(TypeError):
        c += 1j


def test_multi_index():
    from pyfu.units import m

    assert (m * [[2, 3], [4, 5], [6, 7]])[0, 0] == m * 2

    assert array_equals(
        (m * [[2, 3, 4], [5, 6, 7], [8, 9, 10]])[1:3, 0:2],
        m * [[5, 6], [8, 9]])

    with pytest.raises(IndexError):
        _ = (m * [[2, 3, 4], [5, 6, 7], [8, 9, 10]])[1:3, 25483]


def test_predicate_index():
    from pyfu.units import m

    v = m * [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
    assert array_equals(v[v < 6 * m], [2, 3, 4, 5] * m)


def test_extract_unit():
    from pyfu.units import m

    # Singleton.
    u = ValueArray(np.array([m * 2]).reshape(()))
    assert u.base_units == m.base_units
    assert array_equals(u, np.array([2]).reshape(()) * m)

    # A line.
    u = ValueArray([m * 2, m * 3])
    assert u.base_units == m.base_units
    assert array_equals(u, [2, 3] * m)

    # Multidimensional.
    u = ValueArray(np.array([[m * 2, m * 3], [m * 4, m * 5]]))
    assert u.base_units == m.base_units
    assert array_equals(u, np.array([[2, 3], [4, 5]]) * m)


def test_numpy_kron():
    from pyfu.units import km, ns

    u = km * [2, 3, 5]
    v = ns * [7, 11]
    w = np.kron(u, v)
    c = km * ns
    assert array_equals(
        w,
        [14 * c, 22 * c, 21 * c, 33 * c, 35 * c, 55 * c])
