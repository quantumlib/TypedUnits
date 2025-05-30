# Copyright 2024 The TUnits Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import pickle

import numpy as np
import pytest
from tunits.core import raw_WithUnit, raw_UnitArray

from tunits import ValueArray, UnitMismatchError, Value
import tunits as tu


def test_construction() -> None:
    from tunits.units import ns, ps

    assert isinstance(ns * [1, 2, 3], ValueArray)
    assert np.array_equal(ns * [1, 2, 3], ps * [1000, 2000, 3000])


def test_slicing() -> None:
    from tunits.units import ms, ns

    assert np.array_equal((ms * [0, 1, 2, 3, 4])[3:], ms * [3, 4])
    assert np.array_equal((ns * [0, 1, 2, 3, 4])[::2], ns * [0, 2, 4])


def test_set_item() -> None:
    from tunits.units import km, m, s

    v = m * [1, 2, 3]

    with pytest.raises(UnitMismatchError):
        v[0] = 2 * s

    v[0] = 2 * km
    v[1] = 16 * m

    assert np.array_equal(v, m * [2000, 16, 3])


def test_addition() -> None:
    from tunits.units import km, m

    assert np.array_equal(km * [1, 2, 3] + m * [2, 3, 5], m * [1002, 2003, 3005])

    assert np.array_equal(km * [1, 2, 3] + 5 * m, m * [1005, 2005, 3005])

    with pytest.raises(UnitMismatchError):
        _ = 1.0 + km * [1, 2, 3]


def test_multiplication() -> None:
    from tunits.units import km, m

    assert np.array_equal((km * [1, 2, 3]) * (m * [2, 3, 5]), (km * m) * [2, 6, 15])

    assert np.array_equal((km * [1, 2, 3]) * 5j, km * [5j, 10j, 15j])

    u = km * [1, 2, 3]
    assert isinstance(u, ValueArray)
    assert u.display_units == km.display_units
    assert u[1] == 2 * km

    u = km * np.array([1.0, 2, 3])
    assert isinstance(u, ValueArray)
    assert u.display_units == km.display_units
    assert u[1] == 2 * km


def test_power() -> None:
    from tunits.units import s

    assert np.array_equal((s * [1, 2, 3]) ** 2, s * s * [1, 4, 9])


def test_dimensionless_act_like_arrays() -> None:
    x = ValueArray(1.5 * np.arange(5), '')
    y = ValueArray(1.5 * np.arange(5), 'us/ns')

    np.testing.assert_allclose(np.sqrt(x), np.sqrt(1.5 * np.arange(5)))
    np.testing.assert_allclose(np.sin(y), np.sin(1.5 * np.arange(5) * 1000))


def test_repr() -> None:
    from tunits.units import km, kg, s

    assert repr(s * []) == "ValueArray(array([], dtype=float64), 's')"
    assert repr(km * [2, 3]) == "ValueArray(array([2., 3.]), 'km')"
    assert repr(km * kg * [3j]) == "ValueArray(array([0.+3.j]), 'kg*km')"
    assert repr(km**2 * [-1] / kg**3 * s) == "ValueArray(array([-1.]), 'km^2*s/kg^3')"
    assert repr(km ** (2 / 3.0) * [-1] / kg**3 * s) == "ValueArray(array([-1.]), 'km^(2/3)*s/kg^3')"

    expected_repr = f"ValueArray({repr(np.array(range(50000), dtype=float))}, 'km')"
    assert repr(list(range(50000)) * km) == expected_repr

    # Fallback case.
    v: ValueArray = raw_WithUnit(
        [1, 2, 3],
        {
            'factor': 3.0,
            'ratio': {
                'numer': 2,
                'denom': 5,
            },
            'exp10': 10,
        },
        raw_UnitArray([('muffin', 1, 1)]),
        raw_UnitArray([('cookie', 1, 1)]),
        Value,
        ValueArray,
    )
    assert (
        repr(v) == "raw_WithUnit(array([1, 2, 3]), "
        "{'factor': 3.0, "
        "'ratio': {'numer': 2, 'denom': 5}, "
        "'exp10': 10}, "
        "raw_UnitArray([('muffin', 1, 1)]), "
        "raw_UnitArray([('cookie', 1, 1)]), Value, ValueArray)"
    )


def test_str() -> None:
    from tunits.units import mm

    assert str(mm**3 * []) == '[] mm^3'
    assert str(mm * [2, 3, 5]) == '[2. 3. 5.] mm'


def test_array_dtype() -> None:
    from tunits.units import dekahertz, s

    a = np.array(s * [1, 2, 3] * dekahertz, dtype=complex)
    a += 1j
    assert np.array_equal(a, [10 + 1j, 20 + 1j, 30 + 1j])

    b = np.array(s * [1, 2, 3] * dekahertz, dtype=np.float64)
    with pytest.raises(TypeError):
        b += 1j  # type: ignore

    c = np.array(s * [1, 2, 3] * dekahertz)  # infer not complex
    with pytest.raises(TypeError):
        c += 1j


def test_multi_index() -> None:
    from tunits.units import m

    assert (m * [[2, 3], [4, 5], [6, 7]])[0, 0] == m * 2

    assert np.array_equal((m * [[2, 3, 4], [5, 6, 7], [8, 9, 10]])[1:3, 0:2], m * [[5, 6], [8, 9]])

    with pytest.raises(IndexError):
        _ = (m * [[2, 3, 4], [5, 6, 7], [8, 9, 10]])[1:3, 25483]

    idx = np.argmax([1, 4, -1])
    assert (m * [1, 2, 4])[idx] == m * 2


def test_predicate_index() -> None:
    from tunits.units import m

    v = m * [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
    assert np.array_equal(v[v < 6 * m], m * [2, 3, 4, 5])


def test_extract_unit() -> None:
    from tunits.units import m

    # Singleton.
    u = ValueArray(np.array([m * 2]).reshape(()))
    assert u.base_units == m.base_units
    assert np.array_equal(u, np.array([2]).reshape(()) * m)

    # A line.
    u = ValueArray([m * 2, m * 3])
    assert u.base_units == m.base_units
    assert np.array_equal(u, m * [2, 3])

    # Multidimensional.
    u = ValueArray(np.array([[m * 2, m * 3], [m * 4, m * 5]]))
    assert u.base_units == m.base_units
    assert np.array_equal(u, np.array([[2, 3], [4, 5]]) * m)


def test_numpy_kron() -> None:
    from tunits.units import km, ns

    u = km * [2, 3, 5]
    v = ns * [7, 11]
    w = np.kron(u, v)
    c = km * ns
    assert np.array_equal(w, [14 * c, 22 * c, 21 * c, 33 * c, 35 * c, 55 * c])


def test_multiplication_with_dimensionless_preserves_ratios() -> None:
    A, B = ValueArray([1], 'GHz^2'), ValueArray([1200], 'MHz/GHz')
    assert A * B == B * A == ValueArray([1.2], 'GHz^2')


def test_divison_with_dimensionless_preserves_ratios() -> None:
    A, B = ValueArray([1], 'GHz^2'), ValueArray([1200], 'MHz/GHz')
    assert B / A == ValueArray([1.2], 'GHz^-2')

    assert A / B == ValueArray([10 / 12], 'GHz^2')


def test_units() -> None:
    A, B = ValueArray(np.random.random(10), 'GHz^2'), ValueArray(np.random.random(10), 'MHz/GHz')
    assert A.units == 'GHz^2'
    assert B.units == 'MHz/GHz'


def test_base_unit() -> None:
    A, B = ValueArray(np.random.random(10), 'GHz^2'), ValueArray(np.random.random(10), 'MHz/GHz')
    assert A.base_unit == Value(1, 'Hz^2')
    assert B.base_unit == Value(1, '')


def test_sign() -> None:
    for x in np.random.random((10, 3, 4)):
        v = ValueArray(x, 'ns')
        np.testing.assert_equal(v.sign(), np.sign(x))


def test_dimensionless() -> None:
    A, B = ValueArray([1, 2, 3], 'GHz^2'), ValueArray(np.arange(5), 'MHz/GHz')

    with pytest.raises(ValueError):
        _ = A.dimensionless()

    np.testing.assert_equal(B.dimensionless(), np.arange(5) / 1000)


def test_pick_roundtrip() -> None:
    units = itertools.product(
        [tu.ns, tu.GHz, tu.dBm, tu.deg, 1, tu.GHz**0.5, tu.m**0.75, tu.s**0.25], repeat=3
    )
    for value in np.random.random((5, 20)):
        for unit_list in units:
            unit = np.prod(unit_list)  # type: ignore[arg-type]
            x = value * unit
            s = pickle.dumps(x)
            assert all(x == pickle.loads(s))


class _InvalidUnit:
    pass


def test_invalid_unit_raises_error() -> None:
    with pytest.raises(ValueError):
        _ = tu.ValueArray([1, 2], _InvalidUnit())


def test_format() -> None:
    u = tu.GHz * np.random.random(10)
    assert f'{u}' == str(u)


def test_ufunc() -> None:
    x = np.float64(0.42)
    y = tu.GHz * np.arange(4)[1:]

    assert np.multiply(x, y).allclose(np.multiply(y, x))  # type: ignore[union-attr]
    assert np.divide(x, y).allclose(np.divide(np.int64(1), np.divide(y, x)))  # type: ignore[union-attr]

    with pytest.raises(UnitMismatchError):
        _ = np.add(x, y)
    with pytest.raises(UnitMismatchError):
        _ = np.add(y, x)

    with pytest.raises(UnitMismatchError):
        _ = np.subtract(x, y)
    with pytest.raises(UnitMismatchError):
        _ = np.subtract(y, x)
