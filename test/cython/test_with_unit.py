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

from typing import Any
import copy

import numpy as np
import pytest

from tunits.core import raw_WithUnit, raw_UnitArray, WithUnit

from tunits import UnitMismatchError, ValueArray, Value
from test.test_utils import frac, conv, val

dimensionless = raw_UnitArray([])
s = raw_UnitArray([('s', 1, 1)])
rad = raw_UnitArray([('rad', 1, 1)])
h = raw_UnitArray([('s', 3600, 1)])  # Note: is s**3600, not 3600 seconds.
m = raw_UnitArray([('m', 1, 1)])
kg = raw_UnitArray([('kg', 1, 1)])
mps = raw_UnitArray([('m', 1, 1), ('s', -1, 1)])
kph = raw_UnitArray([('m', 1000, 1), ('s', -1, 3600)])


def deep_equal(a: WithUnit, b: WithUnit) -> bool:
    """Compares value, units, numerator, denominator, and exponent.

    Args:
       a (WithUnit): First object
       b (WithUnit): Second object

    Returns:
       bool, whether a and b have exactly the same properties.
    """
    if isinstance(a, ValueArray):
        if not np.array_equal(a, b):
            return False
    else:
        if a != b:
            return False
    if isinstance(a, ValueArray):
        if not np.array_equal(a.value, b.value):
            return False
    else:
        if a.value != b.value:
            return False
    return a.numer == b.numer and a.denom == b.denom and a.factor == b.factor and a.exp10 == b.exp10


def test_raw_versus_properties() -> None:
    x = val(2, conv(factor=3, numer=4, denom=5, exp10=6), mps, kph)
    assert x.value == 2
    assert x.factor == 3
    assert x.numer == 4
    assert x.denom == 5
    assert x.exp10 == 6
    assert x.base_units == mps
    assert x.display_units == kph

    assert isinstance(val(2), Value)
    assert isinstance(val(2j), Value)
    assert isinstance(val([2]), ValueArray)


def test_abs() -> None:
    assert abs(val(2)) == val(2)

    # If we have a negative unit, abs is w.r.t. the derived unit.
    assert abs(val(-2)) == val(2)
    assert abs(val(2, conv(-1.5))) == val(-3)
    assert abs(val(2, conv(numer=-2))) == val(-4)
    assert abs(val(2, conv(-1.5, numer=-2))) == val(6)


def test_equality() -> None:
    equivalence_groups: list[list[Any]] = [
        [""],
        ["other types"],
        [list],
        [None],
        [dimensionless],
        # Wrapped values equal unwrapped values.
        [0, val(0)],
        [2, val(2)],
        [1 + 2j, val(1 + 2j)],
        [2.5, val(2.5)],
        # Units matter.
        [1, val(1)],
        [val(1, units=s)],
        [val(1, units=m)],
        # The display unit *text* doesn't matter to equality.
        [
            val(1, conv(factor=2, numer=5, denom=4, exp10=5), mps, kph),
            val(1, conv(factor=2, numer=5, denom=4, exp10=5), mps, s),
        ],
        [
            val(-1, units=s, display_units=m),
            val(-1, units=s, display_units=s),
        ],
        # Varying each parameter causes differences.
        [val(9, conv(factor=2, numer=5, denom=4, exp10=5), mps, kph)],
        [val(1, conv(factor=9, numer=5, denom=4, exp10=5), mps, kph)],
        [val(1, conv(factor=2, numer=9, denom=4, exp10=5), mps, kph)],
        [val(1, conv(factor=2, numer=5, denom=9, exp10=5), mps, kph)],
        [val(1, conv(factor=2, numer=5, denom=4, exp10=9), mps, kph)],
        [val(1, conv(factor=2, numer=5, denom=4, exp10=5), s, kph)],
        # You can trade between parameters.
        [
            val(10, conv(factor=2, numer=3, denom=4, exp10=5), mps, kph),
            val(1, conv(factor=20, numer=3, denom=4, exp10=5), mps, kph),
            val(1, conv(factor=2, numer=30, denom=4, exp10=5), mps, kph),
            val(1, conv(factor=2, numer=3, denom=4, exp10=6), mps, kph),
            val(1, conv(factor=2, numer=3, denom=40, exp10=7), mps, kph),
        ],
        [val(1, conv(factor=2, numer=3, denom=4, exp10=5), mps, kph)],
        [val(3, units=s, display_units=h), val(3, units=s)],
        [val(3, units=h)],
    ]
    for g1 in equivalence_groups:
        for g2 in equivalence_groups:
            for e1 in g1:
                for e2 in g2:
                    match = g1 is g2
                    if match:
                        assert e1 == e2
                    else:
                        assert e1 != e2


def test_ordering() -> None:
    with pytest.raises(UnitMismatchError):
        _ = val(1) < val(1, units=m)
        _ = val(1) > val(1, units=m)
        _ = val(1) <= val(1, units=m)
        _ = val(1) >= val(1, units=m)

    ascending_groups = [
        [val(0, units=m)],
        [val(1, units=m)],
        [val(2, units=m), val(1, conv(2), units=m, display_units=s)],
        [val(3.14, conv(2, 2, 3, 1), units=m, display_units=s)],
    ]

    for i in range(len(ascending_groups)):
        for a in ascending_groups[i]:
            for b in ascending_groups[i]:
                assert a <= b
                assert a >= b
                assert b <= a
                assert b >= a
                assert not (a < b)
                assert not (a > b)
                assert not (b < a)
                assert not (b > a)

            for j in range(len(ascending_groups))[i + 1 :]:
                for b in ascending_groups[j]:
                    assert a < b
                    assert a <= b
                    assert b > a
                    assert b >= a
                    assert not (b < a)
                    assert not (b <= a)
                    assert not (a > b)
                    assert not (a >= b)


def test_array_equality() -> None:
    assert np.array_equal(5 == val([]), [])
    assert np.array_equal([] == val([], conv(2), mps), [])
    assert np.array_equal([1] == val([0]), [False])
    assert np.array_equal([1] == val([1]), [True])
    assert np.array_equal([1, 2] == val([3, 4]), [False, False])
    assert np.array_equal([1, 2] == val(np.array([3, 4])), [False, False])
    assert np.array_equal([1, 2] == val([1, 2]), [True, True])
    assert np.array_equal([1, 2] == val([9, 2]), [False, True])
    assert np.array_equal([1, 2] == val([1, 9]), [True, False])
    assert np.array_equal([1, 2] == val([1, 2], units=m), [False, False])
    assert np.array_equal(val([1, 2], units=s) == val([1, 2], units=m), [False, False])
    assert np.array_equal(val([1, 2], units=m) == val([1, 2], units=m), [True, True])
    assert np.array_equal([1, 2] == val([0.5, 1], conv(2)), [True, True])


def test_array_ordering() -> None:
    with pytest.raises(UnitMismatchError):
        _ = val([]) < val([], units=m)
        _ = val([]) > val([], units=m)
        _ = val([]) <= val([], units=m)
        _ = val([]) >= val([], units=m)

    assert np.array_equal(val([2, 3, 4], units=m) < val([3, 3, 3], units=m), [True, False, False])
    assert np.array_equal(val([2, 3, 4], units=m) <= val([3, 3, 3], units=m), [True, True, False])
    assert np.array_equal(val([2, 3, 4], units=m) >= val([3, 3, 3], units=m), [False, True, True])
    assert np.array_equal(val([2, 3, 4], units=m) > val([3, 3, 3], units=m), [False, False, True])


def test_int() -> None:
    with pytest.raises(TypeError):
        int(val(1, units=mps))
    with pytest.raises(TypeError):
        int(val(1j))
    with pytest.raises(TypeError):
        int(val([1, 2]))

    u = int(val(5))
    assert isinstance(u, int)
    assert u == 5

    u = int(val(2.5))
    assert isinstance(u, int)
    assert u == 2

    u = int(val(2.5, conv(2.5)))
    assert isinstance(u, int)
    assert u == 6

    u = int(val(5, conv(2, 3, 4, 5)))
    assert isinstance(u, int)
    assert u == 750000


def test_float() -> None:
    with pytest.raises(TypeError):
        float(val(1, units=mps))
    with pytest.raises(TypeError):
        float(val(1j))
    with pytest.raises(TypeError):
        float(val([1, 2]))

    u = float(val(5))
    assert isinstance(u, float)
    assert u == 5

    u = float(val(2.5))
    assert isinstance(u, float)
    assert u == 2.5

    u = float(val(5, conv(2, 3, 4, 5)))
    assert isinstance(u, float)
    assert u == 750000


def test_complex() -> None:
    with pytest.raises(TypeError):
        complex(val(1j, units=m))
    with pytest.raises(TypeError):
        complex(val([1, 2]))

    u = complex(val(5))
    assert isinstance(u, complex)
    assert u == 5

    v = complex(val(5 + 6j, conv(2, 3, 4, 5)))
    assert isinstance(v, complex)
    assert v == 750000 + 900000j


def test_array() -> None:
    u = np.array(val([1, 2], units=m))
    assert isinstance(u, np.ndarray)
    assert isinstance(u[0], Value)
    assert np.array_equal([val(1, units=m), val(2, units=m)], u)

    u = np.array(val([val(2, units=m), val(3, units=m)]))
    assert isinstance(u, np.ndarray)
    assert isinstance(u[0], float)
    assert np.array_equal([2, 3], u)

    u = np.array(val([2, 3]))
    assert isinstance(u, np.ndarray)
    assert np.array_equal([2, 3], u)

    u = np.array(val([2, 3 + 1j], conv(2, 3, 4, 5)))
    assert isinstance(u, np.ndarray)
    assert np.array_equal([300000, 450000 + 150000j], u)


def test_copy() -> None:
    a = val(2, conv(3, 4, 5, 6), mps, kph)
    assert a is copy.copy(a)
    assert a is copy.deepcopy(a)

    b = val(1 + 2j, conv(3, 4, 5, 6), mps, kph)
    assert b is copy.copy(b)
    assert b is copy.deepcopy(b)

    c = val([10, 11], conv(3, 4, 5, 6), mps, kph)
    assert c is not copy.copy(c)
    assert c is not copy.deepcopy(c)
    assert np.array_equal(c, copy.copy(c))
    assert np.array_equal(c, copy.deepcopy(c))

    # Copy can be edited independently.
    c2 = copy.copy(c)
    c[1] = val(42, units=mps)
    assert not np.array_equal(c, c2)


def test_addition() -> None:
    with pytest.raises(UnitMismatchError):
        _ = val(2, units=m) + val(3, units=s)
    with pytest.raises(UnitMismatchError):
        _ = val(2, units=m) + val(3, units=m * m)
    with pytest.raises(UnitMismatchError):
        _ = val(2, units=m) + 3

    assert val(2) + val(3 + 1j) == 5 + 1j
    assert np.array_equal(val(2) + val([2, 3]), val([4, 5]))

    a = val(7, conv(5), units=s)
    b = val(3, units=s)
    assert a + b == val(38, units=s)

    # Prefers using the finer-grained conversion in the result.
    a = val(7, conv(5), units=s, display_units=kg)
    b = val(3, conv(4), units=s, display_units=m)
    c = val(11.75, conv(4), units=s, display_units=kg)
    assert deep_equal(a + b, c)
    assert deep_equal(b + a, c)

    assert val(2, conv(3, 4, 5, 6)) + val(7) == 4800007

    v1 = val(1, units=m)
    v2 = val(1, conv(exp10=3), units=m, display_units=s)
    assert 1.0 * v2 / v1 + 5.0 == 1005

    # Tricky precision.
    a = val(1, conv(denom=101))
    b = val(1, conv(denom=101 * 103))
    assert deep_equal(a + b, val(104, conv(denom=101 * 103)))
    assert deep_equal(b + a, val(104, conv(denom=101 * 103)))

    # Adding dimensionless zero is fine, even if units don't match.
    assert deep_equal(val(3, units=s) + 0, val(3, units=s))
    assert deep_equal(0.0 + val(3, units=s), val(3, units=s))
    with pytest.raises(UnitMismatchError):
        _ = val(3, units=s) + val(0, units=m)
        _ = val(0, units=s) + val(3, units=m)


def test_subtraction() -> None:
    with pytest.raises(UnitMismatchError):
        _ = val(2, units=m) - val(3, units=s)
    with pytest.raises(UnitMismatchError):
        _ = val(2, units=m) - val(3, units=m * m)
    with pytest.raises(UnitMismatchError):
        _ = val(2, units=m) - 3

    assert val(2) - val(5 + 1j) == -3 - 1j
    assert np.array_equal(val(2) - val([2, 3]), val([0, -1]))

    a = val(7, conv(5), units=s)
    b = val(3, units=s)
    assert a - b == val(32, units=s)

    # Subtracting dimensionless zero is fine, even if units don't match.
    assert deep_equal(val(3, units=s) - 0, val(3, units=s))
    assert deep_equal(0.0 - val(3, units=s), val(-3, units=s))
    with pytest.raises(UnitMismatchError):
        _ = val(3, units=s) - val(0, units=m)
        _ = val(0, units=s) - val(3, units=m)


def test_multiplication() -> None:
    assert val(2) * val(5) == 10

    assert deep_equal(val(2, units=m) * val(3, units=s), val(6, units=m * s))

    assert deep_equal(
        (val(2, conv(3, 4, 5, 6), m, s) * val(7, conv(8, 9, 10, 11), mps, kph)),
        val(14, conv(24, 18, 25, 17), m * mps, s * kph),
    )


def test_division() -> None:
    assert val(5) / val(2) == 2.5

    assert deep_equal(val(7, units=m) / val(4, units=s), val(1.75, units=m / s))

    assert deep_equal(
        (val(7, conv(3, 9, 10, 11), mps, kph) / val(2, conv(8, 4, 5, 6), m, s)),
        val(3.5, conv(0.375, 9, 8, 5), mps / m, kph / s),
    )


def test_int_division() -> None:
    with pytest.raises(UnitMismatchError):
        _ = val(1, units=m) // val(1, units=s)

    assert isinstance(val(5) // val(2), float)
    assert isinstance(val(7, units=m) // val(4, units=m), float)

    assert val(5) // val(2) == 2
    assert val(-5) // val(-2) == 2
    assert val(-5) // val(2) == -3
    assert val(5) // val(-2) == -3

    assert val(7, units=m) // val(4, units=m) == 1

    assert val(7, conv(2), m, s) // val(4, units=m, display_units=h) == 3


def test_mod() -> None:
    with pytest.raises(UnitMismatchError):
        _ = val(1, units=m) % val(1, units=s)

    assert deep_equal(val(5) % val(3), val(2))
    assert deep_equal(val(-5) % val(3), val(1))
    assert deep_equal(val(7, units=m) % val(4, units=m), val(3, units=m))

    assert deep_equal(val(7, conv(3), m, s) % val(4, conv(2), m, h), val(2.5, conv(2), m, h))


def test_div_mod() -> None:
    with pytest.raises(UnitMismatchError):
        _ = divmod(val(1, units=m), val(1, units=s))

    q, r = divmod(val(7, conv(3), m, s), val(4, conv(2), m, h))
    assert q == 2
    assert deep_equal(r, val(2.5, conv(2), m, h))


def test_pow() -> None:
    with pytest.raises(TypeError):
        _ = 2 ** val(1, units=m)  # type: ignore

    assert deep_equal(val(2, units=m) ** -2, val(0.25, units=m**-2))
    assert deep_equal(val(2, units=m) ** -1, val(0.5, units=m**-1))
    assert deep_equal(val(2, units=m) ** 0, val(1))
    assert deep_equal(val(2, units=m) ** 1, val(2, units=m))
    assert deep_equal(val(2, units=m) ** 2, val(4, units=m**2))
    assert deep_equal(val(2, units=m) ** 3, val(8, units=m**3))

    assert deep_equal(val(4, units=m) ** -0.5, val(0.5, units=m**-0.5))
    assert deep_equal(val(4, units=m) ** 0.5, val(2, units=m**0.5))
    assert deep_equal(val(4, units=m) ** 1.5, val(8, units=m**1.5))

    # Fractional powers that should work.
    for i in [1, 2, 3, 4, 6, 12]:
        assert deep_equal(val(2**i, units=m**i) ** (1.0 / i), val(2, units=m))

    # Conversion keeping/losing precision.
    assert deep_equal(val(4, conv(numer=4)) ** 0.5, val(2, conv(numer=2)))
    assert deep_equal(val(4, conv(numer=2)) ** 0.5, val(2, conv(factor=2**0.5)))


def test_pos() -> None:
    assert deep_equal(+val(2, conv(3, 5, 7, 11), mps, kph), val(2, conv(3, 5, 7, 11), mps, kph))


def test_neg() -> None:
    assert deep_equal(-val(2, conv(3, 5, 7, 11), mps, kph), val(-2, conv(3, 5, 7, 11), mps, kph))


def test_non_zero() -> None:
    assert not bool(val(0, conv(3, 5, 7, 11), mps, kph))
    assert not bool(val(0))
    assert bool(val(2, conv(3, 5, 7, 11), mps, kph))
    assert bool(val(2))
    assert bool(val([1]))
    assert np.size(val([])) == 0


def test_numpy_method_is_finite() -> None:
    with pytest.raises(TypeError):
        np.isfinite(val([], units=m))
    with pytest.raises(TypeError):
        np.isfinite(val([1], units=m))

    v = val([2, 3, -2, float('nan'), float('inf')], conv(1, 2, 3, 4))
    assert np.array_equal(np.isfinite(v), [True, True, True, False, False])

    v = val([[2, 3], [-2, float('nan')]])
    assert np.array_equal(np.isfinite(v), [[True, True], [True, False]])


def test_get_item() -> None:
    u = val(1, conv(exp10=1))
    v = val(2, conv(numer=3, denom=5, exp10=7), mps, kph)

    # Wrong kinds of index (unit array, slice).
    with pytest.raises(TypeError):
        _ = u[mps]
    with pytest.raises(TypeError):
        _ = u[1:2]

    # Safety against dimensionless unit ambiguity.
    with pytest.raises(TypeError):
        _ = u[1.0]
    with pytest.raises(TypeError):
        _ = u[1.0]
        _ = u[u]
    with pytest.raises(TypeError):
        _ = u[1]
    with pytest.raises(TypeError):
        _ = u[2 * v / v]
    assert u[v / v] == 10

    # Wrong unit.
    with pytest.raises(UnitMismatchError):
        _ = u[v]

    assert u[''] == 10
    assert v[v] == 1
    assert v[val(2, conv(denom=5, exp10=7), mps, kph)] == 3
    assert v[val(2, conv(numer=3, exp10=7), mps, kph)] == 0.2
    assert val(2, conv(numer=3, exp10=7), mps, kph)[v] == 5


def test_iter() -> None:
    a = []
    for e in val([1, 2, 4], conv(numer=2), mps, kph):
        a.append(e)
    assert len(a) == 3

    assert deep_equal(a[0], val(1, conv(numer=2), mps, kph))
    assert deep_equal(a[1], val(2, conv(numer=2), mps, kph))
    assert deep_equal(a[2], val(4, conv(numer=2), mps, kph))


def test_hash() -> None:
    d: dict[Any, Any] = dict()
    v = val(2, conv(denom=5), mps, kph)
    w = val(3, conv(exp10=7), mps, kph)
    d[v] = 5
    d[w] = "b"
    assert d[v] == 5

    assert d[w] == "b"


def test_str() -> None:
    assert str(val(2, conv(3, 4, 5, 6), s, m)) == '2.0 m'
    assert str(val(2j, conv(3, 4, 5, 6), s)) == '2j s'
    assert str(val(1, units=m, display_units=s)) == 's'
    assert str(val(1, units=m)) == 'm'
    assert str(val(1, units=h)) == 's^3600'
    assert str(val([2, 3, 5], units=h, display_units=m)) == '[2 3 5] m'
    assert str(val([2, 3], units=h, display_units=mps)) == '[2 3] m/s'


def test_is_compatible() -> None:
    equivalence_groups = [
        [val(0), val(1)],
        [val(1, units=m), val(5, conv(3), units=m, display_units=s)],
        [val(9, units=s)],
        [val(13, units=m**2), val(5, units=m**2)],
    ]
    for g1 in equivalence_groups:
        for g2 in equivalence_groups:
            for e1 in g1:
                for e2 in g2:
                    assert e1.isCompatible(e2) == (g1 is g2)


def test_is_angle() -> None:
    assert not val(1).isAngle()
    assert not val(1).is_angle
    assert not val(1, units=m).isAngle()
    assert not val(1, units=m).is_angle
    assert val(1, units=rad).isAngle()
    assert val(1, units=rad).is_angle
    assert not val(1, units=rad**2).isAngle()
    assert not val(1, units=rad**2).is_angle


def test_in_units_of() -> None:
    with pytest.raises(UnitMismatchError):
        val(1, units=m).inUnitsOf(val(2, units=s))

    assert val(5).inUnitsOf(8) == 0.625
    assert deep_equal(
        val(5, conv(3), units=m, display_units=s).inUnitsOf(
            val(8, conv(denom=7), units=m, display_units=kg)
        ),
        val(13.125, conv(denom=7), units=m, display_units=kg),
    )


def test_unit() -> None:
    assert deep_equal(val(7, conv(2, 3, 4, 5), m, s).unit, val(1, conv(2, 3, 4, 5), m, s))
