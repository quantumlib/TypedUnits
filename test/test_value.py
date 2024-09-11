import numbers

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

import numpy as np
import pytest

from tunits import Value, UnitMismatchError


def test_construction() -> None:
    """
    can construct values with or without dimensions
    """
    x = 2 * Value(1, '')
    y = Value(5, 'ns')
    assert isinstance(x, Value)
    assert isinstance(y, Value)
    assert x.is_dimensionless
    assert isinstance(3j * x, Value)


def test_dimensionless() -> None:
    """Test that dimensionless values act like floats"""
    x = Value(1.5, '')
    y = Value(1.5, 'us/ns')
    assert x == 1.5
    assert np.ceil(x) == 2.0
    assert np.floor(x) == 1.0
    assert y == 1500.0


def test_addition() -> None:
    """
    addition and subtraction only make sense when units can be made to match
    """
    from tunits.units import kilometer

    n = Value(2, '')
    x = Value(1.0, kilometer)
    y = Value(3, 'meter')
    a = Value(20, 's')
    assert x + y == Value(1003, 'meter')
    assert x != y
    assert x != a
    with pytest.raises(UnitMismatchError):
        _ = y + a
    with pytest.raises(UnitMismatchError):
        _ = x + 3.0
    _ = x + y
    assert x - y == Value(997, 'm')
    assert isinstance(x * 1j + y, Value)
    assert n + 1 == 3


def test_multiplication() -> None:
    """
    multiplication/division makes sense regardless
    the units become the product/quotient unit
    cancellation to give something dimensionless where relevant
    """
    from tunits.units import meter, mm, second

    x = Value(1.0 + 2j, meter)
    y = Value(3, mm)
    a = Value(20, second)
    assert a * x == x * a
    assert (x / y).is_dimensionless


def test_power() -> None:
    """
    can raise to numeric powers
    """
    from tunits.units import km, m, minute, s, um, mm

    _ = mm * np.complex128(3)
    _ = mm * np.complex64(3)
    _ = mm * np.uint64(3)
    _ = mm * np.int64(3)
    x = 2 * mm
    y = 4 * mm
    z = (x * y) ** 0.5
    assert abs(z**2 - Value(8, 'mm^2')) < Value(1e-6, mm**2)
    assert Value(16000000, 'um^2') ** 0.5 == 4 * mm
    assert (16 * um * m) ** 0.5 == 4 * mm
    assert (minute**2) ** 0.5 == minute
    assert (1000 * m * km) ** 0.5 == km
    assert np.isclose((60 * s * minute) ** 0.5 / s, minute / s)


def test_repr() -> None:
    """
    the repr for a Value has the numeric value and the unit information
    """
    from tunits.units import km, kg, mm

    assert repr(Value(1, mm)) == "Value(1, 'mm')"
    assert repr(Value(4, mm)) == "Value(4, 'mm')"
    assert repr(Value(1j + 5, km * kg)) == "Value((5+1j), 'kg*km')"


def test_str() -> None:
    """
    the displayable str uses the
    appropriate string representation for the units
    """
    from tunits.units import mm, meter, kilometer, rad, cyc

    assert str(Value(1, mm)) == 'mm'
    assert str(Value(4, mm)) == '4 mm'
    assert str(2 * meter * kilometer) == '2 km*m'
    assert str(cyc) == 'cyc'
    assert str(3.25 * cyc**2) == '3.25 cyc^2'
    assert str(3.25 * cyc * rad) == '3.25 cyc*rad'
    assert str((4 * kilometer) ** 0.5) == '2.0 km^(1/2)'


def test_div_mod() -> None:
    """
    when the operands match units
    truncated division and remainder
    the division is like normal division with dimensionless answer
    the remainder retains the unit
    """
    from tunits.units import us, ns, m

    x = 4.0009765625 * us
    assert x // (4 * ns) == 1000
    assert x % (4 * ns) == 0.9765625 * ns
    q, r = divmod(x, 2 * ns)
    assert q == 2000
    assert r == x - 4 * us
    with pytest.raises(UnitMismatchError):
        _ = x // (4 * m)


def test_conversion() -> None:
    """
    can convert Values to other units with dictionary syntax
    or explicit in_units_of
    if the desired units of does not match, then a UnitMismatchError
    """
    from tunits.units import mm

    x = Value(3, 'm')
    assert x['mm'] == 3000.0
    assert x[mm] == 3000.0
    with pytest.raises(UnitMismatchError):
        _ = x['s']
    y = Value(1000, 'Mg')
    assert y.in_base_units().value == 1000000.0
    assert x.in_units_of('mm') == 3000 * mm


def test_parsing_by_comparison() -> None:
    """
    with potential conversion factors required
    we can compare two values with matching dimensional analysis
    with inequalities and equalities
    UnitMismatchError if they are not
    """
    assert Value(1, 'in') < Value(1, 'm')
    assert Value(1, 'cm') < Value(1, 'in')
    assert Value(1, 'gauss') < Value(1, 'mT')
    assert Value(1, 'minute') < Value(100, 's')
    with pytest.raises(UnitMismatchError):
        _ = Value(1, 'minute') < Value(100, 'm')

    assert Value(10, 'hertz') == Value(10, 'Hz')
    assert Value(10, 'Mg') == Value(10000, 'kg')
    assert Value(10, 'Mg') == Value(10000000, 'g')

    assert Value(10, 'decibel') != Value(10, 'mol')
    assert Value(1, 'millisecond') == Value(1, 'ms')
    assert Value(1, '') == Value(1, 'm/m')


def test_radians_vs_steradians() -> None:
    """
    specific equalities for solid angle measurements
    """
    assert Value(1, 'rad') != Value(1, 'sr')
    assert Value(2, 'rad') ** 2 == Value(4, 'sr')
    assert Value(16, 'rad') == Value(256, 'sr') ** 0.5
    assert Value(32, 'rad') * Value(2, 'sr') == Value(16, 'sr') ** 1.5
    assert Value(1, 'rad') ** (4 / 3.0) == Value(1, 'sr') ** (2 / 3.0)


def test_division() -> None:
    """
    division makes sense regardless of units
    the units become the quotient unit
    cancellation to give something dimensionless where relevant
    when the operands match units
    can also perform truncated division and remainder
    the division is like normal division with dimensionless answer
    the remainder retains the unit
    """
    from tunits.units import km, s, m

    assert 5 * km / (2 * s) == Value(2500, 'm/s')
    with pytest.raises(UnitMismatchError):
        _ = 5 * km // (2 * s)
    assert (5 * km).__truediv__(2 * s) == Value(2500, 'm/s')
    with pytest.raises(UnitMismatchError):
        assert (5 * km).__floordiv__(2 * s) == Value(2500, 'm/s')

    assert (5 * km) / (64 * m) == 78.125
    assert (5 * km) // (64 * m) == 78
    assert (5 * km).__truediv__(64 * m) == 78.125
    assert (5 * km).__floordiv__(64 * m) == 78
    assert (5 * km).__mod__(64 * m) == 8 * m


def test_get_item() -> None:
    """
    in certain cases can use dictionary syntax for conversion
    """
    from tunits.units import ns, s

    with pytest.raises(TypeError):
        _ = (ns / s)[2 * s / ns]
    two_thousand_unitless = Value(2, "s / ns")
    with pytest.raises(TypeError):
        _ = (ns / s)[two_thousand_unitless]
    one_thousandth_unitless = Value(1, "ns / s")
    assert (ns / s)[one_thousandth_unitless] == 1
    with pytest.raises(TypeError):
        _ = (ns / s)[Value(3, '')]
    assert Value(1, '')[Value(1, '')] == 1
    assert Value(1, '')[ns / s] == 10**9


def test_cycles() -> None:
    """
    explicit angle conversions
    """
    from tunits.units import cyc, rad

    assert np.isclose((3.14159265 * rad)[cyc], 0.5)
    assert np.isclose((1.0 * rad)[cyc], 0.15915494309)
    assert np.isclose((1.0 * cyc)[2 * rad], 3.14159265)


def test_decibels_vs_decibel_milliwatts() -> None:
    from tunits.units import dBm, dB, W

    assert dBm != dB * W
    assert not dBm.is_compatible(dB * W)


def test_hash() -> None:
    """
    hash does not depend on how the value is presented
    they can be equal but presented differently because of conversion
    and still give the same hash
    dimensionless quantities behave just like underlying numbers
    """
    x = Value(3, 'ks')
    y = Value(3000, 's')
    assert hash(x) == hash(y)
    z = Value(3.1, '')
    assert hash(z) == hash(3.1)
    assert hash(Value(4j, 'V')) is not None


def test_numpy_sqrt() -> None:
    """
    like in test_power, can raise to the power 1/2
    but this time with np.sqrt instead of ** 0.5
    """
    from tunits.units import m, km, cm

    u = np.sqrt(8 * km * m) - cm
    v = 8943.27191 * cm
    assert np.isclose(u / v, 1)
    u_prime = (8 * km * m) ** 0.5 - cm
    assert np.isclose(u / u_prime, 1)

    u = np.sqrt(8 * km / m)
    assert np.isclose(u, 89.4427191)

    u = np.sqrt((8 * km / m).in_base_units())
    assert np.isclose(u, 89.4427191)
