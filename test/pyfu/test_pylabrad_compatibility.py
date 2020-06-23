import numpy as np
import pytest

import pyfu.like_pylabrad_units as fu
from pyfu.like_pylabrad_units import UnitMismatchError

ValueArray = fu.ValueArray
Value = fu.Value


def test_arithmetic() -> None:
    m = fu.Unit('m')
    kg = fu.Unit('kg')
    km = fu.Unit('km')

    assert fu.Value(5.0, None) * m == 5.0 * m

    # addition
    assert 1.0 * kg + 0.0 * kg == 1.0 * kg
    with pytest.raises(fu.UnitMismatchError):
        _ = 1.0 * kg + 1.0 * m
    with pytest.raises(fu.UnitMismatchError):
        _ = 1.0 * kg + 2.0
    assert km == 1000 * m
    assert 1.0 * km / m + 5.0 == 1005
    assert 1.0 * kg is not None


def test_value_array() -> None:
    # Slicing
    assert (ValueArray([1, 2, 3], 'm')[0:2] == ValueArray([1, 2], 'm')).all()
    # Cast to unit
    assert (ValueArray([1.2, 4, 5], 'm')['m'] == np.array([1.2, 4, 5])).all()
    # Addition and subtraction of compatible units
    assert (ValueArray([3, 4], 'm') + ValueArray([100, 200], 'cm') == ValueArray([4, 6], 'm')).all()
    assert (
        ValueArray([2, 3, 4], 'm') - ValueArray([100, 200, 300], 'cm') == ValueArray([1, 1, 1], 'm')
    ).all()
    # Division with units remaining
    assert (
        ValueArray([3, 4, 5], 'm') / ValueArray([1, 2, 5], 's') == ValueArray([3, 2, 1], 'm/s')
    ).all()
    # Division with no units remaining
    assert (
        ValueArray([3, 4, 5], 'm') / ValueArray([1, 2, 5], 'm') == ValueArray([3, 2, 1], '')
    ).all()
    # Powers
    assert (ValueArray([2, 3], 'm') ** 2 == ValueArray([4, 9], 'm^2')).all()
    assert (ValueArray([2, 3], 'GHz') * Value(3, 'ns')).dtype == np.float64


def test_dimensionless_angle() -> None:
    a = np.array(fu.DimensionlessArray([1, 2, 3]))
    assert len(a) == 3
    assert a[0] == 1
    assert a[1] == 2
    assert a[2] == 3


def test_is_finite() -> None:
    assert np.isfinite(ValueArray([1, 2], '')).all()
    assert (np.isfinite(ValueArray([1, float('nan')], '')) == np.array([True, False])).all()


def test_negative_powers() -> None:
    assert str(fu.Unit('1/s')) in ['s^-1', '1/s']
    assert str(fu.Unit('1/s^1/2')) in ['s^-1/2', '1/s^(1/2)']


def test_type_conversions() -> None:
    m = fu.Unit('m')
    V = fu.Unit('V')
    GHz = fu.Unit('GHz')
    x1 = 1.0 * m
    x2 = 5j * V
    a = np.arange(10) * 1.0
    va = fu.ValueArray(np.arange(10) * 1.0, 'GHz')

    # Unit times number
    assert isinstance(1.0 * m, fu.Value)
    assert isinstance(1 * m, fu.Value)
    assert isinstance(m * 1.0, fu.Value)
    assert isinstance(m * 1, fu.Value)

    # Value times value or number
    assert isinstance(x1 * x1, fu.Value)
    assert isinstance(x1 * 5, fu.Value)
    assert isinstance(0 * x1, fu.Value)

    # Unit times complex
    assert isinstance((1 + 1j) * V, fu.Complex)
    assert isinstance(V * (1 + 1j), fu.Complex)

    # Value times Complex/complex
    assert isinstance(x1 * 1j, fu.Complex)
    assert isinstance(1j * x1, fu.Complex)
    assert isinstance(x2 * x1, fu.Complex)
    assert isinstance(x1 * x2, fu.Complex)

    # Unit/Value/ValueArray times array
    assert isinstance(x1 * a, fu.ValueArray)
    assert isinstance(x2 * a, fu.ValueArray)
    assert isinstance(GHz * a, fu.ValueArray)
    assert isinstance(va * a, fu.ValueArray)

    # Unit/Value/ValueArray times ValueArray
    assert isinstance(x1 * va, fu.ValueArray)
    assert isinstance(x2 * va, fu.ValueArray)
    assert isinstance(GHz * va, fu.ValueArray)
    assert isinstance(va * va, fu.ValueArray)

    # array times ?
    assert isinstance(a * x1, fu.ValueArray)
    assert isinstance(a * x2, fu.ValueArray)
    assert isinstance(a * GHz, fu.ValueArray)
    assert isinstance(a * va, fu.ValueArray)
    assert isinstance(va * va, fu.ValueArray)

    # values
    assert (a * x1)[2] == 2 * m
    assert (a * x2)[2] == 10j * V
    assert (a * GHz)[2] == 2 * GHz
    assert (a * (GHz * GHz))[2] == 2 * GHz * GHz
    assert ((GHz * GHz) * a)[2] == 2 * GHz * GHz
    assert (a * va)[2] == 4 * GHz
    assert (va * va)[2] == 4 * GHz * GHz

    # ValueArray times ?
    assert isinstance(va * x1, fu.ValueArray)
    assert isinstance(va * x2, fu.ValueArray)
    assert isinstance(va * GHz, fu.ValueArray)
    assert isinstance(va * a, fu.ValueArray)


def test_comparison() -> None:
    s = fu.Unit('s')
    ms = fu.Unit('ms')
    kg = fu.Unit('kg')
    assert 1 * s > 10 * ms, '1*s > 10*ms'
    assert 1 * s >= 10 * ms, '1*s >= 10*ms'
    assert 1 * s < 10000 * ms, '1*s > 10000*ms'
    assert 1 * s <= 10000 * ms, '1*s >= 10000*ms'
    assert 10 * ms < 1 * s, '10*ms < 1*s'
    assert 10 * ms <= 1 * s, '10*ms <= 1*s'
    assert 10000 * ms > 1 * s, '10000*ms < 1*s'
    assert 10000 * ms >= 1 * s, '10000*ms <= 1*s'
    with pytest.raises(TypeError):
        _ = 1 * s > 1 * kg

    assert not (1 * s == 1 * kg)
    assert 0 * s == 0 * ms
    assert 0 * s == 0
    assert not 0 * s == 0 * kg
    assert 0 * s != 0 * kg
    assert 4 * s > 0 * s
    assert 4 * s > 0
    with pytest.raises(TypeError):
        _ = 4 * s > 1


def test_complex():
    V = fu.Unit('V')

    assert 1j * V != 1.0 * V
    assert 1j * V == 1.0j * V
    assert 1.0 * V == (1 + 0j) * V
    with pytest.raises(TypeError):
        _ = 1.0j * V < 2j * V


def test_dimensionless():
    ns = fu.Unit('ns')
    GHz = fu.Unit('GHz')

    assert float((5 * ns) * (5 * GHz)) == 25.0
    assert hasattr((5 * ns) * (5 * GHz), 'inUnitsOf')
    assert ((5 * ns) * (5 * GHz)).isDimensionless()
    assert (5 * ns) * (5 * GHz) < 50
    assert isinstance(fu.WithUnit(1, ''), fu.WithUnit)
    assert isinstance(5.0 * fu.WithUnit(1, ''), fu.Value)

    assert (5 * ns * 5j * GHz) == 25j
    assert (5 * ns * 5j * GHz).isDimensionless()


def test_angle():
    rad = fu.Unit('rad')
    assert rad.is_angle
    assert rad.isAngle()
    x = fu.Unit('rad*m/s')
    assert not x.is_angle
    assert not (3.14 * rad).isDimensionless()
    assert not (3.14 * rad ** 2).isDimensionless()
    with pytest.raises(UnitMismatchError):
        _ = float(2.0 * rad)


def test_inf_nan() -> None:
    ms = fu.Unit('ms')
    GHz = fu.Unit('GHz')
    MHz = fu.Unit('MHz')

    assert float('inf') * GHz == float('inf') * MHz
    assert float('inf') * GHz != float('inf') * ms
    assert float('inf') * GHz != -float('inf') * GHz
    assert float('nan') * GHz != float('nan') * GHz
    assert float('nan') * GHz != float('nan') * ms


def test_in_units_of():
    s = fu.Unit('s')
    ms = fu.Unit('ms')
    assert (1 * s).inUnitsOf(ms) == 1000 * ms
    assert (1 * s).inUnitsOf('ms') == 1000 * ms


def test_base_unit_powers() -> None:
    x = Value(1, 'ns^2')
    assert x.inBaseUnits() == Value(1e-18, 's^2')


def test_unit_powers() -> None:
    assert fu.Unit('ns') ** 2 == fu.Unit('ns^2')


def test_array_priority() -> None:
    """numpy issue 6133

    DimensionlessX needs to support all arithmetic operations when the
    other side is an array.  Numpy's __array_priority__ machinery doesn't
    handle NotImplemented results correctly, so the higher priority element
    *must* be able to handle all operations.

    In numpy 1.9 this becomes more critical because numpy scalars like
    np.float64 get converted to arrays before being passed to binary
    arithmetic operations.
    """
    x = np.float64(1)
    y = fu.Value(2)
    assert x < y
    z = np.arange(5)
    assert ((x < z) == [False, False, True, True, True]).all()


def test_none() -> None:
    with pytest.raises(Exception):
        fu.Unit(None)
    with pytest.raises(TypeError):
        _ = None * fu.Unit('MHz')


def test_non_si():
    fu.addNonSI('count', True)
    x = 5 * fu.Unit('kcount')
    assert x['count'] == 5000.0
    assert x.inBaseUnits() == 5000.0 * fu.Unit('count')
    assert (x ** 2).unit == fu.Unit('kcount^2')


def test_unit_auto_creation():
    ts = fu.Unit('pants/s')
    assert (1 * ts)['pants/h'] == 3600.0
    assert str(ts) == 'pants/s'


def test_unit_manual_creation():
    fu.addNonSI('tshirt')
    ts = fu.Unit('tshirt/s')
    assert (1 * ts)['tshirt/h'] == 3600.0
    assert str(ts) == 'tshirt/s'


def test_iter():
    from pyfu.like_pylabrad_units import ns, kg

    data = np.arange(5) * ns
    for x in data:
        assert isinstance(x, fu.Value)
    with pytest.raises(TypeError):
        iter(5 * kg)
    with pytest.raises(TypeError):
        for _ in 5 * kg:
            pass
    assert not np.iterable(5 * kg)


def test_name():
    assert fu.ns.name == 'ns'


def test_equality_against_formulas():
    from pyfu.like_pylabrad_units import m, s, J

    assert m == 'm'
    assert m != 'km'
    assert m != 's'
    assert 'm' == m

    assert s == 's'
    assert s != 'm'

    assert J / s == 'W'
    assert J / m ** 2 * s * s == 'kg'

    # This behavior is specific to the compatibility layer.
    from pyfu.units import kilogram as not_compatible_kilogram

    assert not_compatible_kilogram != 'kg'


def test_sqrt():
    from pyfu.like_pylabrad_units import kg, kiloliter, m

    assert (kg ** 2).sqrt() == kg
    assert kiloliter.sqrt() == m ** 1.5


def test_is_compatible():
    from pyfu.like_pylabrad_units import ns, kg, s

    x = 5 * ns
    assert x.isCompatible('s')
    assert not x.isCompatible(kg)
    assert ns.isCompatible(s)
    assert ns.isCompatible(ns)
    assert ns.isCompatible(ns * 2.0)
    assert not ns.isCompatible(kg)
    assert not ns.isCompatible(kg * 2.0)
    assert not x.isCompatible(4)
    assert (x / ns).isCompatible(4)
    with pytest.raises(Exception):
        x.isCompatible(dict())


def test_scaled_get_item():
    from pyfu.like_pylabrad_units import ns, s

    v = s * 1.0
    assert v[ns] == 10 ** 9
    assert v[ns * 2] == 10 ** 9 / 2
    assert (v * 3)[(ns * 3)] == 10 ** 9
    assert (5 * s / ns)[''] == 5 * 10 ** 9


def test_flatten_value_units():
    from pyfu.like_pylabrad_units import ns, m

    assert Value(ns * 5, 'meter') == ns * 5 * m


def test_flatten_shared_units_into_parent():
    from pyfu.like_pylabrad_units import ns, m

    with pytest.raises(UnitMismatchError):
        ValueArray([ns, m])

    v = ValueArray([-5 * ns, 203 * ns, 0.2 * ns])
    assert not v.isDimensionless()
    assert v[0] == -5 * ns
    assert v[1] == 203 * ns
    assert v[2] == 0.2 * ns


def test_auto_wrap_value_in_array():
    from pyfu.like_pylabrad_units import ns

    assert np.min([3 * ns, 2 * ns, 5 * ns]) == 2 * ns


def test_put_in_array():
    from pyfu.like_pylabrad_units import ns

    a = np.array(ns * 0)
    assert a[()] == ns * 0

    a = np.array([ns * 0])
    assert len(a) == 1
    assert a[0] == ns * 0


def test_unwrap_value_array():
    from pyfu.like_pylabrad_units import ns

    a = np.array([1, 2, 3, 4] * ns)
    assert len(a) == 4
    assert a[0] == ns
    assert a[1] == ns * 2
    assert a[2] == ns * 3
    assert a[3] == ns * 4


def test_real_imag() -> None:
    value = Value(1, 'GHz')
    assert isinstance(value.real, Value)
    assert isinstance(value.imag, Value)
    assert value.real == Value(1, 'GHz')
    assert value.imag == Value(0, 'GHz')

    complex_value = Value(1 + 2j, 'GHz')
    assert isinstance(complex_value.real, Value)
    assert isinstance(complex_value.imag, Value)
    assert complex_value.real == Value(1, 'GHz')
    assert complex_value.imag == Value(2, 'GHz')

    complex_value_array = ValueArray([1, 1j, 3 + 2j], 'GHz')
    assert isinstance(complex_value_array.real, ValueArray)
    assert isinstance(complex_value_array.imag, ValueArray)
    assert all(complex_value_array.real == ValueArray([1, 0, 3], 'GHz'))
    assert all(complex_value_array.imag == ValueArray([0, 1, 2], 'GHz'))


def test_units_rounding() -> None:
    GHz = fu.Unit('GHz')
    MHz = fu.Unit('MHz')

    value = Value(1.1, 'GHz')
    assert value.round('GHz') == 1 * GHz
    assert value.round('GHz').unit == GHz
    assert value.round(GHz) == 1 * GHz
    assert value.round(GHz).unit == GHz
    assert value.round('MHz') == 1100 * MHz
    assert value.round('MHz').unit == MHz
    assert value.round(MHz) == 1100 * MHz
    assert value.round(MHz).unit == MHz

    value_array = ValueArray([1, 2.2, 3.5], 'GHz')
    assert all(value_array.round('GHz') == ValueArray([1 * GHz, 2 * GHz, 4 * GHz]))
    assert value_array.round('GHz').unit == GHz
    assert all(value_array.round('MHz') == ValueArray([1000 * MHz, 2200 * MHz, 3500 * MHz]))
    assert value_array.round('MHz').unit == MHz
