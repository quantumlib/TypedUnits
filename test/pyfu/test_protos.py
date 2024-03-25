import itertools
import pytest
import numpy as np
from pyfu import units, Value, ValueArray, Complex, tunits_pb2

_ONE_UNIT = [
    units.dBm,  # type: ignore
    units.rad,  # type: ignore
    units.GHz,  # type: ignore
    units.MHz,  # type: ignore
    units.V,  # type: ignore
    units.mV,  # type: ignore
    units.ns,  # type: ignore
    units.us,  # type: ignore
    units.dB,  # type: ignore
]

_TWO_UNITS = [
    a * b
    for a, b in itertools.product(
        _ONE_UNIT,
        repeat=2,
    )
] + [
    a / b
    for a, b in itertools.product(
        _ONE_UNIT,
        repeat=2,
    )
]


@pytest.mark.parametrize('unit', _ONE_UNIT + _TWO_UNITS)
def test_value_conversion_trip(unit):
    rs = np.random.RandomState(0)
    for value in rs.random(10):
        v = value * unit
        assert Value.from_proto(v.to_proto()) == v


@pytest.mark.parametrize('unit', _ONE_UNIT + _TWO_UNITS)
def test_complex_conversion_trip(unit):
    rs = np.random.RandomState(0)
    for value in rs.random(10):
        v = 1j * value * unit
        assert Complex.from_proto(v.to_proto()) == v


@pytest.mark.parametrize('unit', _ONE_UNIT + _TWO_UNITS)
def test_valuearray_conversion_trip(unit):
    rs = np.random.RandomState(0)
    for value in rs.random((4, 2, 4, 3)):
        v = value * unit
        got = ValueArray.from_proto(v.to_proto())
        assert got.unit == unit
        np.testing.assert_allclose(got.value, value)


@pytest.mark.parametrize('unit', _ONE_UNIT + _TWO_UNITS)
def test_complex_valuearray_conversion_trip(unit):
    rs = np.random.RandomState(0)
    for real, imag in zip(rs.random((4, 2, 4, 3)), rs.random((4, 2, 4, 3))):
        v = (real + 1j * imag) * unit
        got = ValueArray.from_proto(v.to_proto())
        assert got.unit == unit
        np.testing.assert_allclose(got.value, real + 1j * imag)


@pytest.mark.parametrize(
    'unit',
    [
        units.A,  # type: ignore
        units.V * units.Ohm,  # type: ignore
    ],
)
def test_unsupported_unit_conversion_raises_valueerror(unit):

    with pytest.raises(ValueError):
        _ = unit.to_proto()


def test_valuearray_shape_mismatch():
    msg = tunits_pb2.ValueArray()
    msg.reals.values.extend([1, 2])
    with pytest.raises(ValueError):
        ValueArray.from_proto(msg)


def test_emptyarray_is_invalid():
    msg = tunits_pb2.ValueArray()
    with pytest.raises(ValueError):
        _ = ValueArray.from_proto(msg)


def test_empty_unit_is_valid():
    msg = tunits_pb2.Value(real_value=1)
    _ = Value.from_proto(msg)


def test_empty_value_is_invalid():
    msg = tunits_pb2.Value(units=[tunits_pb2.Unit()])
    with pytest.raises(ValueError):
        _ = Value.from_proto(msg)

    with pytest.raises(ValueError):
        _ = Complex.from_proto(msg)


def test_unit_exponent_with_zero_denominator_raises():
    with pytest.raises(ValueError):
        _ = Value.from_proto(
            tunits_pb2.Value(units=[tunits_pb2.Unit(exponent=tunits_pb2.Fraction(denominator=0))])
        )


def test_scale_values_are_correct():
    from pyfu._all_cythonized import _SCALE_PREFIXES

    assert len(_SCALE_PREFIXES) == len(
        tunits_pb2.Scale.items()
    ), f'differing number of scales in proto and _SCALE_PREFIXES. If you are adding new scales please update the _SCALE_PREFIXES map'

    scale_to_prefix = {
        'YOTTA': 'Y',
        'ZETTA': 'Z',
        'EXA': 'E',
        'PETA': 'P',
        'TERA': 'T',
        'GIGA': 'G',
        'MEGA': 'M',
        'KILO': 'k',
        'HECTO': 'h',
        'DECAD': 'da',
        'UNITY': '',
        'DECI': 'd',
        'CENTI': 'c',
        'MILLI': 'm',
        'MICRO': 'u',
        'NANO': 'n',
        'PICO': 'p',
        'FEMTO': 'f',
        'ATTO': 'a',
        'ZEPTO': 'z',
        'YOCTO': 'y',
    }
    for scale, prefix in scale_to_prefix.items():
        assert tunits_pb2.Scale.Value(scale) == _SCALE_PREFIXES[prefix]