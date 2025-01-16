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

from typing import cast
import itertools
import pytest
import numpy as np
from tunits import units, Value, ValueArray, tunits_pb2
from tunits.core import SCALE_PREFIXES

_ONE_UNIT: list[Value] = [
    units.dBm,
    units.rad,
    units.GHz,
    units.MHz,
    units.V,
    units.mV,
    units.ns,
    units.us,
    units.dB,
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
def test_value_conversion_trip(unit: Value) -> None:
    rs = np.random.RandomState(0)
    for value in rs.random(10):
        v = value * unit
        assert Value.from_proto(v.to_proto()) == v


@pytest.mark.parametrize('unit', _ONE_UNIT + _TWO_UNITS)
def test_complex_conversion_trip(unit: Value) -> None:
    rs = np.random.RandomState(0)
    for value in rs.random(10):
        v = 1j * value * unit
        assert Value.from_proto(v.to_proto()) == v


@pytest.mark.parametrize('unit', _ONE_UNIT + _TWO_UNITS)
def test_valuearray_conversion_trip(unit: Value) -> None:
    rs = np.random.RandomState(0)
    for value in rs.random((4, 2, 4, 3)):
        v = value * unit
        got = ValueArray.from_proto(v.to_proto())
        assert got.unit == unit
        np.testing.assert_allclose(got.value, value)


@pytest.mark.parametrize('unit', _ONE_UNIT + _TWO_UNITS)
def test_complex_valuearray_conversion_trip(unit: Value) -> None:
    rs = np.random.RandomState(0)
    for real, imag in zip(rs.random((4, 2, 4, 3)), rs.random((4, 2, 4, 3))):
        real_ = cast(np.typing.NDArray[np.float64], real)
        value = real_ + 1j * imag
        v = unit * value
        got = ValueArray.from_proto(v.to_proto())
        assert got.unit == unit
        np.testing.assert_allclose(got.value, real + 1j * imag)


@pytest.mark.parametrize(
    'unit',
    [
        units.A,
        units.V * units.Ohm,
    ],
)
def test_unsupported_unit_conversion_raises_valueerror(unit: Value) -> None:
    with pytest.raises(ValueError):
        _ = unit.to_proto()


def test_valuearray_shape_mismatch() -> None:
    msg = tunits_pb2.ValueArray()
    msg.reals.values.extend([1, 2])
    with pytest.raises(ValueError):
        ValueArray.from_proto(msg)


def test_emptyarray_is_invalid() -> None:
    msg = tunits_pb2.ValueArray()
    with pytest.raises(ValueError):
        _ = ValueArray.from_proto(msg)


def test_empty_unit_is_valid() -> None:
    msg = tunits_pb2.Value(real_value=1)
    _ = Value.from_proto(msg)


def test_empty_value_is_invalid() -> None:
    msg = tunits_pb2.Value(units=[tunits_pb2.Unit()])
    with pytest.raises(ValueError):
        _ = Value.from_proto(msg)

    with pytest.raises(ValueError):
        _ = Value.from_proto(msg)


def test_unit_exponent_with_zero_denominator_raises() -> None:
    with pytest.raises(ValueError):
        _ = Value.from_proto(
            tunits_pb2.Value(units=[tunits_pb2.Unit(exponent=tunits_pb2.Fraction(denominator=0))])
        )


def test_scale_values_are_correct() -> None:
    assert len(SCALE_PREFIXES) == len(
        tunits_pb2.Scale.items()
    ), f'differing number of scales in proto and SCALE_PREFIXES. If you are adding new scales please update the SCALE_PREFIXES map'

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
        assert tunits_pb2.Scale.Value(scale) == SCALE_PREFIXES[prefix]
