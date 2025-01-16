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

"""Provides methods from conversion to/from proto representation."""

from typing import Optional, Tuple
from io import BytesIO
import functools

import numpy as np


_PROTO_TO_UNIT_STRING = {
    'DECIBEL_MILLIWATTS': 'dBm',
    'RADIANS': 'rad',
    'HERTZ': 'Hz',
    'VOLT': 'V',
    'SECOND': 's',
    'DECIBEL': 'dB',
}

_UNIT_STRING_TO_PROTO = {v: k for k, v in _PROTO_TO_UNIT_STRING.items()}
SCALE_PREFIXES = {
    'y': -24,
    'z': -21,
    'a': -18,
    'f': -15,
    'p': -12,
    'n': -9,
    'u': -6,
    'm': -3,
    'c': -2,
    'd': -1,
    '': 0,
    'da': 1,
    'h': 2,
    'k': 3,
    'M': 6,
    'G': 9,
    'T': 12,
    'P': 15,
    'E': 18,
    'Z': 21,
    'Y': 24,
}

_ENUM_TO_SCALE_SYMBOL = {v: k for k, v in SCALE_PREFIXES.items()}

_SERIALIZATION_ERROR_MESSAGE = (
    "can't map unit={} to a proto enum value. If it's "
    "equivalent to a supported unit, please use inUnitsOf to cast to that unit"
)


@functools.cache
def _construct_unit(unit_enum: int, scale_enum: Optional[int] = None) -> 'Value':
    from tunits.proto import tunits_pb2

    unit_name = _PROTO_TO_UNIT_STRING.get(tunits_pb2.UnitEnum.Name(unit_enum), None)
    scale = '' if scale_enum is None else _ENUM_TO_SCALE_SYMBOL[scale_enum]
    return _try_interpret_as_with_unit(scale + unit_name)


def _proto_to_unit(unit: 'tunits_pb2.Unit') -> 'Value':
    """
    Returns the equivalent string representation of a given tunits_pb2.Unit.
    """
    u = _construct_unit(unit.unit, unit.scale if unit.HasField('scale') else None)
    if unit.HasField('exponent'):
        if unit.exponent.denominator == 0:
            raise ValueError(f'invalid unit exponent {unit.exponent}')
        u = u ** (unit.exponent.numerator / unit.exponent.denominator)
    return u


def _proto_to_units(units_protos: Sequence['tunits_pb2.Unit']) -> 'Value':
    ret = None
    for u in units_protos:
        if ret is None:
            ret = _proto_to_unit(u)
        else:
            ret *= _proto_to_unit(u)
    return ret


def _unit_name_to_proto(unit_name: str) -> Optional['tunits_pb2.Unit']:
    from tunits.proto import tunits_pb2

    if unit_name in _UNIT_STRING_TO_PROTO:
        return tunits_pb2.Unit(
            unit=_UNIT_STRING_TO_PROTO[unit_name],
        )
    for scale_prefix, scale_value in SCALE_PREFIXES.items():
        suffix = unit_name[len(scale_prefix) :]
        if unit_name.startswith(scale_prefix) and suffix in _UNIT_STRING_TO_PROTO:
            return tunits_pb2.Unit(
                unit=_UNIT_STRING_TO_PROTO[suffix],
                scale=scale_value,
            )
    return None


def _unit_to_proto(unit: Tuple[str, int, int]) -> 'tunits_pb2.Unit':
    ret = _unit_name_to_proto(unit[0])
    if ret is None:
        raise ValueError(_SERIALIZATION_ERROR_MESSAGE.format(unit))
    if unit[1:] != (1, 1):
        ret.exponent.numerator = unit[1]
        ret.exponent.denominator = unit[2]
    return ret


def _units_to_proto(units: 'raw_UnitArray') -> Sequence['tunits_pb2.Unit']:
    return [_unit_to_proto(u) for u in units]


def _ndarray_to_proto(
    arr: np.ndarray, msg: Optional['tunits_pb2.ValueArray'] = None
) -> 'tunits_pb2.ValueArray':
    from tunits.proto import tunits_pb2

    if msg is None:
        msg = tunits_pb2.ValueArray()
    msg.shape.extend(arr.shape)
    if arr.dtype in (np.complex64, np.complex128):
        msg.complexes.values.extend(
            tunits_pb2.Complex(real=v.real, imaginary=v.imag) for v in arr.flatten()
        )
    else:
        msg.reals.values.extend(arr.flatten())
    return msg


def _ndarray_from_proto(msg: 'tunits_pb2.ValueArray') -> np.ndarray:
    if msg.WhichOneof('values') == 'reals':
        arr = np.array(msg.reals.values, dtype=np.float64)
    else:
        arr = np.array(
            [v.real + 1j * v.imaginary for v in msg.complexes.values], dtype=np.complex128
        )
    if len(arr) == 0:
        raise ValueError("empty arrays are not supported in deserialization")
    arr = arr.reshape(msg.shape)
    return arr
