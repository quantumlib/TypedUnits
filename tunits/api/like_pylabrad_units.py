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

"""
A compatibility layer that tries to expose the same API as pylabrad's unit
library.
"""

from typing import TypeVar
from tunits.core import _all_cythonized
import tunits.api.unit as _unit

addNonSI = _unit.add_non_standard_unit
Complex = _all_cythonized.Complex
UnitMismatchError = _all_cythonized.UnitMismatchError
Value = _all_cythonized.Value
ValueArray = _all_cythonized.ValueArray
WithUnit = _all_cythonized.WithUnit


_UNIT_T = TypeVar('_UNIT_T', bound='Unit')


class DimensionlessArray(_all_cythonized.ValueArray):
    pass


class Unit(_all_cythonized.Value):
    """
    Just a Value (WithValue), but with a constructor that accepts formulas.
    """

    def __init__(self, obj):
        if not isinstance(obj, WithUnit):
            obj = _unit.default_unit_database.parse_unit_formula(obj)
        super().__init__(obj)

    @property
    def name(self) -> str:
        return str(self)

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            other = _unit.default_unit_database.parse_unit_formula(other)
        return Value(self) == other

    def __ne__(self, other) -> bool:
        return not (self == other)

    def __mul__(self: _UNIT_T, other) -> _UNIT_T | _all_cythonized.Value:
        product = Value(self) * other
        return Unit(product) if isinstance(other, Unit) else product

    def __truediv__(self: _UNIT_T, other) -> _UNIT_T | _all_cythonized.Value:
        quotient = Value(self) / other
        return Unit(quotient) if isinstance(other, Unit) else quotient

    def __floordiv__(self: _UNIT_T, other) -> _UNIT_T | _all_cythonized.Value:
        quotient = Value(self) / other
        return Unit(quotient) if isinstance(other, Unit) else quotient

    def __pow__(self: _UNIT_T, exponent, modulus=None) -> _UNIT_T:
        return Unit(Value(self).__pow__(exponent, modulus))


# Expose defined units (e.g. 'meter', 'km', 'day') as module variables.
# Note: unless you like overwriting Boltzmann's constant, keep the underscores.
for _k, _v in _unit.default_unit_database.known_units.items():
    globals()[_k] = Unit(_v)
