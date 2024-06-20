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

from typing import TypeVar, Any

import numpy as np

import tunits_core
import tunits.api.unit as _unit


addNonSI = _unit.add_non_standard_unit
UnitMismatchError = tunits_core.UnitMismatchError
Value = tunits_core.Value
ValueArray = tunits_core.ValueArray
WithUnit = tunits_core.WithUnit


class DimensionlessArray(tunits_core.ValueArray):
    pass


_WITH_UNIT = TypeVar('_WITH_UNIT', bound=WithUnit)
ValueType = TypeVar('ValueType', bound=Value)


class Unit(tunits_core.Value):
    """
    Just a Value (WithValue), but with a constructor that accepts formulas.
    """

    def __init__(self, obj: Any) -> None:
        if not isinstance(obj, WithUnit):
            obj = _unit.default_unit_database.parse_unit_formula(obj)
        super().__init__(obj)

    @property
    def name(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            other = _unit.default_unit_database.parse_unit_formula(other)
        if isinstance(other, WithUnit):
            return Value(self) == other
        return False

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def __pow__(self, exponent: Any, modulus: Any = None) -> 'Unit':
        return Unit(Value(self).__pow__(exponent, modulus))

    def __mul__(
        self,
        other: (
            int
            | float
            | np.number[Any]
            | complex
            | list[Any]
            | tuple[Any]
            | np._typing.NDArray[Any]
            | _WITH_UNIT
        ),
    ) -> Any:
        if not isinstance(
            other,
            (Value, ValueArray, Unit, int, float, complex, np.number, np.ndarray, tuple, list),
        ):
            return NotImplemented
        product = super().__mul__(other)
        if isinstance(other, Unit):
            return Unit(product)
        return product

    def __truediv__(
        self,
        other: (
            int
            | float
            | np.number[Any]
            | complex
            | list[Any]
            | tuple[Any]
            | np._typing.NDArray[Any]
            | _WITH_UNIT
        ),
    ) -> Any:
        if not isinstance(
            other,
            (Value, ValueArray, Unit, int, float, complex, np.number, np.ndarray, tuple, list),
        ):
            return NotImplemented
        quotient = Value(self) / other
        if isinstance(other, Unit):
            return Unit(quotient)
        return quotient


# Expose defined units (e.g. 'meter', 'km', 'day') as module variables.
# Note: unless you like overwriting Boltzmann's constant, keep the underscores.
for _k, _v in _unit.default_unit_database.known_units.items():
    globals()[_k] = Unit(_v)
