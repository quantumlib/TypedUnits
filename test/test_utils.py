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

from typing import Any, overload

from numpy._typing import NDArray

import tunits_core


def frac(numer: int = 1, denom: int = 1) -> dict[str, int]:
    """Creates an object that can be passed as a frac struct to cython.

    In the cython core of tunits, `frac` is defined as a C struct. C structs
    appear as python dictionaries to python.
    """
    return {'numer': numer, 'denom': denom}


def unit(key: str) -> tunits_core.UnitArray:
    """Returns a UnitArray containing the given unit."""
    return tunits_core.raw_UnitArray([(key, 1, 1)])


def conv(factor: float = 1.0, numer: int = 1, denom: int = 1, exp10: int = 0) -> dict[str, Any]:
    """Creates an object that can be passed as a conversion struct to cython.

    In the cython core of tunits, `conversion` is defined as a C struct. C structs
    appear as python dictionaries to python.
    """
    return {'factor': factor, 'ratio': frac(numer, denom), 'exp10': exp10}


@overload
def val(
    value: int | float | complex,
    conv: dict[str, Any] = conv(),
    units: tunits_core.UnitArray = tunits_core.raw_UnitArray([]),
    display_units: tunits_core.UnitArray | None = None,
) -> tunits_core.Value: ...


@overload
def val(
    value: list[Any] | tuple[Any] | NDArray[Any],
    conv: dict[str, Any] = conv(),
    units: tunits_core.UnitArray = tunits_core.raw_UnitArray([]),
    display_units: tunits_core.UnitArray | None = None,
) -> tunits_core.ValueArray: ...


def val(
    value: int | float | complex | list[Any] | tuple[Any] | NDArray[Any],
    conv: dict[str, Any] = conv(),
    units: tunits_core.UnitArray = tunits_core.raw_UnitArray([]),
    display_units: tunits_core.UnitArray | None = None,
) -> tunits_core.Value | tunits_core.ValueArray:
    """A factory method for creating values with unit."""
    return tunits_core.raw_WithUnit(
        value, conv, units, units if display_units is None else display_units
    )
