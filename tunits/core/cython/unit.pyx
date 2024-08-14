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

import numpy as np


def _make_unit_database_from_unit_data() -> UnitDatabase:
    db = UnitDatabase()
    db.add_unit('', Value(1))

    for base in ALL_BASE_UNITS:
        db.add_base_unit_data(base, SI_PREFIXES)

    for derived_data in ALL_DERIVED_UNITS:
        db.add_derived_unit_data(derived_data, SI_PREFIXES)

    for phys_const_data in ALL_PHYSICAL_CONSTANT_DATA:
        db.add_physical_constant_data(phys_const_data)

    return db


default_unit_database: UnitDatabase = _make_unit_database_from_unit_data()


def _try_interpret_as_with_unit(
    obj: Any, avoid_ambiguity_with_indexing: bool = False
) -> WithUnit | None:
    """
    This method is given to WithUnit so that it can do a convenient conversion
    from a user-given object (such as a string formula or a float or a WithUnit)
    into just a WithUnit.

    Used by __getitem__, isCompatibleWith, Value's constructor, etc. Anything
    that takes a unit value or unit string.

    :param boolean avoid_ambiguity_with_indexing: When True, this method throws
    an exception when the given object is dimensionless but doesn't have value
    equal to 1 (like units do). (The reasoning is that it may plausibly have
    been intended to be, or could be confused with, an index for ValueArray.
    :return None|WithUnit:
    :raises TypeError: When avoid_ambiguity_with_indexing is set and the object
    is ambiguously similar to an index.
    """
    if isinstance(obj, WithUnit):
        if (
            avoid_ambiguity_with_indexing
            and obj.is_dimensionless
            and (not isinstance(obj, Value) or obj.value != 1)
        ):  # Not a unit?
            raise TypeError("Ambiguous unit key: " + repr(obj))
        return obj

    if isinstance(obj, (int, float, complex)):
        if avoid_ambiguity_with_indexing:
            raise TypeError("Ambiguous unit key: " + repr(obj))
        return Value(obj)

    if isinstance(obj, list) or isinstance(obj, np.ndarray):
        return ValueArray(obj)

    if isinstance(obj, str):
        return default_unit_database.parse_unit_formula(obj)

    return None


init_base_unit_functions(
    _try_interpret_as_with_unit, default_unit_database.is_value_consistent_with_database
)


def add_non_standard_unit(name: str, use_prefixes: bool = False) -> None:
    """
    :param str name:
    :param bool use_prefixes:
    """
    default_unit_database.add_root_unit(name)
    if use_prefixes:
        for data in SI_PREFIXES:
            for prefix in [data.name, data.symbol]:
                default_unit_database.add_scaled_unit(prefix + name, name, exp10=data.exp10)
