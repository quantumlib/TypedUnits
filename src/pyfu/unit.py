from __future__ import absolute_import
from . import (_all_cythonized,
               base_unit_data,
               derived_unit_data,
               physical_constant_data,
               prefix_data,
               unit_database)
import numpy as np

Complex = _all_cythonized.Complex
Value = _all_cythonized.Value
ValueArray = _all_cythonized.ValueArray
WithUnit = _all_cythonized.WithUnit


def _make_unit_database_from_unit_data():
    db = unit_database.UnitDatabase()
    db.add_unit('', WithUnit(1))

    for base in base_unit_data.ALL_BASE_UNITS:
        db.add_base_unit_data(base, prefix_data.SI_PREFIXES)

    for data in derived_unit_data.ALL_DERIVED_UNITS:
        db.add_derived_unit_data(data, prefix_data.SI_PREFIXES)

    for data in physical_constant_data.ALL_PHYSICAL_CONSTANT_DATA:
        db.add_physical_constant_data(data)

    return db

default_unit_database = _make_unit_database_from_unit_data()


def _try_interpret_as_with_unit(obj, avoid_ambiguity_with_indexing=False):
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
    if isinstance(obj, _all_cythonized.WithUnit):
        if (avoid_ambiguity_with_indexing and
                obj.isDimensionless() and
                (not isinstance(obj, Value) or obj.value != 1)):  # Not a unit?
            raise TypeError("Ambiguous unit key: " + repr(obj))
        return obj

    if isinstance(obj, int) or isinstance(obj, float):
        if avoid_ambiguity_with_indexing:
            raise TypeError("Ambiguous unit key: " + repr(obj))
        return Value(obj)

    if isinstance(obj, complex):
        if avoid_ambiguity_with_indexing:
            raise TypeError("Ambiguous unit key: " + repr(obj))
        return Complex(obj)

    if isinstance(obj, list) or isinstance(obj, np.ndarray):
        return ValueArray(obj)

    if isinstance(obj, str):
        return default_unit_database.parse_unit_formula(obj)

    return None

_all_cythonized.init_base_unit_functions(
    _try_interpret_as_with_unit,
    default_unit_database.is_value_consistent_with_database)


def add_non_standard_unit(name, use_prefixes=False):
    """
    :param str name:
    :param bool use_prefixes:
    """
    default_unit_database.add_root_unit(name)
    if use_prefixes:
        for data in prefix_data.SI_PREFIXES:
            for prefix in [data.name, data.symbol]:
                default_unit_database.add_scaled_unit(
                    prefix + name,
                    name,
                    exp10=data.exp10)
