from __future__ import absolute_import
from . import (_all_cythonized,
               base_unit_data,
               derived_unit_data,
               prefix_data,
               unit_database)
import numpy as np

WithUnit = _all_cythonized.WithUnit
Value = _all_cythonized.Value


def _make_unit_database_from_unit_data():
    db = unit_database.UnitDatabase()
    db.add_unit('', WithUnit(1))
    for base in base_unit_data.ALL_BASE_UNITS:
        db.add_base_unit_data(base, prefix_data.SI_PREFIXES)
    for data in derived_unit_data.ALL_DERIVED_UNITS:
        db.add_derived_unit_data(data, prefix_data.SI_PREFIXES)
    return db

default_unit_database = _make_unit_database_from_unit_data()


def _try_interpret_as_with_unit(obj):
    """
    This method is given to WithUnit so that it can do a convenient conversion
    from a user-given object (such as a string formula or a float or a WithUnit)
    into just a WithUnit.
    """
    if isinstance(obj, WithUnit):
        return obj
    if isinstance(obj, str):
        return default_unit_database.parse_unit_formula(obj)
    if isinstance(obj, int) or isinstance(obj, float) or isinstance(obj, list) \
            or isinstance(obj, complex) or isinstance(obj, np.ndarray):
        return WithUnit(obj)
    return None

_all_cythonized.init_base_unit_functions(
    _try_interpret_as_with_unit,
    default_unit_database.is_value_consistent_with_database)


class Unit(Value):
    """
    Just a Value (WithValue), but with a constructor that accepts formulas.
    """
    def __init__(self, obj):
        unit = default_unit_database.parse_unit_formula(obj)
        super(Value, self).__init__(unit)


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
