import __all_cythonized
from __all_cythonized import Value, WithUnit
from base_unit_data import ALL_BASE_UNITS
from derived_unit_data import ALL_DERIVED_UNITS
from prefix_data import SI_PREFIXES
import numpy as np
from unit_database import UnitDatabase


def _make_unit_database_from_unit_data():
    db = UnitDatabase()
    db.add_unit('', WithUnit(1))
    for base in ALL_BASE_UNITS:
        db.add_base_unit_data(base, SI_PREFIXES)
    for data in ALL_DERIVED_UNITS:
        db.add_derived_unit_data(data, SI_PREFIXES)
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

__all_cythonized.init_base_unit_functions(_try_interpret_as_with_unit)


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
        for data in SI_PREFIXES:
            for prefix in [data.name, data.symbol]:
                default_unit_database.add_scaled_unit(
                    prefix + name,
                    name,
                    exp10=data.exp10)
