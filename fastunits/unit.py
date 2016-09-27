#!/usr/bin/env python
from __future__ import division
from fastunits.unitarray import WithUnit, Value, init_base_unit_functions
from fastunits.unit_database import UnitDatabase
from base_unit_data import ALL_BASE_UNITS
from derived_unit_data import ALL_DERIVED_UNITS
from prefix_data import SI_PREFIXES
import numpy as np


def make_unit_database_from_unit_data():
    db = UnitDatabase()
    db.add_unit('', WithUnit(1))
    for base in ALL_BASE_UNITS:
        db.add_base_unit_data(base, SI_PREFIXES)
    for data in ALL_DERIVED_UNITS:
        db.add_derived_unit_data(data, SI_PREFIXES)
    return db

_default_unit_database = make_unit_database_from_unit_data()


def _unit_val_from_str(formula):
    return _default_unit_database.get_unit(formula)


def _try_interpret_as_with_unit(obj):
    """Lookup a unit by name.

    This is a helper called when WithUnit objects need to lookup a unit
    string.  We return the underlying _value, because that is what the C
    API knows how to handle."""
    if isinstance(obj, WithUnit):
        return obj
    if isinstance(obj, str):
        return _default_unit_database.parse_unit(obj)
    if isinstance(obj, int) or isinstance(obj, float) or isinstance(obj, list) \
            or isinstance(obj, complex) or isinstance(obj, np.ndarray):
        return WithUnit(obj)
    return None


init_base_unit_functions(_try_interpret_as_with_unit)


class Unit(Value):
    def __init__(self, obj):
        unit = _default_unit_database.parse_unit(obj)
        super(Value, self).__init__(unit)


def addNonSI(name, use_prefixes=False):
    _default_unit_database.add_root_unit(name)
    if use_prefixes:
        for data in SI_PREFIXES:
            for prefix in [data.name, data.symbol]:
                _default_unit_database.add_scaled_unit(
                    prefix + name,
                    1,
                    1,
                    1,
                    data.exponent,
                    name)
