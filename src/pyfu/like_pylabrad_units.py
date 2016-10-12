"""
A compatibility layer that tries to expose the same API as pylabrad's unit
library.
"""

from __future__ import absolute_import
from . import _all_cythonized, unit as _unit

addNonSI = _unit.add_non_standard_unit
Complex = _all_cythonized.Complex
UnitMismatchError = _all_cythonized.UnitMismatchError
Value = _all_cythonized.Value
ValueArray = _all_cythonized.ValueArray
WithUnit = _all_cythonized.WithUnit


class DimensionlessArray(ValueArray):
    pass


class Unit(Value):
    """
    Just a Value (WithValue), but with a constructor that accepts formulas.
    """
    def __init__(self, obj):
        unit = _unit.default_unit_database.parse_unit_formula(obj)
        super(Value, self).__init__(unit)


# Expose defined units (e.g. 'meter', 'km', 'day') as module variables.
for k, v in _unit.default_unit_database.known_units.items():
    globals()[k] = v
