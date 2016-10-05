"""
A compatibility layer that tries to expose the same API as pylabrad's unit
library.
"""

from __future__ import absolute_import
from . import _all_cythonized, unit as _unit

# Expose the type/method API.
Unit = _unit.Unit
addNonSI = _unit.add_non_standard_unit
Complex = _all_cythonized.Complex
DimensionlessUnit = _all_cythonized.DimensionlessUnit
UnitArray = _all_cythonized.UnitArray
UnitMismatchError = _all_cythonized.UnitMismatchError
Value = _all_cythonized.Value
ValueArray = _all_cythonized.ValueArray
WithUnit = _all_cythonized.WithUnit

# Expose defined units (e.g. 'meter', 'km', 'day') as module variables.
for k, v in _unit.default_unit_database.known_units.items():
    globals()[k] = v
