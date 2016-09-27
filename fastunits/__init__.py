#!/usr/bin/env python
import fastunits.unitarray as __unitarray
import unit as __unit


# Expose the type/method API.
Unit = __unit.Unit
addNonSI = __unit.addNonSI
Complex = __unitarray.Complex
DimensionlessUnit = __unitarray.DimensionlessUnit
UnitArray = __unitarray.UnitArray
UnitMismatchError = __unitarray.UnitMismatchError
Value = __unitarray.Value
ValueArray = __unitarray.ValueArray
WithUnit = __unitarray.WithUnit

# Expose defined units (e.g. 'meter', 'km', 'day') as module variables.
for k, v in __unit.default_unit_database.known_units.items():
    globals()[k] = v
