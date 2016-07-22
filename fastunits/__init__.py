#!/usr/bin/env python
from __future__ import division
import fastunits.unitarray as unitarray
from fastunits.unitarray import WithUnit, Value, Complex, ValueArray, UnitMismatchError
import fastunits.unit_grammar as unit_grammar
from prefix_data import SI_PREFIXES
from derived_unit_data import ALL_DERIVED_UNITS

_unit_cache = {}
class Unit(object):
    """Unit database.

    Values defined in unit_array do not actually store a unit object, the unit names and powers
    are stored within the value object itself.  However, when constructing new values or converting
    between units, we need a database of known units.
    """
    __array_priority__ = 15
    __slots__ = ['_value']

    def __new__(cls, name):
        if isinstance(name, Unit):
            return name
        if name in _unit_cache:
            return _unit_cache[name]
        else:
            return cls.parse_unit_str(name)

    # The following methods are internal constructors used to generate new unit instances
    # to separate that out from the main __new__ method which users will use to construct
    # objects from strings.
    @classmethod
    def _new_from_value(cls, val):
        if not isinstance(val, unitarray.WithUnit):
            raise RuntimeError("Need Value type to create unit")
        if val.value != 1.0:
            raise RuntimeError("Cannot create unit from a value not of unit magnitude")
        obj = object.__new__(cls)
        obj._value = val
        return obj

    @classmethod
    def _unit_from_parse_item(cls, item, neg=0):
        base_name = item.name
        numer = item.num or 1
        denom = item.denom or 1
        sign = -1 if item.neg else 1
        if neg:
            sign = -sign
        if base_name not in _unit_cache:
            base_unit = unitarray.UnitArray(base_name)
            _unit_cache[base_name] = Unit._new_from_value(WithUnit._new_raw(1, 1, 1, 0, base_unit, base_unit))
        element = _unit_cache[base_name]**(1.0*sign*numer/denom)
        return element

    @classmethod
    def _new_derived_unit(cls, name, numer, denom, exp10, base_unit):
        if isinstance(base_unit, str):
            base_unit = Unit(base_unit)
        numer = numer * base_unit._value.numer
        denom = denom * base_unit._value.denom
        exp10 = exp10 + base_unit._value.exp10
        val = WithUnit.raw(1, numer, denom, exp10, base_unit._value.base_units, unitarray.UnitArray(name))
        result = cls._new_from_value(val)
        _unit_cache[name] = result
        return result

    @classmethod
    def _new_base_unit(cls, name):
        if name in _unit_cache:
            raise RuntimeError("Trying to create unit that already exists")
        ua = unitarray.UnitArray(name)
        val = WithUnit.raw(1, 1, 1, 0, ua, ua)
        result = cls._new_from_value(val)
        _unit_cache[name] = result
        return result

    @classmethod
    def parse_unit_str(cls, name):
        parsed = unit_grammar.unit.parseString(name)
        result = Unit('')

        for item in parsed.posexp:
            element = cls._unit_from_parse_item(item, 0)
            result = result * element
        for item in parsed.negexp:
            result = result * cls._unit_from_parse_item(item, -1)
        return result

    # Unit arithmetic is used in two ways: to build compound units
    # or to build new Value instances by multiplying a scalar by
    # a unit object.  Since a "Unit" just has an internal value,
    # representing its units, the later just gets delegated to
    # Value arithmetic.
    def __mul__(self, other):
        if isinstance(other, Unit):
            return Unit._new_from_value(self._value * other._value)
        result = self._value * other
        return result

    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, Unit):
            return Unit._new_from_value(self._value/other._value)
        return self._value/other

    def __rdiv__(self, other):
        if isinstance(other, Unit):
            return Unit._new_from_value(other._value/self._value)
        result = other/self._value
        return result

    def __pow__(self, other):
        return Unit._new_from_value(self._value**other)

    def __copy__(self):
        """Units are immutable, so __copy__ just returns self.
        """
        return self

    def __deepcopy__(self, memo):
        return self

    @property
    def name(self):
        return str(self)

    def __repr__(self):
        return "<Unit '%s'>" % (str(self._value.display_units),)

    def __str__(self):
        return str(self._value.display_units)

    def __eq__(self, other):
        if not isinstance(other, Unit):
            return NotImplemented
        return self._value == other._value

    def __ne__(self, other):
        return not (self == other)

    def conversionFactorTo(self, other):
        if not isinstance(other, Unit):
            raise TypeError("conversionFactorTo called on non-unit")
        if self._value.base_units != other._value.base_units:
            raise TypeError("incompabile units '%s', '%s'" % (self.name, other.name))
        ratio = self._value / other._value
        return ratio.inBaseUnits().value

    def converstionTupleTo(self, other):
        """Deprecated.

        This was needed for support of degree scales with zero offsets like degF and degC.  This library
        doesn't support them, so offset is always 0.
        """
        factor = self.conversionFactorTo(other)
        return factor,0

    def isDimensionless(self):
        return self._value.isDimensionless()

    def isAngle(self):
        return self._value.base_units == _unit_cache['rad'].base_units

# The next two methods are called from the C implementation
# of Value() to implement the parts of the API that interact
# with Unit objects (in particular, the cache of known unit
# instances)-- unit conversion and new object creation.
# It is not allowed to directly modify C PyTypeObjects from python
# so we need a helper method to set these, which is done in
# Value._set_py_func


def _unit_val_from_str(unitstr):
    """Lookup a unit by name.

    This is a helper called when WithUnit objects need to lookup a unit
    string.  We return the underlying _value, because that is what the C
    API knows how to handle."""
    unit = Unit(unitstr)
    return unit._value

def _value_unit(withUnit):
    """This is called by Value to implement the .unit property"""
    v = WithUnit._new_raw(1, withUnit.numer, withUnit.denom, withUnit.exp10, withUnit.base_units, withUnit.display_units)
    return Unit._new_from_value(v)

unitarray.init_base_unit_functions(_value_unit, _unit_val_from_str)


_unit_cache[''] = Unit._new_from_value(WithUnit.raw(1,1,1,0, unitarray.DimensionlessUnit, unitarray.DimensionlessUnit))

SI_BASE_UNITS = ['m', 'kg', 's', 'A', 'K', 'mol', 'cd', 'rad', 'sr']
SI_BASE_UNIT_FULL = ['meter', 'kilogram', 'second', 'ampere', 'kelvin', 'mole', 'candela', 'radian', 'steradian']

for name, long_name in zip(SI_BASE_UNITS, SI_BASE_UNIT_FULL):
    Unit._new_base_unit(name)
    Unit._new_derived_unit(long_name, 1, 1, 0, name)

    if (name == 'kg'):
        Unit._new_derived_unit('g', 1, 1, -3, name)
        Unit._new_derived_unit('gram', 1, 1, -3, name)
        name = 'g'
        long_name = 'gram'

    for pre in SI_PREFIXES:
        if (name == 'g' and pre.symbol == 'k'):
            continue
        Unit._new_derived_unit(pre.symbol + name, 1, 1, pre.exponent, name)
        Unit._new_derived_unit(pre.name + long_name, 1, 1, pre.exponent, name)

for der in ALL_DERIVED_UNITS:
    Unit._new_derived_unit(der.symbol,
                           der.numerator,
                           der.denominator,
                           der.exponent,
                           der.base_unit_expression)

    Unit._new_derived_unit(der.name,
                           der.numerator,
                           der.denominator,
                           der.exponent,
                           der.base_unit_expression)

    if der.use_prefixes:
        for pre in SI_PREFIXES:
            Unit._new_derived_unit(pre.symbol + der.symbol,
                                   der.numerator,
                                   der.denominator,
                                   pre.exponent + der.exponent,
                                   der.base_unit_expression)

            Unit._new_derived_unit(pre.name + der.name,
                                   der.numerator,
                                   der.denominator,
                                   pre.exponent + der.exponent,
                                   der.base_unit_expression)

OTHER_BASE_UNITS = [
    'dB', 'dBm' ]
for name in OTHER_BASE_UNITS:
    Unit._new_base_unit(name)

# Make all the unit objects module variables.
for k,v in _unit_cache.items():
    globals()[k] = v
