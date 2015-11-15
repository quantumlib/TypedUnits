#!/usr/bin/env python

import unit_array
from unit_array import Value
import unit_grammar

"""
import fast_units
fast_units.Unit._new_base_unit('m')
fast_units.Unit._new_derived_unit('km', 1, 1, 3, 'm')
fast_units.Unit('km*m')

"""



class Unit(object):
    _cache = {}
    #__array_priority__ = 15
    __slots__ = ['_value']

    def __new__(cls, name):
        if isinstance(name, Unit):
            return name
        if name in cls._cache:
            return cls._cache[name]
        else:
            return cls.parse_unit_str(name)

    @classmethod
    def _new_from_value(cls, val):
        if not isinstance(val, unit_array.Value):
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
        if base_name not in cls._cache:
            print "base_name not in unit dictionary, adding as extension unit"
            base_unit = unit_array.UnitArray(base_name)
            cls._cache[base_name] = Value(1, 1, 1, 0, base_unit, base_unit)
        else:
            print "power of existing unit %s: %s" % (base_name, cls._cache[base_name])
        print "calculating unit power: %s %d*%d/%d" % (base_name, sign, numer, denom)
        element = cls._cache[base_name]**(1.0*sign*numer/denom)
        print "element: ", element
        return element

    @classmethod
    def _new_derived_unit(cls, name, numer, denom, exp10, base_unit):
        if isinstance(base_unit, str):
            base_unit = Unit(base_unit)
        numer = numer * base_unit._value.numer
        denom = denom * base_unit._value.denom
        exp10 = exp10 + base_unit._value.exp10
        val = Value(1, numer, denom, exp10, base_unit._value.base_units, unit_array.UnitArray(name))
        result = cls._new_from_value(val)
        cls._cache[name] = result
        return result

    @classmethod
    def _new_base_unit(cls, name):
        if name in cls._cache:
            raise RuntimeError("Trying to create unit that already exists")
        ua = unit_array.UnitArray(name)
        val = Value(1, 1, 1, 0, ua, ua)
        result = cls._new_from_value(val)
        cls._cache[name] = result
        return result

    @classmethod
    def parse_unit_str(cls, name):
        parsed = unit_grammar.unit.parseString(name)
        print parsed
        result = Unit('')
          
        for item in parsed.posexp:
            print result
            element = cls._unit_from_parse_item(item, 0)
            result = result * element
        for item in parsed.negexp:
            print result
            result = result * cls._unit_from_parse_item(item, -1)
        return result
            
    def __mul__(self, other):
        if isinstance(other, Unit):
            return Unit._new_from_value(other._value * self._value)
        return self._value * other

    __rmul__ = __mul__
    
    def __div__(self, other):
        if isinstance(other, Unit):
            return Unit._new_from_value(self._value/other._value)
        return self._value/other

    def __rdiv__(self, other):
        if isinstance(other, Unit):
            return Unit._new_from_value(other._value/self._value)
        return other/self._value

    def __pow__(self, other):
        print "unit power: %s [%s]" % (other, type(other))
        return Unit._new_from_value(self._value**other)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @property
    def name(self):
        return str(self)

    def base_unit(self):
        tmp = self._.value.inBaseUnits()
        
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
        return self._value.base_units == self._cache['rad'].base_units

Unit._cache[''] = Unit._new_from_value(Value(1,1,1,0, unit_array.DimensionlessUnit, unit_array.DimensionlessUnit))
