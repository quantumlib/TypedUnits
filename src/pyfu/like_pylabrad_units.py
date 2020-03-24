"""
A compatibility layer that tries to expose the same API as pylabrad's unit
library.
"""

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
        if not isinstance(obj, WithUnit):
            obj = _unit.default_unit_database.parse_unit_formula(obj)
        super(Value, self).__init__(obj)

    @property
    def name(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, str):
            other = _unit.default_unit_database.parse_unit_formula(other)
        return Value(self) == other

    def __ne__(self, other):
        return not (self == other)

    def __mul__(self, other):
        product = Value(self) * other
        return Unit(product) if isinstance(other, Unit) else product

    def __truediv__(self, other):
        quotient = Value(self) / other
        return Unit(quotient) if isinstance(other, Unit) else quotient

    def __floordiv__(self, other):
        quotient = Value(self) / other
        return Unit(quotient) if isinstance(other, Unit) else quotient

    def __pow__(self, exponent, modulus=None):
        return Unit(Value(self).__pow__(exponent, modulus))


# Expose defined units (e.g. 'meter', 'km', 'day') as module variables.
# Note: unless you like overwriting Boltzmann's constant, keep the underscores.
for _k, _v in _unit.default_unit_database.known_units.items():
    globals()[_k] = Unit(_v)
