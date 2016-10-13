from __future__ import absolute_import
import math


class DerivedUnitData:
    """
    Describes the properties of a derived unit.
    """
    def __init__(self,
                 symbol,
                 name,
                 formula,
                 value=1.0,
                 exp10=0,
                 numerator=1,
                 denominator=1,
                 use_prefixes=False):
        """
        :param str symbol: The short name for the unit (e.g. 'm' for meter).
        :param str name: The full name of the unit (e.g. 'meter').
        :param str formula: A formula defining the unit in terms of others.
        :param int|float|complex value: A floating-point scale factor.
        :param int exp10: An integer power-of-10 exponent scale factor.
        :param int numerator: A small integer scale factor.
        :param int denominator: A small integer inverse scale factor.
        :param bool use_prefixes: Should there be 'kiloUNIT', 'milliUNIT', etc.
        """
        self.symbol = symbol
        self.name = name
        self.formula = formula
        self.value = value
        self.exp10 = exp10
        self.numerator = numerator
        self.denominator = denominator
        self.use_prefixes = use_prefixes

__SI_REDUNDANT_BASE_UNITS = [
    DerivedUnitData('sr', 'steradian', 'rad^2', use_prefixes=True),
]

__SI_DERIVED_UNITS = [
    DerivedUnitData('Hz', 'hertz', '1/s', use_prefixes=True),
    DerivedUnitData('N', 'newton', 'kg*m/s^2', use_prefixes=True),
    DerivedUnitData('Pa', 'pascal', 'N/m^2', use_prefixes=True),
    DerivedUnitData('J', 'joule', 'N*m', use_prefixes=True),
    DerivedUnitData('W', 'watt', 'J/s', use_prefixes=True),
    DerivedUnitData('C', 'coulomb', 'A*s', use_prefixes=True),
    DerivedUnitData('V', 'volt', 'W/A', use_prefixes=True),
    DerivedUnitData('F', 'farad', 'C/V', use_prefixes=True),
    DerivedUnitData('Ohm', 'ohm', 'V/A', use_prefixes=True),
    DerivedUnitData('S', 'siemens', 'A/V', use_prefixes=True),
    DerivedUnitData('Wb', 'weber', 'V*s', use_prefixes=True),
    DerivedUnitData('T', 'tesla', 'Wb/m^2', use_prefixes=True),
    DerivedUnitData('Gauss', 'gauss', 'T', exp10=-4, use_prefixes=True),
    DerivedUnitData('H', 'henry', 'Wb/A', use_prefixes=True),
    DerivedUnitData('lm', 'lumen', 'cd*sr', use_prefixes=True),
    DerivedUnitData('lx', 'lux', 'lm/m^2', use_prefixes=True),
    DerivedUnitData('Bq', 'becquerel', 'Hz', use_prefixes=True),
]

__OTHER_UNITS = [
    DerivedUnitData('in', 'inch', 'cm', numerator=254, exp10=-2),
    DerivedUnitData('h', 'hour', 's', numerator=36, exp10=2),
    DerivedUnitData('min', 'minute', 's', numerator=6, exp10=1),
    DerivedUnitData('cyc', 'cycle', 'rad', 2 * math.pi),
]

# Units that aren't technically exact, but close enough for our purposes.
__APPROXIMATE_CIVIL_UNITS = [
    DerivedUnitData('d', 'day', 's', numerator=864, exp10=2),
    DerivedUnitData('yr', 'year', 'day', numerator=36525, exp10=-2),
]

ALL_DERIVED_UNITS = (__SI_REDUNDANT_BASE_UNITS +
                     __SI_DERIVED_UNITS +
                     __OTHER_UNITS +
                     __APPROXIMATE_CIVIL_UNITS)
