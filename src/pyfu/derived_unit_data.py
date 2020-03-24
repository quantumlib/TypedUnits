"""
Defines named physical units, specified in terms of other units, exposed in the
default unit database and as members of pyfu.units (and
pyfu.like_pylabrad_units).

Derived units are defined by conversion parameters and unit formulas. The
formulas can mention base units and earlier derived units (but not later derived
units or physical constants).
"""

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
        :param str None|name: A full name for the unit (e.g. 'meter').
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
    DerivedUnitData('l', 'liter', 'm^3', exp10=-3, use_prefixes=True),
]

__OTHER_UNITS = [
    # Lengths.
    DerivedUnitData('in', 'inch', 'cm', 2, numerator=127, exp10=-2),
    DerivedUnitData('ft', 'foot', 'in', 4, numerator=3),
    DerivedUnitData('yd', 'yard', 'ft', numerator=3),
    DerivedUnitData('nmi', 'nautical_mile', 'm', 1852),
    DerivedUnitData('Ang', 'angstrom', 'm', exp10=-10),
    DerivedUnitData('ly', 'light_year', 'm', 94607304725808, exp10=2),
    DerivedUnitData('lyr', None, 'ly'),
    # Durations.
    DerivedUnitData('h', 'hour', 's', 4, numerator=9, exp10=2),
    DerivedUnitData('min', 'minute', 's', 2, numerator=3, exp10=1),
    # Angles.
    DerivedUnitData('cyc', 'cycle', 'rad', 2 * math.pi),
    DerivedUnitData('deg', None, 'rad', math.pi / 180),
    # Areas.
    DerivedUnitData('ha', 'hectare', 'm^2', exp10=4),
    DerivedUnitData('b', 'barn', 'm^2', exp10=-28),
    # Volumes.
    DerivedUnitData('tsp', 'teaspoon', 'ml', 4.92892159375),
    DerivedUnitData('tbsp', 'tablespoon', 'tsp', numerator=3),
    DerivedUnitData('floz', 'fluid_ounce', 'tbsp', 2),
    DerivedUnitData('cup', None, 'floz', 8),
    DerivedUnitData('pint', None, 'floz', 16),
    DerivedUnitData('qt', 'quart', 'pint', 2),
    DerivedUnitData('galUS', 'us_gallon', 'qt', 4),
    DerivedUnitData('galUK', 'british_gallon', 'l', 4.54609),
    # Mass.
    DerivedUnitData('oz', 'ounce', 'g', 28.349523125),
    DerivedUnitData('lb', 'pound', 'oz', 16),
    DerivedUnitData('ton', None, 'lb', 2000),
    # Pressure.
    DerivedUnitData('psi', 'pounds_per_square_inch', 'Pa', 6894.75729317),
]

# Units that aren't technically exact, but close enough for our purposes.
__APPROXIMATE_CIVIL_UNITS = [
    DerivedUnitData('d', 'day', 's', 32, numerator=27, exp10=2),
    DerivedUnitData('wk', 'week', 'day', numerator=7),
    DerivedUnitData('yr', 'year', 'day', 365.25),
]

ALL_DERIVED_UNITS = (__SI_REDUNDANT_BASE_UNITS +
                     __SI_DERIVED_UNITS +
                     __OTHER_UNITS +
                     __APPROXIMATE_CIVIL_UNITS)
