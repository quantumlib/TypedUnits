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
                 use_prefixes=True):
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
    DerivedUnitData('sr', 'steradian', 'rad^2'),
]

__SI_DERIVED_UNITS = [
    DerivedUnitData('Hz', 'hertz', '1/s'),
    DerivedUnitData('N', 'newton', 'kg*m/s^2'),
    DerivedUnitData('Pa', 'pascal', 'N/m^2'),
    DerivedUnitData('J', 'joule', 'N*m'),
    DerivedUnitData('W', 'watt', 'J/s'),
    DerivedUnitData('C', 'coulomb', 'A*s'),
    DerivedUnitData('V', 'volt', 'W/A'),
    DerivedUnitData('F', 'farad', 'J/C'),
    DerivedUnitData('Ohm', 'ohm', 'V/A'),
    DerivedUnitData('S', 'siemens', 'A/V'),
    DerivedUnitData('Wb', 'weber', 'V*s'),
    DerivedUnitData('T', 'tesla', 'Wb/m^2'),
    DerivedUnitData('Gauss', 'gauss', 'T', -4),
    DerivedUnitData('H', 'henry', 'Wb/A'),
    DerivedUnitData('lm', 'lumen', 'cd*sr'),
    DerivedUnitData('lx', 'lux', 'lm/m^2'),
    DerivedUnitData('Bq', 'becquerel', 'Hz'),
]

__OTHER_UNITS = [
    DerivedUnitData('in', 'inch', 'cm', 1, -2, 254, 1, False),
    DerivedUnitData('h', 'hour', 's', 1, 2, 36, 1, False),
    DerivedUnitData('min', 'minute', 's', 1, 1, 6, 1, False),
    DerivedUnitData('cyc', 'cycle', 'rad', 2 * math.pi, 0, 1, 1, False),
]

# Units that aren't technically exact, but close enough for our purposes.
__APPROXIMATE_CIVIL_UNITS = [
    DerivedUnitData('d', 'day', 's', 1, 2, 864, 1, False),
    DerivedUnitData('yr', 'year', 'day', 1, -2, 36525, 1, False),
]

ALL_DERIVED_UNITS = (__SI_REDUNDANT_BASE_UNITS +
                     __SI_DERIVED_UNITS +
                     __OTHER_UNITS +
                     __APPROXIMATE_CIVIL_UNITS)
