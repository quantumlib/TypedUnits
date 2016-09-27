import math


class DerivedUnit:
    def __init__(self,
                 symbol,
                 name,
                 base_unit_expression,
                 value=1.0,
                 exponent=0,
                 numerator=1,
                 denominator=1,
                 use_prefixes=True):
        """
        :param symbol: The short name for the unit (e.g. 'm' for meter).
        :param name: The full name of the unit (e.g. 'meter').
        :param base_unit_expression: A physical unit scale factor.
        :param value: A floating-point scale factor.
        :param exponent: An integer power-of-10 exponent scale factor.
        :param numerator: A small integer scale factor.
        :param denominator: A small integer inverse scale factor.
        :param use_prefixes: Should there be 'kiloUNIT', 'milliUNIT', etc.
        """
        self.symbol = symbol
        self.name = name
        self.base_unit_expression = base_unit_expression
        self.value = value
        self.exponent = exponent
        self.numerator = numerator
        self.denominator = denominator
        self.use_prefixes = use_prefixes

SI_REDUNDANT_BASE_UNITS = [
    DerivedUnit('sr', 'steradian', 'rad^2'),
]

SI_DERIVED_UNITS = [
    DerivedUnit('Hz', 'hertz', '1/s'),
    DerivedUnit('N', 'newton', 'kg*m/s^2'),
    DerivedUnit('Pa', 'pascal', 'N/m^2'),
    DerivedUnit('J', 'joule', 'N*m'),
    DerivedUnit('W', 'watt', 'J/s'),
    DerivedUnit('C', 'coulomb', 'A*s'),
    DerivedUnit('V', 'volt', 'W/A'),
    DerivedUnit('F', 'farad', 'J/C'),
    DerivedUnit('Ohm', 'ohm', 'V/A'),
    DerivedUnit('S', 'siemens', 'A/V'),
    DerivedUnit('Wb', 'weber', 'V*s'),
    DerivedUnit('T', 'tesla', 'Wb/m^2'),
    DerivedUnit('Gauss', 'gauss', 'T', -4),
    DerivedUnit('H', 'henry', 'Wb/A'),
    DerivedUnit('lm', 'lumen', 'cd*sr'),
    DerivedUnit('lx', 'lux', 'lm/m^2'),
    DerivedUnit('Bq', 'becquerel', 'Hz'),
]

OTHER_UNITS = [
    DerivedUnit('in', 'inch', 'cm', 1, -2, 254, 1, False),
    DerivedUnit('h', 'hour', 's', 1, 2, 36, 1, False),
    DerivedUnit('min', 'minute', 's', 1, 1, 6, 1, False),
    DerivedUnit('cyc', 'cycle', 'rad', 2*math.pi, 0, 1, 1, False),
]

# Units that aren't technically exact, but close enough for our purposes.
APPROXIMATE_CIVIL_UNITS = [
    DerivedUnit('d', 'day', 's', 1, 2, 864, 1, False),
    DerivedUnit('yr', 'year', 'day', 1, -2, 36525, 1, False),
]

ALL_DERIVED_UNITS = (SI_REDUNDANT_BASE_UNITS +
                     SI_DERIVED_UNITS +
                     OTHER_UNITS +
                     APPROXIMATE_CIVIL_UNITS)
