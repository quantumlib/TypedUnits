class DerivedUnit:
    def __init__(self,
                 symbol,
                 name,
                 base_unit_expression,
                 exponent=0,
                 numerator=1,
                 denominator=1,
                 use_prefixes=True):
        self.symbol = symbol
        self.name = name
        self.base_unit_expression = base_unit_expression
        self.exponent = exponent
        self.numerator = numerator
        self.denominator = denominator
        self.use_prefixes = use_prefixes

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
    DerivedUnit('Bq', 'becqurel', 'Hz')
]

OTHER_DERIVED_UNITS = [
    DerivedUnit('in', 'inch', 'cm', -2, 254, 1, False),
    DerivedUnit('d', 'day', 's', 2, 864, 1, False),
    DerivedUnit('hr', 'hour', 's', 2, 36, 1, False),
    DerivedUnit('min', 'minute', 's', 1, 6, 1, False),
    DerivedUnit('yr', 'year', 'day', -2, 36525, 1, False)
]

ALL_DERIVED_UNITS = SI_DERIVED_UNITS + OTHER_DERIVED_UNITS
