class BaseUnit:
    def __init__(self, symbol, name, use_prefixes=True):
        """
        :param symbol: The short name for the unit (e.g. 'm' for meter).
        :param name: The full name of the unit (e.g. 'meter').
        :param use_prefixes: Should there be 'kiloUNIT', 'milliUNIT', etc.
        """
        self.symbol = symbol
        self.name = name
        self.use_prefixes = use_prefixes

SI_BASE_UNITS = [
    BaseUnit('m', 'meter'),
    BaseUnit('kg', 'kilogram'),  # Note: causes special-cased prefix behavior.
    BaseUnit('s', 'second'),
    BaseUnit('A', 'ampere'),
    BaseUnit('K', 'kelvin'),
    BaseUnit('mol', 'mole'),
    BaseUnit('cd', 'candela'),
    BaseUnit('rad', 'radian'),
]

OTHER_BASE_UNITS = [
    BaseUnit('dB', 'decibel', False),
]

ALL_BASE_UNITS = SI_BASE_UNITS + OTHER_BASE_UNITS
