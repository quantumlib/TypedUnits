class BaseUnit:
    def __init__(self, symbol, name, use_prefixes=True):
        self.symbol = symbol
        self.name = name
        self.use_prefixes = use_prefixes

SI_BASE_UNITS = [
    BaseUnit('m', 'meter'),
    BaseUnit('kg', 'kilogram'),
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
