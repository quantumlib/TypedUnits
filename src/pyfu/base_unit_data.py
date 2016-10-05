from __future__ import absolute_import


class BaseUnitData:
    """
    Describes the properties of a base unit.
    """
    def __init__(self, symbol, name, use_prefixes=True):
        """
        :param str symbol: The short name for the unit (e.g. 'm' for meter).
        :param str name: The full name of the unit (e.g. 'meter').
        :param bool use_prefixes: Should there be 'kiloUNIT', 'milliUNIT', etc.
        """
        self.symbol = symbol
        self.name = name
        self.use_prefixes = use_prefixes

__SI_BASE_UNITS = [
    BaseUnitData('m', 'meter'),
    BaseUnitData('kg', 'kilogram'),  # Note: has special-cased prefix behavior.
    BaseUnitData('s', 'second'),
    BaseUnitData('A', 'ampere'),
    BaseUnitData('K', 'kelvin'),
    BaseUnitData('mol', 'mole'),
    BaseUnitData('cd', 'candela'),
    BaseUnitData('rad', 'radian'),
]

__OTHER_BASE_UNITS = [
    BaseUnitData('dB', 'decibel', use_prefixes=False),
]

ALL_BASE_UNITS = __SI_BASE_UNITS + __OTHER_BASE_UNITS
