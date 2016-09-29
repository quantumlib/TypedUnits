class PrefixData:
    """
    Describes the properties of a unit prefix.
    """
    def __init__(self, symbol, name, exp10):
        """
        :param str symbol: The short name for the prefix (e.g. 'G' for giga).
        :param str name: The full name of the prefix (e.g. 'giga').
        :param int exp10: The power of 10 the prefix corresponds to.
        """
        self.symbol = symbol
        self.name = name
        self.exp10 = exp10

SI_PREFIXES = [
    PrefixData('Y', 'yotta', 24),
    PrefixData('Z', 'zetta', 21),
    PrefixData('E', 'exa', 18),
    PrefixData('P', 'peta', 15),
    PrefixData('T', 'tera', 12),
    PrefixData('G', 'giga', 9),
    PrefixData('M', 'mega', 6),
    PrefixData('k', 'kilo', 3),
    PrefixData('h', 'hecto', 2),
    PrefixData('da', 'deka', 1),
    PrefixData('d', 'deci', -1),
    PrefixData('c', 'centi', -2),
    PrefixData('m', 'milli', -3),
    PrefixData('u', 'micro', -6),
    PrefixData('n', 'nano', -9),
    PrefixData('p', 'pico', -12),
    PrefixData('f', 'femto', -15),
    PrefixData('a', 'atto', -18),
    PrefixData('z', 'zepto', -21),
    PrefixData('y', 'yocto', -24),
]
