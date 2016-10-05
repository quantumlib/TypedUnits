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
    PrefixData('Y', 'yotta', exp10=24),
    PrefixData('Z', 'zetta', exp10=21),
    PrefixData('E', 'exa', exp10=18),
    PrefixData('P', 'peta', exp10=15),
    PrefixData('T', 'tera', exp10=12),
    PrefixData('G', 'giga', exp10=9),
    PrefixData('M', 'mega', exp10=6),
    PrefixData('k', 'kilo', exp10=3),
    PrefixData('h', 'hecto', exp10=2),
    PrefixData('da', 'deka', exp10=1),
    PrefixData('d', 'deci', exp10=-1),
    PrefixData('c', 'centi', exp10=-2),
    PrefixData('m', 'milli', exp10=-3),
    PrefixData('u', 'micro', exp10=-6),
    PrefixData('n', 'nano', exp10=-9),
    PrefixData('p', 'pico', exp10=-12),
    PrefixData('f', 'femto', exp10=-15),
    PrefixData('a', 'atto', exp10=-18),
    PrefixData('z', 'zepto', exp10=-21),
    PrefixData('y', 'yocto', exp10=-24),
]
