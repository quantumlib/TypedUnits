class Prefix:
    def __init__(self, symbol, name, exponent):
        self.symbol = symbol
        self.name = name
        self.exponent = exponent

SI_PREFIXES = [
    Prefix('Y', 'yotta', 24),
    Prefix('Z', 'zetta', 21),
    Prefix('E', 'exa', 18),
    Prefix('P', 'peta', 15),
    Prefix('T', 'tera', 12),
    Prefix('G', 'giga', 9),
    Prefix('M', 'mega', 6),
    Prefix('k', 'kilo', 3),
    Prefix('h', 'hecto', 2),
    Prefix('da', 'deka', 1),
    Prefix('d', 'deci', -1),
    Prefix('c', 'centi', -2),
    Prefix('m', 'milli', -3),
    Prefix('u', 'micro', -6),
    Prefix('n',' nano', -9),
    Prefix('p', 'pico', -12),
    Prefix('f', 'femto', -15),
    Prefix('a', 'atto', -18),
    Prefix('z', 'zepto', -21),
    Prefix('y', 'yocto', -24),
]
