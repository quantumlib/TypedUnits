from __all_cythonized import WithUnit, UnitArray
import unit_grammar


class UnitDatabase(object):
    """
    Values defined in unit_array do not actually store a unit object, the unit
    names and powers are stored within the value object itself.  However, when
    constructing new values or converting between units, we need a database of
    known units.
    """
    def __init__(self, auto_create_units=True):
        """
        :param auto_create_units: Determines if unrecognized strings are
        interpreted as new units or not.
        """
        self.known_units = {}
        self.auto_create_units = auto_create_units

    def get_unit(self, unit_name):
        """
        :param str unit_name:
        :return WithUnit:
        """
        if self.auto_create_units and unit_name not in self.known_units:
            self.add_root_unit(unit_name)
        return self.known_units[unit_name]

    def parse_unit_formula(self, formula):
        """
        :param str formula:
        :return WithUnit:
        """
        if formula == '':
            return self.get_unit('')
        parsed = unit_grammar.unit.parseString(formula)
        result = WithUnit(1)
        for item in parsed.posexp:
            result *= self._parse_unit_item(item, +1)
        for item in parsed.negexp:
            result *= self._parse_unit_item(item, -1)
        return result

    def _parse_unit_item(self, item, neg):
        unit_name = item.name
        numer = item.num or 1
        denom = item.denom or 1
        sign = neg * (-1 if item.neg else 1)
        return self.get_unit(unit_name) ** (sign * float(numer) / denom)

    def add_unit(self, unit_name, unit_base_value):
        """
        :param str unit_name:
        :param WithUnit unit_base_value:
        """
        if not isinstance(unit_base_value, WithUnit):
            raise TypeError('unit_base_value must be a WithUnit')
        if unit_name in self.known_units:
            raise RuntimeError("Unit name already taken: " + repr(unit_name))
        self.known_units[unit_name] = unit_base_value

    def add_root_unit(self, unit_name):
        """
        :param str unit_name:
        """
        ua = UnitArray(unit_name)
        unit = WithUnit.raw(1, 1, 1, 0, ua, ua)
        self.add_unit(unit_name, unit)

    def add_scaled_unit(self,
                        unit_name,
                        formula,
                        value=1,
                        numer=1,
                        denom=1,
                        exp10=0):
        """
        :param str unit_name:
        :param str formula:
        :param int|float|complex value:
        :param int numer:
        :param int denom:
        :param int exp10:
        """
        base_unit = self.parse_unit_formula(formula)
        value *= base_unit.value
        numer *= base_unit.numer
        denom *= base_unit.denom
        exp10 += base_unit.exp10

        unit = WithUnit.raw(
            value,
            numer,
            denom,
            exp10,
            base_unit.base_units,
            UnitArray(unit_name))

        self.add_unit(unit_name, unit)

    def add_base_unit_data(self, data, prefixes):
        """
        :param BaseUnitData data:
        :param list[PrefixData] prefixes:
        """
        self.add_root_unit(data.symbol)
        self.add_scaled_unit(data.name, data.symbol)

        symbol = data.symbol
        name = data.name
        if symbol == 'kg':
            symbol = 'g'
            name = 'gram'
            self.add_scaled_unit(symbol, 'kg', exp10=-3)
            self.add_scaled_unit(name, 'kg', exp10=-3)

        if data.use_prefixes:
            for pre in prefixes:
                if symbol == 'g' and pre.symbol == 'k':
                    continue
                for key in [pre.symbol + symbol, pre.name + name]:
                    self.add_scaled_unit(key,
                                         symbol,
                                         exp10=pre.exp10)

    def add_derived_unit_data(self, data, prefixes):
        """
        :param DerivedUnitData data:
        :param list[PrefixData] prefixes:
        """
        for key in [data.symbol, data.name]:
            self.add_scaled_unit(key,
                                 data.formula,
                                 data.value,
                                 data.numerator,
                                 data.denominator,
                                 data.exp10)

        if data.use_prefixes:
            for pre in prefixes:
                for key in [pre.symbol + data.symbol, pre.name + data.name]:
                    self.add_scaled_unit(key,
                                         data.formula,
                                         data.value,
                                         data.numerator,
                                         data.denominator,
                                         data.exp10 + pre.exp10)
