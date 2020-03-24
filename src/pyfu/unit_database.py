
from . import _all_cythonized, unit_grammar


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
        interpreted as new units or not by default.
        """
        self.known_units = {}
        self.auto_create_units = auto_create_units

    def get_unit(self, unit_name, auto_create=None):
        """
        :param str unit_name:
        :param None|bool auto_create: If this is set, a missing unit will be
        created and returned instead of causing an error. If not specified,
        defaults to the 'auto_create_units' attribute of the receiving instance.
        :return WithUnit: The unit with the given name.
        """
        auto_create = (self.auto_create_units
                       if auto_create is None
                       else auto_create)
        if unit_name not in self.known_units:
            if not auto_create:
                raise KeyError("No unit named '%s'." % unit_name)
            self.add_root_unit(unit_name)
        return self.known_units[unit_name]

    def parse_unit_formula(self, formula, auto_create=None):
        """
        :param str formula: Describes a combination of units.
        :param None|bool auto_create: If this is set, missing unit strings will
        cause new units to be created and returned instead of causing an error.
        If not specified, defaults to the 'auto_create_units' attribute of the
        receiving instance.
        :return WithUnit: The value described by the formula.
        """
        if formula in self.known_units:
            return self.known_units[formula]
        parsed = unit_grammar.unit.parseString(formula)
        result = _all_cythonized.WithUnit(parsed.factor or 1)
        for item in parsed.posexp:
            result *= self._parse_unit_item(item, +1, auto_create)
        for item in parsed.negexp:
            result *= self._parse_unit_item(item, -1, auto_create)
        return result

    def _parse_unit_item(self, item, neg, auto_create=None):
        """
        :param item: A unit+exponent group parsed by unit_grammar.
        :param neg: Are we multiplying (+1) or dividing (-1)?
        :param None|bool auto_create: see parse_unit_formula
        """
        unit_name = item.name
        numer = item.num or 1
        denom = item.denom or 1
        sign = neg * (-1 if item.neg else 1)
        unit_val = self.get_unit(unit_name, auto_create)
        return unit_val ** (sign * float(numer) / denom)

    def add_unit(self, unit_name, unit_base_value):
        """
        Adds a unit to the database, pointing it at the given value.
        :param str unit_name: Key for the new unit.
        :param WithUnit unit_base_value: The unit's value.
        """
        if not isinstance(unit_base_value, _all_cythonized.WithUnit):
            raise TypeError('unit_base_value must be a WithUnit')
        if unit_name in self.known_units:
            raise RuntimeError(
                "Unit name '%s' already taken by '%s'." %
                    (unit_name, self.known_units[unit_name].inBaseUnits()))
        self.known_units[unit_name] = unit_base_value

    def add_root_unit(self, unit_name):
        """
        Adds a plain unit, not defined in terms of anything else, to the database.
        :param str unit_name: Key and unit array entry for the new unit.
        """
        ua = _all_cythonized.UnitArray(unit_name)
        unit = _all_cythonized.raw_WithUnit(
            1,
            {'factor': 1.0, 'ratio': {'numer': 1, 'denom': 1}, 'exp10': 0},
            ua,
            ua)
        self.add_unit(unit_name, unit)

    def add_alternate_unit_name(self, alternate_name, unit_name):
        """
        Adds an alternate name for a unit, mapping to exactly the same value.
        :param str alternate_name: The new alternate name for the unit.
        :param str unit_name: The existing name for the unit.
        """
        self.add_unit(
            alternate_name,
            self.get_unit(unit_name, auto_create=False))

    def add_scaled_unit(self,
                        unit_name,
                        formula,
                        factor=1.0,
                        numer=1,
                        denom=1,
                        exp10=0):
        """
        Creates and adds a derived unit to the database. The unit's value is
        computed by parsing the given formula (in terms of existing units) and
        applying the given scaling parameters.
        :param str unit_name: Name of the derived unit.
        :param str formula: Math expression containing a unit combination.
        :param float factor: A lossy factor for converting to the base unit.
        :param int numer: An exact factor for converting to the base unit.
        :param int denom: An exact divisor for converting to the base unit.
        :param int exp10: An exact power-of-10 for converting to the base unit.
        """
        parent = self.parse_unit_formula(formula, auto_create=False)

        unit = _all_cythonized.raw_WithUnit(
            1,
            {
                'factor': factor * parent.factor * parent.value,
                'ratio': {
                    'numer': numer * parent.numer,
                    'denom': denom * parent.denom
                },
                'exp10': exp10 + parent.exp10
            },
            parent.base_units,
            _all_cythonized.UnitArray(unit_name))

        self.add_unit(unit_name, unit)

    def add_base_unit_data(self, data, prefixes):
        """
        Adds a unit, with alternate names and prefixes, defined by a
        BaseUnitData and some PrefixData.
        :param BaseUnitData data:
        :param list[PrefixData] prefixes:
        """
        self.add_root_unit(data.symbol)
        self.add_alternate_unit_name(data.name, data.symbol)

        symbol = data.symbol
        name = data.name
        if symbol == 'kg':
            symbol = 'g'
            name = 'gram'
            self.add_scaled_unit('g', 'kg', exp10=-3)
            self.add_alternate_unit_name('gram', 'g')

        if data.use_prefixes:
            for pre in prefixes:
                if symbol == 'g' and pre.symbol == 'k':
                    continue
                self.add_scaled_unit(pre.symbol + symbol,
                                     symbol,
                                     exp10=pre.exp10)
                self.add_alternate_unit_name(pre.name + name,
                                             pre.symbol + symbol)

    def add_derived_unit_data(self, data, prefixes):
        """
        Adds a unit, with alternate names and prefixes, defined by a
        DerivedUnitData and some PrefixData.
        :param DerivedUnitData data:
        :param list[PrefixData] prefixes:
        """
        self.add_scaled_unit(data.symbol,
                             data.formula,
                             data.value,
                             data.numerator,
                             data.denominator,
                             data.exp10)
        if data.name is not None:
            self.add_alternate_unit_name(data.name, data.symbol)

        if data.use_prefixes:
            for pre in prefixes:
                self.add_scaled_unit(pre.symbol + data.symbol,
                                     data.formula,
                                     data.value,
                                     data.numerator,
                                     data.denominator,
                                     data.exp10 + pre.exp10)
                if data.name is not None:
                    self.add_alternate_unit_name(pre.name + data.name,
                                                 pre.symbol + data.symbol)

    def add_physical_constant_data(self, data):
        """
        Adds a physical constant, i.e. a unit that doesn't override the display
        units of a value, defined by a PhysicalConstantData.
        :param PhysicalConstantData data:
        """
        val = self.parse_unit_formula(data.formula, auto_create=False)
        val *= data.factor
        self.add_unit(data.symbol, val)
        if data.name is not None:
            self.add_alternate_unit_name(data.name, data.symbol)

    def _expected_value_for_unit_array(self, unit_array):
        """
        Determines the expected conversion factor for the given UnitArray by
        looking up its unit names in the database and multiplying up each of
        their contributions.

        :param (str, int, int) unit_array: An array of unit names and exponents.
        :return Value: A value containing the net conversion factor.
        """
        result = _all_cythonized.Value(1)
        for name, exp_numer, exp_denom in unit_array:
            if name not in self.known_units:
                return None
            result *= self.known_units[name] ** (float(exp_numer) / exp_denom)
        return result

    def is_value_consistent_with_database(self, value):
        """
        Determines if the value's base and display units are known and that
        the conversion factor between them is consistent with the known unit
        scales.

        :param WithUnit value:
        :return bool:
        """
        base = self._expected_value_for_unit_array(value.base_units)
        display = self._expected_value_for_unit_array(value.display_units)

        # Any unknown units?
        if base is None or display is None:
            return False

        # Any units mapped to a different base unit?
        if display.base_units != value.base_units:
            return False

        # Conversion makes sense?
        conv = value.unit * base / display
        c = conv.numer / conv.denom * (10**conv.exp10) * conv.factor
        return abs(c - 1) < 0.0001
