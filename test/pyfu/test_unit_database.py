import unittest
from pyfu._all_cythonized import raw_WithUnit, raw_UnitArray
from pyfu.unit_database import UnitDatabase
from pyfu.base_unit_data import BaseUnitData
from pyfu.derived_unit_data import DerivedUnitData
from pyfu.prefix_data import PrefixData, SI_PREFIXES
from pyparsing import ParseException


class UnitDatabaseTests(unittest.TestCase):
    def testAutoCreate(self):
        with self.assertRaises(KeyError):
            UnitDatabase(auto_create_units=False).parse_unit_formula('tests')

        db = UnitDatabase(auto_create_units=True)
        u = db.parse_unit_formula('tests')
        self.assertEquals(5, 5 * u / u)

    def testAddRootUnit(self):
        db = UnitDatabase(auto_create_units=False)
        db.add_root_unit('cats')

        # Root unit is simple.
        c = db.get_unit('cats')
        self.assertEquals(c.base_units, raw_UnitArray([('cats', 1, 1)]))
        self.assertEquals(c.display_units, raw_UnitArray([('cats', 1, 1)]))
        self.assertEquals(c.numer, 1)
        self.assertEquals(c.denom, 1)
        self.assertEquals(c.exp10, 0)
        self.assertEquals(c.value, 1)

        # No dups.
        with self.assertRaises(RuntimeError):
            db.add_root_unit('cats')

    def testAddBaseUnitWithPrefixes(self):
        db = UnitDatabase(auto_create_units=False)
        db.add_base_unit_data(
            BaseUnitData('b', 'base', True),
            [
                PrefixData('p_', 'pre_', 1),
                PrefixData('q_', 'qu_', 2),
            ])

        # Long form *is* short form.
        self.assertIs(db.get_unit('base'), db.get_unit('b'))
        self.assertIs(db.get_unit('pre_base'), db.get_unit('p_b'))
        self.assertIs(db.get_unit('qu_base'), db.get_unit('q_b'))

        # Root unit is simple.
        u = db.get_unit('b')
        self.assertEquals(u.base_units, raw_UnitArray([('b', 1, 1)]))
        self.assertEquals(u.display_units, raw_UnitArray([('b', 1, 1)]))
        self.assertEquals(u.numer, 1)
        self.assertEquals(u.denom, 1)
        self.assertEquals(u.exp10, 0)
        self.assertEquals(u.value, 1)

        # Prefixes do scaling.
        self.assertEquals(db.get_unit('p_b'), u * 10)
        self.assertEquals(db.get_unit('q_b'), u * 100)

        # No mixing long prefixes with short units or vice versa.
        with self.assertRaises(KeyError):
            db.get_unit('p_base')
        with self.assertRaises(KeyError):
            db.get_unit('pre_b')

    def testAddBaseUnitWithoutPrefixes(self):
        db = UnitDatabase(auto_create_units=False)
        db.add_base_unit_data(
            BaseUnitData('b', 'base', False),
            [
                PrefixData('p_', 'pre_', 1),
                PrefixData('q_', 'qu_', 2),
            ])

        # Long form *is* short form.
        self.assertIs(db.get_unit('base'), db.get_unit('b'))

        # Root unit is simple.
        u = db.get_unit('b')
        self.assertEquals(u.base_units, raw_UnitArray([('b', 1, 1)]))
        self.assertEquals(u.display_units, raw_UnitArray([('b', 1, 1)]))
        self.assertEquals(u.numer, 1)
        self.assertEquals(u.denom, 1)
        self.assertEquals(u.exp10, 0)
        self.assertEquals(u.value, 1)

        # No prefixing.
        with self.assertRaises(KeyError):
            db.get_unit('p_b')
        with self.assertRaises(KeyError):
            db.get_unit('q_b')
        with self.assertRaises(KeyError):
            db.get_unit('pre_base')
        with self.assertRaises(KeyError):
            db.get_unit('qu_base')
        with self.assertRaises(KeyError):
            db.get_unit('p_base')
        with self.assertRaises(KeyError):
            db.get_unit('pre_b')

    def testAddDerivedUnitWithPrefixes(self):
        db = UnitDatabase(auto_create_units=False)
        with self.assertRaises(KeyError):
            db.add_derived_unit_data(
                DerivedUnitData('tails', 't', 'shirts'),
                [])

        db.add_root_unit('shirts')
        db.add_derived_unit_data(
            DerivedUnitData(
                't',
                'tails',
                'shirts',
                value=7.0,
                exp10=5,
                numerator=3,
                denominator=2,
                use_prefixes=True),
            [
                PrefixData('s_', 'super_', 1),
                PrefixData('d_', 'duper_', 2),
            ])

        v = db.get_unit('shirts') * 7.0 * (10**5 * 3) / 2

        self.assertEquals(db.get_unit('tails'), v)
        self.assertEquals(db.get_unit('t'), v)
        self.assertEquals(db.get_unit('s_t'), v * 10)
        self.assertEquals(db.get_unit('super_tails'), v * 10)
        self.assertEquals(db.get_unit('d_t'), v * 100)
        self.assertEquals(db.get_unit('duper_tails'), v * 100)
        with self.assertRaises(KeyError):
            db.get_unit('s_tails')
        with self.assertRaises(KeyError):
            db.get_unit('super_t')

    def testAddDerivedUnitWithoutPrefixes(self):
        db = UnitDatabase(auto_create_units=False)

        db.add_root_unit('shirts')
        db.add_derived_unit_data(
            DerivedUnitData(
                't',
                'tails',
                'shirts',
                value=7.0,
                exp10=5,
                numerator=3,
                denominator=2,
                use_prefixes=False),
            [PrefixData('s_', 'super_', 1)])

        v = db.get_unit('shirts') * 7 * (10**5 * 3) / 2

        self.assertEquals(db.get_unit('tails'), v)
        self.assertEquals(db.get_unit('t'), v)
        with self.assertRaises(KeyError):
            db.get_unit('super_tails')
        with self.assertRaises(KeyError):
            db.get_unit('s_t')
        with self.assertRaises(KeyError):
            db.get_unit('s_tails')
        with self.assertRaises(KeyError):
            db.get_unit('super_t')

    def testKilogramSpecialCase(self):
        db = UnitDatabase(auto_create_units=False)
        db.add_base_unit_data(BaseUnitData('kg', 'kilogram'), SI_PREFIXES)
        self.assertEquals(
            db.get_unit('g').base_units,
            raw_UnitArray([('kg', 1, 1)]))
        self.assertEquals(db.get_unit('g') * 1000, db.get_unit('kg'))
        self.assertEquals(db.get_unit('kg') * 1000, db.get_unit('Mg'))

    def testParseUnitFormula(self):
        db = UnitDatabase(auto_create_units=False)
        db.add_root_unit('cats')
        db.add_root_unit('dogs')
        db.add_root_unit('mice')
        cats = db.get_unit('cats')
        dogs = db.get_unit('dogs')
        mice = db.get_unit('mice')

        with self.assertRaises(ParseException):
            db.parse_unit_formula('cats^dogs')

        self.assertEquals(db.parse_unit_formula('cats'), cats)
        self.assertEquals(db.parse_unit_formula('cats^2'), cats**2)
        self.assertEquals(db.parse_unit_formula('cats^-2'), cats**-2)
        self.assertEquals(db.parse_unit_formula('cats*cats'), cats**2)
        self.assertEquals(db.parse_unit_formula('cats*dogs'), cats * dogs)
        self.assertEquals(db.parse_unit_formula('cats/dogs'), cats / dogs)
        self.assertEquals(db.parse_unit_formula('cats/dogs^2'), cats / dogs**2)
        self.assertEquals(
            db.parse_unit_formula('cats/dogs*mice'), (cats / dogs) * mice)

if __name__ == "__main__":
    unittest.main()
