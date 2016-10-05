import unittest
from pyfu import UnitArray, WithUnit
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
        c = db.get_unit('cats')
        u = UnitArray.raw([('cats', 1, 1)])
        self.assertEquals(c.base_units, u)
        self.assertEquals(c.display_units, u)
        self.assertEquals(c.numer, 1)
        self.assertEquals(c.denom, 1)
        self.assertEquals(c.exp10, 0)
        self.assertEquals(c.value, 1)
        self.assertEquals(c, WithUnit.raw(1, 1, 1, 0, u, u))

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

        u = UnitArray.raw([('b', 1, 1)])
        v = WithUnit.raw(1, 1, 1, 0, u, u)

        self.assertEquals(db.get_unit('base'), v)
        self.assertEquals(db.get_unit('b'), v)
        self.assertEquals(db.get_unit('p_b'), v * 10)
        self.assertEquals(db.get_unit('q_b'), v * 100)
        self.assertEquals(db.get_unit('pre_base'), v * 10)
        self.assertEquals(db.get_unit('qu_base'), v * 100)
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

        u = UnitArray.raw([('b', 1, 1)])
        v = WithUnit.raw(1, 1, 1, 0, u, u)

        self.assertEquals(db.get_unit('base'), v)
        self.assertEquals(db.get_unit('b'), v)
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
                value=7j,
                exp10=5,
                numerator=3,
                denominator=2,
                use_prefixes=True),
            [
                PrefixData('s_', 'super_', 1),
                PrefixData('d_', 'duper_', 2),
            ])

        v = db.get_unit('shirts') * 7j * (10**5 * 3) / 2

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
                value=7j,
                exp10=5,
                numerator=3,
                denominator=2,
                use_prefixes=False),
            [PrefixData('s_', 'super_', 1)])

        v = db.get_unit('shirts') * 7j * (10**5 * 3) / 2

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
            UnitArray.raw([('kg', 1, 1)]))
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
