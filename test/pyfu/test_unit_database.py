import pytest
from pyfu._all_cythonized import raw_WithUnit, raw_UnitArray
from pyparsing import ParseException

from pyfu.base_unit_data import BaseUnitData
from pyfu.derived_unit_data import DerivedUnitData
from pyfu.prefix_data import PrefixData, SI_PREFIXES
from pyfu.unit_database import UnitDatabase


def frac(numer=1, denom=1):
    return {'numer': numer, 'denom': denom}


def unit(key):
    return raw_UnitArray([(key, 1, 1)])


def conv(factor=1.0, numer=1, denom=1, exp10=0):
    return {'factor': factor, 'ratio': frac(numer, denom), 'exp10': exp10}


# noinspection PyShadowingNames
def val(value,
        conv=conv(),
        units=raw_UnitArray([]),
        display_units=None):
    return raw_WithUnit(
        value,
        conv,
        units,
        units if display_units is None else display_units)


def test_auto_create():
    with pytest.raises(KeyError):
        UnitDatabase(auto_create_units=False).parse_unit_formula('tests')

    db = UnitDatabase(auto_create_units=True)
    u = db.parse_unit_formula('tests')
    assert 5 == 5 * u / u

def test_add_root_unit():
    db = UnitDatabase(auto_create_units=False)
    db.add_root_unit('cats')

    # Root unit is simple.
    c = db.get_unit('cats')
    assert c.base_units == raw_UnitArray([('cats', 1, 1)])
    assert c.display_units == raw_UnitArray([('cats', 1, 1)])
    assert c.numer == 1
    assert c.denom == 1
    assert c.exp10 == 0
    assert c.value == 1

    # No dups.
    with pytest.raises(RuntimeError):
        db.add_root_unit('cats')

def test_add_base_unit_with_prefixes():
    db = UnitDatabase(auto_create_units=False)
    db.add_base_unit_data(
        BaseUnitData('b', 'base', True),
        [
            PrefixData('p_', 'pre_', 1),
            PrefixData('q_', 'qu_', 2),
        ])

    # Long form *is* short form.
    assert db.get_unit('base') is db.get_unit('b')
    assert db.get_unit('pre_base') is db.get_unit('p_b')
    assert db.get_unit('qu_base') is db.get_unit('q_b')

    # Root unit is simple.
    u = db.get_unit('b')
    assert u.base_units == raw_UnitArray([('b', 1, 1)])
    assert u.display_units == raw_UnitArray([('b', 1, 1)])
    assert u.numer == 1
    assert u.denom == 1
    assert u.exp10 == 0
    assert u.value == 1

    # Prefixes do scaling.
    assert db.get_unit('p_b') == u * 10
    assert db.get_unit('q_b') == u * 100

    # No mixing long prefixes with short units or vice versa.
    with pytest.raises(KeyError):
        db.get_unit('p_base')
    with pytest.raises(KeyError):
        db.get_unit('pre_b')

def test_add_base_unit_without_prefixes():
    db = UnitDatabase(auto_create_units=False)
    db.add_base_unit_data(
        BaseUnitData('b', 'base', False),
        [
            PrefixData('p_', 'pre_', 1),
            PrefixData('q_', 'qu_', 2),
        ])

    # Long form *is* short form.
    assert db.get_unit('base') is db.get_unit('b')

    # Root unit is simple.
    u = db.get_unit('b')
    assert u.base_units == raw_UnitArray([('b', 1, 1)])
    assert u.display_units == raw_UnitArray([('b', 1, 1)])
    assert u.numer == 1
    assert u.denom == 1
    assert u.exp10 == 0
    assert u.value == 1

    # No prefixing.
    with pytest.raises(KeyError):
        db.get_unit('p_b')
    with pytest.raises(KeyError):
        db.get_unit('q_b')
    with pytest.raises(KeyError):
        db.get_unit('pre_base')
    with pytest.raises(KeyError):
        db.get_unit('qu_base')
    with pytest.raises(KeyError):
        db.get_unit('p_base')
    with pytest.raises(KeyError):
        db.get_unit('pre_b')

def test_add_derived_unit_with_prefixes():
    db = UnitDatabase(auto_create_units=False)
    with pytest.raises(KeyError):
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

    assert db.get_unit('tails') == v
    assert db.get_unit('t') == v
    assert db.get_unit('s_t') == v * 10
    assert db.get_unit('super_tails') == v * 10
    assert db.get_unit('d_t') == v * 100
    assert db.get_unit('duper_tails') == v * 100
    with pytest.raises(KeyError):
        db.get_unit('s_tails')
    with pytest.raises(KeyError):
        db.get_unit('super_t')

def test_add_derived_unit_without_prefixes():
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

    assert db.get_unit('tails') == v
    assert db.get_unit('t') == v
    with pytest.raises(KeyError):
        db.get_unit('super_tails')
    with pytest.raises(KeyError):
        db.get_unit('s_t')
    with pytest.raises(KeyError):
        db.get_unit('s_tails')
    with pytest.raises(KeyError):
        db.get_unit('super_t')

def test_auto_create_disabled_when_purposefully_adding_units():
    db = UnitDatabase(auto_create_units=True)

    with pytest.raises(KeyError):
        db.add_derived_unit_data(
            DerivedUnitData('d', 'der', 'missing'), [])

    with pytest.raises(KeyError):
        db.add_scaled_unit('new', 'missing')

    with pytest.raises(KeyError):
        db.add_alternate_unit_name('new', 'missing')

def test_get_unit_auto_create_override():
    db_auto = UnitDatabase(auto_create_units=True)
    db_manual = UnitDatabase(auto_create_units=False)

    u = db_auto.get_unit('missing')
    assert str(u) == 'missing'
    with pytest.raises(KeyError):
        db_manual.get_unit('missing')

    with pytest.raises(KeyError):
        db_manual.get_unit('gone', auto_create=False)
    with pytest.raises(KeyError):
        db_manual.get_unit('gone', auto_create=False)

    u = db_auto.get_unit('empty', auto_create=True)
    assert str(u) == 'empty'
    u = db_manual.get_unit('empty', auto_create=True)
    assert str(u) == 'empty'

def test_kilogram_special_case():
    db = UnitDatabase(auto_create_units=False)
    db.add_base_unit_data(BaseUnitData('kg', 'kilogram'), SI_PREFIXES)
    assert db.get_unit('g').base_units == raw_UnitArray([('kg', 1, 1)])
    assert db.get_unit('g') * 1000 == db.get_unit('kg')
    assert db.get_unit('kg') * 1000 == db.get_unit('Mg')

def test_parse_unit_formula():
    db = UnitDatabase(auto_create_units=False)
    db.add_root_unit('cats')
    db.add_root_unit('dogs')
    db.add_root_unit('mice')
    cats = db.get_unit('cats')
    dogs = db.get_unit('dogs')
    mice = db.get_unit('mice')

    with pytest.raises(ParseException):
        db.parse_unit_formula('cats^dogs')

    assert db.parse_unit_formula('cats') == cats
    assert db.parse_unit_formula('cats^2') == cats**2
    assert db.parse_unit_formula('cats^-2') == cats**-2
    assert db.parse_unit_formula('cats*cats') == cats**2
    assert db.parse_unit_formula('cats*dogs') == cats * dogs
    assert db.parse_unit_formula('cats/dogs') == cats / dogs
    assert db.parse_unit_formula('cats/dogs^2') == cats / dogs**2
    assert db.parse_unit_formula('cats/dogs*mice') == (cats / dogs) * mice

def test_parse_float_formula():
    db = UnitDatabase(auto_create_units=False)
    db.add_root_unit('J')
    db.add_root_unit('s')
    db.add_root_unit('C')
    J = db.get_unit('J')
    s = db.get_unit('s')
    C = db.get_unit('C')

    assert (db.parse_unit_formula('2.06783276917e-15 J*s/C') ==
            2.06783276917e-15 * J * s / C)

def test_is_consistent_with_database():
    db = UnitDatabase(auto_create_units=True)

    # Empty.
    assert db.is_value_consistent_with_database(val(5))

    # Missing.
    assert not db.is_value_consistent_with_database(val(
        6,
        units=unit('theorems')))

    # Present.
    db.add_root_unit('theorems')
    assert db.is_value_consistent_with_database(val(
        6,
        units=unit('theorems')))

    # Self-contradictory conversion.
    assert not db.is_value_consistent_with_database(val(
        6,
        conv=conv(3),
        units=unit('theorems')))

    # Inconsistent conversion.
    db.add_scaled_unit('kilo_theorems', 'theorems', exp10=3)
    assert not db.is_value_consistent_with_database(val(
        6,
        units=unit('theorems'),
        display_units=unit('kilo_theorems')))

    # Consistent conversion.
    assert db.is_value_consistent_with_database(val(
        6,
        conv=conv(numer=1000),
        units=unit('theorems'),
        display_units=unit('kilo_theorems')))
    assert db.is_value_consistent_with_database(val(
        6,
        conv=conv(exp10=3),
        units=unit('theorems'),
        display_units=unit('kilo_theorems')))

    # Disagreement over what the root unit is.
    assert not db.is_value_consistent_with_database(val(
        6,
        conv=conv(exp10=-3),
        units=unit('kilo_theorems'),
        display_units=unit('theorems')))

    # Nearly consistent conversion.
    db.add_scaled_unit('factoids', 'theorems', 3.141, 3, 5, -7)
    assert db.is_value_consistent_with_database(val(
        10,
        conv(3.141, 3, 5, -7),
        unit('theorems'),
        unit('factoids')))
    assert db.is_value_consistent_with_database(val(
        10,
        conv(3.14100000000001, 3, 5, -7),
        unit('theorems'),
        unit('factoids')))

    # Combinations.
    assert not db.is_value_consistent_with_database(val(
        10,
        conv(-3),
        unit('theorems') * unit('theorems'),
        unit('kilo_theorems')))
    assert not db.is_value_consistent_with_database(val(
        10,
        conv(-3),
        unit('theorems') * unit('theorems'),
        unit('kilo_theorems') * unit('factoids')))
    assert db.is_value_consistent_with_database(val(
        10,
        conv(3.141, 3, 5, -4),
        unit('theorems') * unit('theorems'),
        unit('kilo_theorems') * unit('factoids')))

    # Exponents.
    assert not db.is_value_consistent_with_database(val(
        10,
        conv(exp10=-3),
        unit('theorems')**2,
        unit('kilo_theorems')**2))
    assert db.is_value_consistent_with_database(val(
        10,
        conv(exp10=6),
        unit('theorems')**2,
        unit('kilo_theorems')**2))
    assert db.is_value_consistent_with_database(val(
        10,
        conv(exp10=1),
        unit('theorems')**(1 / 3.0),
        unit('kilo_theorems')**(1 / 3.0)))
