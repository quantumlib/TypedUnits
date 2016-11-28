import pickle

import pytest
# noinspection PyProtectedMember
from pyfu._all_cythonized import raw_UnitArray, UnitArray

du = raw_UnitArray([])


def test_construction_versus_items():
    empty = UnitArray()
    assert len(empty) == 0
    assert list(empty) == []

    singleton = UnitArray('arbitrary')
    assert len(singleton) == 1
    assert singleton[0] == ('arbitrary', 1, 1)
    assert list(singleton) == [('arbitrary', 1, 1)]

    with pytest.raises(TypeError):
        raw_UnitArray(1)
    with pytest.raises(TypeError):
        raw_UnitArray((2, 'a', 'c'))

    raw0 = raw_UnitArray([])
    assert len(raw0) == 0
    assert list(raw0) == []

    raw1 = raw_UnitArray([('a', 2, 3)])
    assert len(raw1) == 1
    assert raw1[0] == ('a', 2, 3)
    assert list(raw1) == [('a', 2, 3)]

    raw2 = raw_UnitArray([('a', 3, 7), ('b', 6, 15)])
    assert len(raw2) == 2
    assert raw2[0] == ('a', 3, 7)
    assert raw2[1] == ('b', 2, 5)
    assert list(raw2) == [('a', 3, 7), ('b', 2, 5)]

def test_repr():
    assert repr(du) == 'raw_UnitArray([])'
    assert repr(UnitArray('a')), "raw_UnitArray([('a', 1 == 1)])"

    assert repr(raw_UnitArray([])) == "raw_UnitArray([])"
    assert repr(raw_UnitArray([('a', 2, 3)])) == "raw_UnitArray([('a', 2, 3)])"
    assert (repr(raw_UnitArray([('a', 2, 3), ('b', -5, 7)])) ==
            "raw_UnitArray([('a', 2, 3), ('b', -5, 7)])")

def test_str():
    assert str(du) == ''
    assert str(UnitArray('a')) == 'a'

    assert str(raw_UnitArray([('b', -1, 1)])) == '1/b'
    assert str(raw_UnitArray([('a', 2, 3), ('b', -5, 7)])) == 'a^(2/3)/b^(5/7)'
    assert str(raw_UnitArray([('a', 1, 1),
                              ('b', -1, 1),
                              ('c', 1, 1),
                              ('d', -1, 1)])) == 'a*c/b/d'
    assert str(raw_UnitArray([('a', 2, 1),
                              ('b', -1, 2),
                              ('c', 1, 1),
                              ('d', -1, 1)])) == 'a^2*c/b^(1/2)/d'

def test_equality():
    equivalence_groups = [
        [0],
        [[]],
        [""],
        ["other types"],
        [list],
        [None],

        [du, UnitArray(), raw_UnitArray([])],
        [UnitArray('a'), raw_UnitArray([('a', 1, 1)])],
        [raw_UnitArray([('a', 2, 1)]), raw_UnitArray([('a', 6, 3)])],
        [raw_UnitArray([('b', 2, 1)]), raw_UnitArray([('b', -6, -3)])],
        [raw_UnitArray([('b', -2, 1)]), raw_UnitArray([('b', 2, -1)])],
        [raw_UnitArray([('a', 2, 1), ('a', 2, 1)])],
        [raw_UnitArray([('a', 2, 1), ('b', 2, 1)])],
        [raw_UnitArray([('b', 2, 1), ('a', 2, 1)])],
        [raw_UnitArray([('a', 1, 1), ('b', 1, 1), ('c', 1, 1)])] * 2,
    ]
    for g1 in equivalence_groups:
        for g2 in equivalence_groups:
            for e1 in g1:
                for e2 in g2:
                    if g1 is g2:
                        assert e1 == e2
                    else:
                        assert e1 != e2

def test_multiplicative_identity():
    various = [
        UnitArray('a'),
        raw_UnitArray([('a', 2, 3), ('b', 1, 1)]),
        du
    ]
    for e in various:
        assert du * e == e
        assert e * du == e
        assert e / du == e

def test_multiplication():
    assert UnitArray('a') * UnitArray('b') == raw_UnitArray([('a', 1, 1),
                                                             ('b', 1, 1)])
    assert UnitArray('b') * UnitArray('a') == raw_UnitArray([('a', 1, 1),
                                                             ('b', 1, 1)])
    assert (raw_UnitArray([('a', 2, 7)]) * raw_UnitArray([('a', 3, 5)]) ==
            raw_UnitArray([('a', 31, 35)]))
    assert (raw_UnitArray([('a', 1, 1), ('b', 3, 5)]) * UnitArray('b') ==
            raw_UnitArray([('a', 1, 1), ('b', 8, 5)]))
    assert (raw_UnitArray([('b', -3, 5), ('a', 1, 1)]) * UnitArray('b') ==
            raw_UnitArray([('b', 2, 5), ('a', 1, 1)]))

def test_division():
    assert du / UnitArray('b') == raw_UnitArray([('b', -1, 1)])
    assert UnitArray('a') / UnitArray('b') == raw_UnitArray([('a', 1, 1),
                                                             ('b', -1, 1)])
    assert UnitArray('b') / UnitArray('a') == raw_UnitArray([('a', -1, 1),
                                                             ('b', 1, 1)])
    assert (raw_UnitArray([('a', 2, 7)]) / raw_UnitArray([('a', 3, 5)]) ==
            raw_UnitArray([('a', -11, 35)]))
    assert (raw_UnitArray([('a', 1, 1), ('b', 3, 5)]) / UnitArray('b') ==
            raw_UnitArray([('a', 1, 1), ('b', -2, 5)]))
    assert (raw_UnitArray([('b', -3, 5), ('a', 1, 1)]) / UnitArray('b') ==
            raw_UnitArray([('b', -8, 5), ('a', 1, 1)]))

def test_pow():
    assert du**2 == du
    assert UnitArray('a')**0 == du
    assert UnitArray('a')**2 == raw_UnitArray([('a', 2, 1)])
    assert UnitArray('a')**-1 == raw_UnitArray([('a', -1, 1)])
    assert UnitArray('a')**(1.0 / 3) == raw_UnitArray([('a', 1, 3)])
    assert UnitArray('a')**(7.0 / 12) == raw_UnitArray([('a', 7, 12)])
    assert UnitArray('a')**(1.0 / 12) == raw_UnitArray([('a', 1, 12)])

    assert (raw_UnitArray([('a', 2, 3), ('b', -5, 7)])**(37.0 / 12) ==
            raw_UnitArray([('a', 37, 18), ('b', -5 * 37, 7 * 12)]))

def test_pickling():
    examples = [
        du,
        raw_UnitArray([('a', 2, 7)]),
        raw_UnitArray([('a', 2, 7), ('b', 1, 3)])
    ]
    for e in examples:
        assert e == pickle.loads(pickle.dumps(e))
