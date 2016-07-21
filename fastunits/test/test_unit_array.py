#!/usr/bin/python
import unittest
from fastunits.unitarray import UnitArray, DimensionlessUnit

class UnitsArrayTests(unittest.TestCase):
    def testConstructionVersusItems(self):
        empty = UnitArray()
        self.assertEqual(len(empty), 0)
        self.assertEqual(list(empty), [])

        singleton = UnitArray('arbitrary')
        self.assertEqual(len(singleton), 1)
        self.assertEqual(singleton[0], ('arbitrary', 1, 1))
        self.assertEqual(list(singleton), [('arbitrary', 1, 1)])

        self.assertRaises(TypeError, lambda: UnitArray.raw(1))
        self.assertRaises(TypeError, lambda: UnitArray.raw((2, 'a', 'c')))

        raw0 = UnitArray.raw([])
        self.assertEqual(len(raw0), 0)
        self.assertEqual(list(raw0), [])

        raw1 = UnitArray.raw([('a', 2, 3)])
        self.assertEqual(len(raw1), 1)
        self.assertEqual(raw1[0], ('a', 2, 3))
        self.assertEqual(list(raw1), [('a', 2, 3)])

        raw2 = UnitArray.raw([('a', 3, 7), ('b', 6, 15)])
        self.assertEqual(len(raw2), 2)
        self.assertEqual(raw2[0], ('a', 3, 7))
        self.assertEqual(raw2[1], ('b', 2, 5))
        self.assertEqual(list(raw2), [('a', 3, 7), ('b', 2, 5)])

    def testRepr(self):
        self.assertEqual(repr(DimensionlessUnit), 'UnitArray.raw([])')
        self.assertEqual(repr(UnitArray('a')), "UnitArray.raw([('a', 1, 1)])")

        self.assertEqual(
            repr(UnitArray.raw([])),
            "UnitArray.raw([])")
        self.assertEqual(
            repr(UnitArray.raw([('a', 2, 3)])),
            "UnitArray.raw([('a', 2, 3)])")
        self.assertEqual(
            repr(UnitArray.raw([('a', 2, 3), ('b', -5, 7)])),
            "UnitArray.raw([('a', 2, 3), ('b', -5, 7)])")

    def testStr(self):
        self.assertEqual(str(DimensionlessUnit), '')
        self.assertEqual(str(UnitArray('a')), 'a')

        self.assertEqual(
            str(UnitArray.raw([('b', -1, 1)])),
            '1/b')
        self.assertEqual(
            str(UnitArray.raw([('a', 2, 3), ('b', -5, 7)])),
            'a^(2/3)/b^(5/7)')
        self.assertEqual(
            str(UnitArray.raw([
                ('a', 1, 1), ('b', -1, 1), ('c', 1, 1), ('d', -1, 1)])),
            'a*c/b/d')
        self.assertEqual(
            str(UnitArray.raw([
                ('a', 2, 1), ('b', -1, 2), ('c', 1, 1), ('d', -1, 1)])),
            'a^2*c/b^(1/2)/d')

    def testEquality(self):
        equivalence_groups = [
            [0],
            [[]],
            [""],
            ["other types"],
            [UnitsArrayTests],
            [None],

            [DimensionlessUnit, UnitArray(), UnitArray.raw([])],
            [UnitArray('a'), UnitArray.raw([('a', 1, 1)])],
            [UnitArray.raw([('a', 2, 1)]), UnitArray.raw([('a', 6, 3)])],
            [UnitArray.raw([('b', 2, 1)]), UnitArray.raw([('b', -6, -3)])],
            [UnitArray.raw([('b', -2, 1)]), UnitArray.raw([('b', 2, -1)])],
            [UnitArray.raw([('a', 2, 1), ('a', 2, 1)])],
            [UnitArray.raw([('a', 2, 1), ('b', 2, 1)])],
            [UnitArray.raw([('b', 2, 1), ('a', 2, 1)])],
            [UnitArray.raw([('a', 1, 1), ('b', 1, 1), ('c', 1, 1)])]*2,
        ]
        for g1 in equivalence_groups:
            for g2 in equivalence_groups:
                for e1 in g1:
                    for e2 in g2:
                        match = g1 is g2
                        if match:
                            self.assertEqual(e1, e2)
                        else:
                            self.assertNotEqual(e1, e2)
                        self.assertEqual(e1 == e2, match)
                        self.assertEqual(e1 != e2, not match)

    def testMultiplicativeIdentity(self):
        various = [
            UnitArray('a'),
            UnitArray.raw([('a', 2, 3), ('b', 1, 1)]),
            DimensionlessUnit
        ]
        for e in various:
            self.assertEqual(DimensionlessUnit * e, e)
            self.assertEqual(e * DimensionlessUnit, e)
            self.assertEqual(e / DimensionlessUnit, e)

    def testMultiplication(self):
        self.assertEqual(UnitArray('a') * UnitArray('b'),
                         UnitArray.raw([('a', 1, 1), ('b', 1, 1)]))
        self.assertEqual(UnitArray('b') * UnitArray('a'),
                         UnitArray.raw([('a', 1, 1), ('b', 1, 1)]))
        self.assertEqual(
            UnitArray.raw([('a', 2, 7)]) * UnitArray.raw([('a', 3, 5)]),
            UnitArray.raw([('a', 31, 35)]))
        self.assertEqual(
            UnitArray.raw([('a', 1, 1), ('b', 3, 5)]) * UnitArray('b'),
            UnitArray.raw([('a', 1, 1), ('b', 8, 5)]))
        self.assertEqual(
            UnitArray.raw([('b', -3, 5), ('a', 1, 1)]) * UnitArray('b'),
            UnitArray.raw([('b', 2, 5), ('a', 1, 1)]))

    def testDivision(self):
        self.assertEqual(DimensionlessUnit / UnitArray('b'),
                         UnitArray.raw([('b', -1, 1)]))
        self.assertEqual(UnitArray('a') / UnitArray('b'),
                         UnitArray.raw([('a', 1, 1), ('b', -1, 1)]))
        self.assertEqual(UnitArray('b') / UnitArray('a'),
                         UnitArray.raw([('a', -1, 1), ('b', 1, 1)]))
        self.assertEqual(
            UnitArray.raw([('a', 2, 7)]) / UnitArray.raw([('a', 3, 5)]),
            UnitArray.raw([('a', -11, 35)]))
        self.assertEqual(
            UnitArray.raw([('a', 1, 1), ('b', 3, 5)]) / UnitArray('b'),
            UnitArray.raw([('a', 1, 1), ('b', -2, 5)]))
        self.assertEqual(
            UnitArray.raw([('b', -3, 5), ('a', 1, 1)]) / UnitArray('b'),
            UnitArray.raw([('b', -8, 5), ('a', 1, 1)]))

    def testPow(self):
        self.assertEqual(DimensionlessUnit**2, DimensionlessUnit)
        self.assertEqual(UnitArray('a')**0, DimensionlessUnit)
        self.assertEqual(UnitArray('a')**2, UnitArray.raw([('a', 2, 1)]))
        self.assertEqual(UnitArray('a')**-1, UnitArray.raw([('a', -1, 1)]))
        self.assertEqual(UnitArray('a')**(1.0/3), UnitArray.raw([('a', 1, 3)]))
        self.assertEqual(UnitArray('a')**(7.0/12), UnitArray.raw([('a', 7, 12)]))
        self.assertEqual(UnitArray('a')**(1.0/12), UnitArray.raw([('a', 1, 12)]))

        self.assertEqual(UnitArray.raw([('a', 2, 3), ('b', -5, 7)])**(37.0/12),
                         UnitArray.raw([('a', 37, 18), ('b', -5*37, 7*12)]))

if __name__ == "__main__":
    unittest.main()

