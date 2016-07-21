#!/usr/bin/python
import unittest
from fastunits.unitarray import WithUnit, UnitArray, DimensionlessUnit

s = UnitArray.raw([('s', 1, 1)])
h = UnitArray.raw([('s', 3600, 1)])
mps = UnitArray.raw([('m', 1, 1), ('s', 1, 1)])
kph = UnitArray.raw([('m', 1000, 1), ('s', 1, 3600)])

class WithUnitTests(unittest.TestCase):
    def testRawVersusProperties(self):
        x = WithUnit.raw(1, 2, 3, 4, mps, kph)
        self.assertEqual(x.value, 1)
        self.assertEqual(x.numer, 2)
        self.assertEqual(x.denom, 3)
        self.assertEqual(x.exp10, 4)
        self.assertEqual(x.base_units, mps)
        self.assertEqual(x.display_units, kph)

        x = WithUnit.raw(-2.5, 6, 15, -2, s, h)
        self.assertEqual(x.value, -2.5)
        self.assertEqual(x.numer, 2)
        self.assertEqual(x.denom, 5)
        self.assertEqual(x.exp10, -2)
        self.assertEqual(x.base_units, s)
        self.assertEqual(x.display_units, h)

    def testEquality(self):
        equivalence_groups = [
            [0, WithUnit.raw(0, 1, 1, 0, DimensionlessUnit, DimensionlessUnit)],
            [[]],
            [""],
            ["other types"],
            [WithUnitTests],
            [None],
            [DimensionlessUnit],
            [1 + 2j],

            [
                WithUnit.raw(1, 2, 3, 4, mps, kph),
                WithUnit.raw(1, 2, 3, 4, mps, mps),
                WithUnit.raw(10, 2, 3, 3, mps, mps),
                WithUnit.raw(1, 20, 3, 3, mps, mps),
                WithUnit.raw(1, 2, 30, 5, mps, mps),
            ],
            [WithUnit.raw(1, 20, 3, 3, kph, mps)],
            [WithUnit.raw(1, 20, 3*3600, 3, kph, mps)],
            [WithUnit.raw(99, 2, 3, 4, mps, kph)],
            [WithUnit.raw(1, 99, 3, 4, mps, kph)],
            [WithUnit.raw(1, 2, 99, 4, mps, kph)],
            [WithUnit.raw(1, 2, 3, 99, mps, kph)],
            [WithUnit.raw(1, 2, 3, 0, mps, kph)],
            [WithUnit.raw(1, 2, 3, -99, mps, kph)],
            [WithUnit.raw(1, 2, 3, 10, kph, kph)],
            [WithUnit.raw(1, 2, 3, 4, s, h), WithUnit.raw(1, 2, 3, 4, s, s)],
            [WithUnit.raw(1, 2, 3, 4, h, h)],
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

if __name__ == "__main__":
    unittest.main()
