#!/usr/bin/python
import copy
import numpy as np
import pickle
import unittest
from fastunits.unitarray import WithUnit, UnitArray, DimensionlessUnit

s = UnitArray.raw([('s', 1, 1)])
h = UnitArray.raw([('s', 3600, 1)])
m = UnitArray.raw([('m', 1, 1)])
mps = UnitArray.raw([('m', 1, 1), ('s', -1, 1)])
kph = UnitArray.raw([('m', 1000, 1), ('s', -1, 3600)])


class WithUnitTests(unittest.TestCase):
    def assertNumpyArrayEqual(self, a, b):
        if len(a) != len(b) or not np.all(a == b):
            standardMsg = 'not np.all(%s == %s)' % (repr(a), repr(b))
            msg = self._formatMessage(None, standardMsg)
            raise self.failureException(msg)

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
        u = DimensionlessUnit

        equivalence_groups = [
            [""],
            ["other types"],
            [WithUnitTests],
            [None],
            [DimensionlessUnit],

            [0, WithUnit.raw(0, 1, 1, 0, u, u)],
            [1 + 2j, WithUnit.raw(1 + 2j, 1, 1, 0, u, u)],

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

    def testArrayEquality(self):
        u = DimensionlessUnit
        self.assertNumpyArrayEqual(5 == WithUnit.raw([], 2, 3, 5, mps, kph),
                                   [])
        self.assertNumpyArrayEqual([] == WithUnit.raw([], 2, 3, 5, mps, kph),
                                   [])
        self.assertNumpyArrayEqual(
            [1, 2] == WithUnit.raw([1, 2], 2, 3, 5, mps, kph),
            [False, False])
        self.assertNumpyArrayEqual(
            [1, 2] == WithUnit.raw([1, 2], 1, 1, 0, u, u),
            np.array([True, True]))
        self.assertNumpyArrayEqual(
            [3, 2] == WithUnit.raw([1, 2], 1, 1, 0, u, u),
            np.array([False, True]))
        self.assertNumpyArrayEqual(
            [3, 2] == WithUnit.raw([1, 2], 3, 1, 0, u, u),
            np.array([True, False]))
        self.assertNumpyArrayEqual(
            3 == WithUnit.raw([1, 2], 3, 1, 0, u, u),
            np.array([True, False]))
        self.assertNumpyArrayEqual(
            6 == WithUnit.raw([1, 2], 3, 1, 0, u, u),
            np.array([False, True]))

    def testFloat(self):
        self.assertRaises(lambda e: float(WithUnit.raw(1, 2, 3, 4, mps, kph)))
        self.assertRaises(lambda e: float(
            WithUnit.raw(1j, 2, 3, 4, DimensionlessUnit, DimensionlessUnit)))
        self.assertRaises(lambda e: float(WithUnit.raw(
            np.array([1, 2]), 2, 3, 4, DimensionlessUnit, DimensionlessUnit)))

        u = float(
            WithUnit.raw(5, 1, 1, 0, DimensionlessUnit, DimensionlessUnit))
        self.assertIsInstance(u, float)
        self.assertEqual(u, 5)

        self.assertEqual(float(
            WithUnit.raw(1, 2, 3, 4, DimensionlessUnit, DimensionlessUnit)),
            2.0 / 3.0 * 1E4)

    def testComplex(self):
        self.assertRaises(lambda e: complex(WithUnit.raw(1j, 2, 3, 4, mps, kph)))
        self.assertRaises(lambda e: complex(WithUnit.raw(
            np.array([1, 2]), 2, 3, 4, DimensionlessUnit, DimensionlessUnit)))

        u = complex(
            WithUnit.raw(5, 1, 1, 0, DimensionlessUnit, DimensionlessUnit))
        self.assertIsInstance(u, complex)
        self.assertEqual(u, 5)

        self.assertEqual(complex(
            WithUnit.raw(1, 2, 3, 4, DimensionlessUnit, DimensionlessUnit)),
            2.0 / 3.0 * 1E4)
        self.assertEqual(complex(
            WithUnit.raw(1+3j, 2, 5, 4, DimensionlessUnit, DimensionlessUnit)),
            (1+3j) * 2.0 / 5.0 * 1E4)

    def testArray(self):
        self.assertRaises(lambda e: np.array(
            WithUnit.raw(np.array([1, 2]), 2, 3, 4, mps, kph)))
        self.assertRaises(lambda e: np.array(
            WithUnit.raw(1, 2, 3, 4, DimensionlessUnit, DimensionlessUnit)))

        u = np.array(WithUnit.raw(
            np.array([2, 3]), 1, 1, 0, DimensionlessUnit, DimensionlessUnit))
        self.assertIsInstance(u, np.ndarray)
        self.assertNumpyArrayEqual([2, 3], u)

        self.assertNumpyArrayEqual(np.array(WithUnit.raw(
            np.array([2, 3]), 2, 3, 4, DimensionlessUnit, DimensionlessUnit)),
            np.array([4.0 / 3.0 * 1E4, 2E4]))

    def testCopy(self):
        a = WithUnit.raw(1, 2, 3, 4, mps, kph)
        self.assertIs(a, copy.copy(a))
        self.assertIs(a, copy.deepcopy(a))

        b = WithUnit.raw(1+2j, 2, 3, 4, mps, kph)
        self.assertIs(b, copy.copy(b))
        self.assertIs(b, copy.deepcopy(b))

        c = WithUnit.raw(np.array([10, 11]), 2, 3, 4, mps, kph)
        self.assertIsNot(c, copy.copy(c))
        self.assertIsNot(c, copy.deepcopy(c))
        self.assertTrue(np.all(c == copy.copy(c)))
        self.assertTrue(np.all(c == copy.deepcopy(c)))

        c2 = copy.copy(c)
        c[1] = WithUnit.raw(52, 1, 1, 0, mps, mps)
        self.assertFalse(np.all(c == c2))

    def testPickle(self):
        examples = [
            WithUnit(1),
            WithUnit.raw(2, 3, 5, 7, mps, kph)
        ]
        for e in examples:
            self.assertEqual(e, pickle.loads(pickle.dumps(e)))

    def testMultiplication(self):
        self.assertEqual(
            WithUnit.raw(2, 3, 5, 7, mps, kph) * 5,
            WithUnit.raw(10, 3, 5, 7, mps, kph))
        self.assertEqual(
            5 * WithUnit.raw(2, 3, 5, 7, mps, kph),
            WithUnit.raw(10, 3, 5, 7, mps, kph))
        self.assertEqual(WithUnit.raw(2, 3, 5, 7, mps, kph) *
                         WithUnit.raw(11, 13, 17, 19, s, h),
                         WithUnit.raw(22, 39, 85, 26, m, m))

    def testNumpyMethod_isFinite(self):
        val = WithUnit.raw(
            np.array([2, 3, -2, float('nan'), float('inf')]),
            2, 3, 4, DimensionlessUnit, DimensionlessUnit)

        self.assertNumpyArrayEqual(
            np.isfinite(val),
            [True, True, True, False, False])

if __name__ == "__main__":
    unittest.main()
