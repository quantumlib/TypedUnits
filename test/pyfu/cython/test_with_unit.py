import copy
import numpy as np
import pickle
import unittest
from pyfu import WithUnit, DimensionlessUnit as du, UnitMismatchError
from pyfu.__all_cythonized import raw_WithUnit, raw_UnitArray

s = raw_UnitArray([('s', 1, 1)])
h = raw_UnitArray([('s', 3600, 1)])  # Note: is s**3600, not 3600 seconds.
m = raw_UnitArray([('m', 1, 1)])
mps = raw_UnitArray([('m', 1, 1), ('s', -1, 1)])
kph = raw_UnitArray([('m', 1000, 1), ('s', -1, 3600)])


def wrap(value, numer=1, denom=1, exp10=0, units=du, display_units=None):
    if display_units is None:
        display_units = units
    return raw_WithUnit(value, numer, denom, exp10, units, display_units)


class WithUnitTests(unittest.TestCase):
    def assertNumpyArrayEqual(self, a, b):
        # noinspection PyTypeChecker
        if len(a) != len(b) or not np.all(a == b):
            msg = 'not np.all(%s == %s)' % (repr(a), repr(b))
            msg = self._formatMessage(None, msg)
            raise self.failureException(msg)

    def testRawVersusProperties(self):
        x = wrap(1, 2, 3, 4, mps, kph)
        self.assertEqual(x.value, 1)
        self.assertEqual(x.numer, 2)
        self.assertEqual(x.denom, 3)
        self.assertEqual(x.exp10, 4)
        self.assertEqual(x.base_units, mps)
        self.assertEqual(x.display_units, kph)

        x = wrap(-2.5, 6, 15, -2, s, h)
        self.assertEqual(x.value, -2.5)
        self.assertEqual(x.numer, 2)
        self.assertEqual(x.denom, 5)
        self.assertEqual(x.exp10, -2)
        self.assertEqual(x.base_units, s)
        self.assertEqual(x.display_units, h)

    def testEquality(self):
        equivalence_groups = [
            [""],
            ["other types"],
            [WithUnitTests],
            [None],
            [du],

            [0, wrap(0)],
            [1 + 2j, wrap(1 + 2j)],

            [
                wrap(1, 2, 3, 4, mps, kph),
                wrap(1, 2, 3, 4, mps, mps),
                wrap(10, 2, 3, 3, mps, mps),
                wrap(1, 20, 3, 3, mps, mps),
                wrap(1, 2, 30, 5, mps, mps),
            ],
            [wrap(1, 20, 3, 3, kph, mps)],
            [wrap(1, 20, 3 * 3600, 3, kph, mps)],
            [wrap(99, 2, 3, 4, mps, kph)],
            [wrap(1, 99, 3, 4, mps, kph)],
            [wrap(1, 2, 99, 4, mps, kph)],
            [wrap(1, 2, 3, 99, mps, kph)],
            [wrap(1, 2, 3, 0, mps, kph)],
            [wrap(1, 2, 3, -99, mps, kph)],
            [wrap(1, 2, 3, 10, kph, kph)],
            [wrap(1, 2, 3, 4, s, h), wrap(1, 2, 3, 4, s, s)],
            [wrap(1, 2, 3, 4, h, h)],
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
        self.assertNumpyArrayEqual(5 == wrap([], 2, 3, 5, mps, kph),
                                   [])
        self.assertNumpyArrayEqual([] == wrap([], 2, 3, 5, mps, kph),
                                   [])
        self.assertNumpyArrayEqual(
            [1, 2] == wrap([1, 2], 2, 3, 5, mps, kph),
            [False, False])
        self.assertNumpyArrayEqual(
            [1, 2] == wrap([1, 2]),
            np.array([True, True]))
        self.assertNumpyArrayEqual(
            [3, 2] == wrap([1, 2]),
            np.array([False, True]))
        self.assertNumpyArrayEqual(
            [3, 2] == wrap([1, 2], numer=3),
            np.array([True, False]))
        self.assertNumpyArrayEqual(
            3 == wrap([1, 2], numer=3),
            np.array([True, False]))
        self.assertNumpyArrayEqual(
            6 == wrap([1, 2], numer=3),
            np.array([False, True]))

    def testFloat(self):
        with self.assertRaises(TypeError):
            float(wrap(1, 2, 3, 4, mps, kph))
        with self.assertRaises(TypeError):
            float(wrap(1j, 2, 3, 4))
        with self.assertRaises(TypeError):
            float(wrap(np.array([1, 2]), 2, 3, 4))

        u = float(wrap(5))
        self.assertIsInstance(u, float)
        self.assertEqual(u, 5)

        self.assertEqual(float(
            wrap(1, 2, 3, 4)),
            2.0 / 3.0 * 1E4)

    def testComplex(self):
        with self.assertRaises(TypeError):
            complex(wrap(1j, 2, 3, 4, mps, kph))
        with self.assertRaises(TypeError):
            complex(wrap(np.array([1, 2]), 2, 3, 4))

        u = complex(wrap(5))
        self.assertIsInstance(u, complex)
        self.assertEqual(u, 5)

        self.assertEqual(complex(
            wrap(1, 2, 3, 4)),
            2.0 / 3.0 * 1E4)
        self.assertEqual(complex(
            wrap(1 + 3j, 2, 5, 4)),
            (1+3j) * 2.0 / 5.0 * 1E4)

    def testArray(self):
        with self.assertRaises(TypeError):
            np.array(wrap(np.array([1, 2]), 2, 3, 4, mps, kph))

        u = np.array(wrap(np.array([2, 3])))
        self.assertIsInstance(u, np.ndarray)
        self.assertNumpyArrayEqual([2, 3], u)

        self.assertNumpyArrayEqual(np.array(wrap(
            np.array([2, 3]), 2, 3, 4)),
            np.array([4.0 / 3.0 * 1E4, 2E4]))

    def testCopy(self):
        a = wrap(1, 2, 3, 4, mps, kph)
        self.assertIs(a, copy.copy(a))
        self.assertIs(a, copy.deepcopy(a))

        b = wrap(1 + 2j, 2, 3, 4, mps, kph)
        self.assertIs(b, copy.copy(b))
        self.assertIs(b, copy.deepcopy(b))

        c = wrap(np.array([10, 11]), 2, 3, 4, mps, kph)
        self.assertIsNot(c, copy.copy(c))
        self.assertIsNot(c, copy.deepcopy(c))
        self.assertNumpyArrayEqual(c, copy.copy(c))
        self.assertNumpyArrayEqual(c, copy.deepcopy(c))

        c2 = copy.copy(c)
        c[1] = wrap(52, 1, 1, 0, mps, mps)
        # noinspection PyTypeChecker
        self.assertFalse(np.all(c == c2))

    def testPickle(self):
        examples = [
            WithUnit(1),
            wrap(2, 3, 5, 7, mps, kph)
        ]
        for e in examples:
            self.assertEqual(e, pickle.loads(pickle.dumps(e)))

    def testMultiplication(self):
        self.assertEqual(
            wrap(2, 3, 5, 7, mps, kph) * 5,
            wrap(10, 3, 5, 7, mps, kph))
        self.assertEqual(
            5 * wrap(2, 3, 5, 7, mps, kph),
            wrap(10, 3, 5, 7, mps, kph))
        self.assertEqual(wrap(2, 3, 5, 7, mps, kph) *
                         wrap(11, 13, 17, 19, s, h),
                         wrap(22, 39, 85, 26, m, m))

    def testNumpyMethod_isFinite(self):
        v = wrap(np.array([2, 3, -2, float('nan'), float('inf')]), 2, 3, 4)

        self.assertNumpyArrayEqual(
            np.isfinite(v),
            [True, True, True, False, False])

    def testGetItem_scaling(self):
        u = wrap(1, exp10=1)
        v = wrap(2, 3, 5, 7, mps, kph)

        # Wrong kinds of index (unit array, slice).
        with self.assertRaises(TypeError):
            _ = wrap(u[mps])
        with self.assertRaises(TypeError):
            _ = wrap(u[1:2])

        # Safety against dimensionless unit ambiguity.
        with self.assertRaises(TypeError):
            _ = u[1.0]
        with self.assertRaises(TypeError):
            _ = wrap(u[1])
        with self.assertRaises(TypeError):
            _ = u[v / v]

        # Wrong unit.
        with self.assertRaises(UnitMismatchError):
            _ = u[v]

        self.assertEquals(u[''], 10)
        self.assertEquals(v[v], 1)
        self.assertEquals(v[wrap(2, 1, 5, 7, mps, kph)], 3)
        self.assertEquals(v[wrap(2, 3, 1, 7, mps, kph)], 0.2)
        self.assertEquals(v[wrap(2, 3, 5, 0, mps, kph)], 10 ** 7)
        self.assertEquals(wrap(2, 3, 1, 7, mps, kph)[v], 5)

        self.assertAlmostEquals(wrap(2, 1, 5, 7, mps, kph)[v], 1 / 3.0)
        self.assertAlmostEquals(wrap(2, 3, 5, 0, mps, kph)[v], 10 ** -7)

    def testIter(self):
        a = []
        for e in wrap([1, 2, 4], 2, 3, 4, mps, kph):
            a.append(e)
        self.assertEquals(len(a), 3)
        self.assertEquals(a[0], wrap(1, 2, 3, 4, mps, kph))
        self.assertEquals(a[1], wrap(2, 2, 3, 4, mps, kph))
        self.assertEquals(a[2], wrap(4, 2, 3, 4, mps, kph))

    def testHash(self):
        d = dict()
        v = wrap(2, 1, 5, 7, mps, kph)
        w = wrap(3, 1, 5, 7, mps, kph)
        d[v] = 5
        d[w] = "b"
        self.assertEquals(d[v], 5)
        self.assertEquals(d[w], "b")

    def testStr(self):
        self.assertEquals(str(wrap(2, 3, 5, 7, s, h)), '2.0 s^3600')
        self.assertEquals(str(wrap(2, 3, 5, 7, h, s)), '2.0 s')
        self.assertEquals(str(wrap(2, units=s)), '2.0 s')
        self.assertEquals(str(wrap(2j, units=s)), '2j s')
        self.assertEquals(str(wrap(1, units=h, display_units=s)), 's')
        self.assertEquals(str(wrap([2, 3, 5], units=h, display_units=m)),
                          '[2 3 5] m')
        self.assertEquals(str(wrap([2, 3], units=h, display_units=mps)),
                          '[2 3] m/s')

if __name__ == "__main__":
    unittest.main()
