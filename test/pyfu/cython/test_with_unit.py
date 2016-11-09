import copy
import numpy as np
import pickle
import unittest
from pyfu import UnitMismatchError, ValueArray, Value, Complex
from pyfu._all_cythonized import raw_WithUnit, raw_UnitArray

dimensionless = raw_UnitArray([])
s = raw_UnitArray([('s', 1, 1)])
rad = raw_UnitArray([('rad', 1, 1)])
h = raw_UnitArray([('s', 3600, 1)])  # Note: is s**3600, not 3600 seconds.
m = raw_UnitArray([('m', 1, 1)])
kg = raw_UnitArray([('kg', 1, 1)])
mps = raw_UnitArray([('m', 1, 1), ('s', -1, 1)])
kph = raw_UnitArray([('m', 1000, 1), ('s', -1, 3600)])


def frac(numer=1, denom=1):
    return {'numer': numer, 'denom': denom}


def conv(factor=1.0, numer=1, denom=1, exp10=0):
    return {'factor': factor, 'ratio': frac(numer, denom), 'exp10': exp10}


# noinspection PyShadowingNames
def val(value,
        conv=conv(),
        units=dimensionless,
        display_units=None):
    return raw_WithUnit(
        value,
        conv,
        units,
        units if display_units is None else display_units)


class WithUnitTests(unittest.TestCase):
    def assertDeepEqual(self, a, b):
        if isinstance(a, ValueArray):
            self.assertNumpyArrayEqual(a, b)
        else:
            self.assertEqual(a, b)
        self.assertEqual(a.numer, b.numer)
        self.assertEqual(a.denom, b.denom)
        self.assertEqual(a.factor, b.factor)
        self.assertEqual(a.exp10, b.exp10)
        if isinstance(a, ValueArray):
            self.assertNumpyArrayEqual(a.value, b.value)
        else:
            self.assertEqual(a.value, b.value)

    def assertNumpyArrayEqual(self, a, b):
        if len(a) != len(b):
            msg = 'len(%s) != len(%s)' % (repr(a), repr(b))
            msg = self._formatMessage(None, msg)
            raise self.failureException(msg)

        # noinspection PyTypeChecker
        if not np.all(a == b):
            msg = 'not np.all(%s == %s)' % (repr(a), repr(b))
            msg = self._formatMessage(None, msg)
            raise self.failureException(msg)

    def assertNotNumpyArrayEqual(self, a, b):
        if len(a) != len(b):
            msg = 'len(%s) != len(%s)' % (repr(a), repr(b))
            msg = self._formatMessage(None, msg)
            raise self.failureException(msg)

        # noinspection PyTypeChecker
        if np.all(a == b):
            msg = 'not not np.all(%s == %s)' % (repr(a), repr(b))
            msg = self._formatMessage(None, msg)
            raise self.failureException(msg)

    def testRawVersusProperties(self):
        x = val(2, conv(factor=3, numer=4, denom=5, exp10=6), mps, kph)
        self.assertEqual(x.value, 2)
        self.assertEqual(x.factor, 3)
        self.assertEqual(x.numer, 4)
        self.assertEqual(x.denom, 5)
        self.assertEqual(x.exp10, 6)
        self.assertEqual(x.base_units, mps)
        self.assertEqual(x.display_units, kph)

        self.assertIsInstance(val(2), Value)
        self.assertIsInstance(val(2j), Complex)
        self.assertIsInstance(val([2]), ValueArray)

    def testAbs(self):
        self.assertEqual(abs(val(2)), val(2))

        # If we have a negative unit, abs is w.r.t. the derived unit.
        self.assertEqual(abs(val(-2)), val(2))
        self.assertEqual(abs(val(2, conv(-1.5))), val(-3))
        self.assertEqual(abs(val(2, conv(numer=-2))), val(-4))
        self.assertEqual(abs(val(2, conv(-1.5, numer=-2))), val(6))

    def testEquality(self):
        equivalence_groups = [
            [""],
            ["other types"],
            [WithUnitTests],
            [None],
            [dimensionless],

            # Wrapped values equal unwrapped values.
            [0, val(0)],
            [2, val(2)],
            [1 + 2j, val(1 + 2j)],
            [2.5, val(2.5)],

            # Units matter.
            [1, val(1)],
            [val(1, units=s)],
            [val(1, units=m)],

            # The display unit *text* doesn't matter to equality.
            [
                val(1, conv(factor=2, numer=5, denom=4, exp10=5), mps, kph),
                val(1, conv(factor=2, numer=5, denom=4, exp10=5), mps, s),
            ],
            [
                val(-1, units=s, display_units=m),
                val(-1, units=s, display_units=s),
            ],

            # Varying each parameter causes differences.
            [val(9, conv(factor=2, numer=5, denom=4, exp10=5), mps, kph)],
            [val(1, conv(factor=9, numer=5, denom=4, exp10=5), mps, kph)],
            [val(1, conv(factor=2, numer=9, denom=4, exp10=5), mps, kph)],
            [val(1, conv(factor=2, numer=5, denom=9, exp10=5), mps, kph)],
            [val(1, conv(factor=2, numer=5, denom=4, exp10=9), mps, kph)],
            [val(1, conv(factor=2, numer=5, denom=4, exp10=5), s, kph)],

            # You can trade between parameters.
            [
                val(10, conv(factor=2, numer=3, denom=4, exp10=5), mps, kph),
                val(1, conv(factor=20, numer=3, denom=4, exp10=5), mps, kph),
                val(1, conv(factor=2, numer=30, denom=4, exp10=5), mps, kph),
                val(1, conv(factor=2, numer=3, denom=4, exp10=6), mps, kph),
                val(1, conv(factor=2, numer=3, denom=40, exp10=7), mps, kph),
            ],

            [val(1, conv(factor=2, numer=3, denom=4, exp10=5), mps, kph)],
            [val(3, units=s, display_units=h), val(3, units=s)],
            [val(3, units=h)]
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

    def testOrdering(self):
        with self.assertRaises(UnitMismatchError):
            _ = val(1) < val(1, units=m)
            _ = val(1) > val(1, units=m)
            _ = val(1) <= val(1, units=m)
            _ = val(1) >= val(1, units=m)

        ascending_groups = [
            [val(0, units=m)],
            [val(1, units=m)],
            [val(2, units=m), val(1, conv(2), units=m, display_units=s)],
            [val(3.14, conv(2, 2, 3, 1), units=m, display_units=s)],
        ]

        for i in range(len(ascending_groups)):
            for a in ascending_groups[i]:
                for b in ascending_groups[i]:
                    self.assertLessEqual(a, b)
                    self.assertGreaterEqual(a, b)
                    self.assertFalse(a < b)
                    self.assertFalse(a > b)
                    self.assertTrue(a <= b)
                    self.assertTrue(a >= b)

                for j in range(len(ascending_groups))[i + 1:]:
                    for b in ascending_groups[j]:
                        self.assertLess(a, b)
                        self.assertGreater(b, a)
                        self.assertTrue(a < b)
                        self.assertFalse(a > b)
                        self.assertTrue(a <= b)
                        self.assertFalse(a >= b)

    def testArrayEquality(self):
        self.assertNumpyArrayEqual(5 == val([]), [])
        self.assertNumpyArrayEqual([] == val([], conv(2), mps), [])
        self.assertNumpyArrayEqual([1] == val([0]), [False])
        self.assertNumpyArrayEqual([1] == val([1]), [True])
        self.assertNumpyArrayEqual([1, 2] == val([3, 4]), [False, False])
        self.assertNumpyArrayEqual([1, 2] == val(np.array([3, 4])),
                                   [False, False])
        self.assertNumpyArrayEqual([1, 2] == val([1, 2]), [True, True])
        self.assertNumpyArrayEqual([1, 2] == val([9, 2]), [False, True])
        self.assertNumpyArrayEqual([1, 2] == val([1, 9]), [True, False])
        self.assertNumpyArrayEqual([1, 2] == val([1, 2], units=m),
                                   [False, False])
        self.assertNumpyArrayEqual(val([1, 2], units=s) == val([1, 2], units=m),
                                   [False, False])
        self.assertNumpyArrayEqual(val([1, 2], units=m) == val([1, 2], units=m),
                                   [True, True])
        self.assertNumpyArrayEqual([1, 2] == val([0.5, 1], conv(2)),
                                   [True, True])

    def testArrayOrdering(self):
        with self.assertRaises(UnitMismatchError):
            _ = val([]) < val([], units=m)
            _ = val([]) > val([], units=m)
            _ = val([]) <= val([], units=m)
            _ = val([]) >= val([], units=m)

        self.assertNumpyArrayEqual(
            val([2, 3, 4], units=m) < val([3, 3, 3], units=m),
            [True, False, False])
        self.assertNumpyArrayEqual(
            val([2, 3, 4], units=m) <= val([3, 3, 3], units=m),
            [True, True, False])
        self.assertNumpyArrayEqual(
            val([2, 3, 4], units=m) >= val([3, 3, 3], units=m),
            [False, True, True])
        self.assertNumpyArrayEqual(
            val([2, 3, 4], units=m) > val([3, 3, 3], units=m),
            [False, False, True])

    def testInt(self):
        with self.assertRaises(TypeError):
            int(val(1, units=mps))
        with self.assertRaises(TypeError):
            int(val(1j))
        with self.assertRaises(TypeError):
            int(val([1, 2]))

        u = int(val(5))
        self.assertIsInstance(u, int)
        self.assertEqual(u, 5)

        u = int(val(2.5))
        self.assertIsInstance(u, int)
        self.assertEqual(u, 2)

        u = int(val(2.5, conv(2.5)))
        self.assertIsInstance(u, int)
        self.assertEqual(u, 6)

        u = int(val(5, conv(2, 3, 4, 5)))
        self.assertIsInstance(u, int)
        self.assertEqual(u, 750000)

    def testFloat(self):
        with self.assertRaises(TypeError):
            float(val(1, units=mps))
        with self.assertRaises(TypeError):
            float(val(1j))
        with self.assertRaises(TypeError):
            float(val([1, 2]))

        u = float(val(5))
        self.assertIsInstance(u, float)
        self.assertEqual(u, 5)

        u = float(val(2.5))
        self.assertIsInstance(u, float)
        self.assertEqual(u, 2.5)

        u = float(val(5, conv(2, 3, 4, 5)))
        self.assertIsInstance(u, float)
        self.assertEqual(u, 750000)

    def testComplex(self):
        with self.assertRaises(TypeError):
            complex(val(1j, units=m))
        with self.assertRaises(TypeError):
            complex(val([1, 2]))

        u = complex(val(5))
        self.assertIsInstance(u, complex)
        self.assertEqual(u, 5)

        v = complex(val(5 + 6j, conv(2, 3, 4, 5)))
        self.assertIsInstance(v, complex)
        self.assertEqual(v, 750000 + 900000j)

    def testArray(self):
        u = np.array(val([1, 2], units=m))
        self.assertIsInstance(u, np.ndarray)
        self.assertIsInstance(u[0], Value)
        self.assertNumpyArrayEqual([val(1, units=m), val(2, units=m)], u)

        u = np.array(val([val(2, units=m), val(3, units=m)]))
        self.assertIsInstance(u, np.ndarray)
        self.assertIsInstance(u[0], float)
        self.assertNumpyArrayEqual([2, 3], u)

        u = np.array(val([2, 3]))
        self.assertIsInstance(u, np.ndarray)
        self.assertNumpyArrayEqual([2, 3], u)

        u = np.array(val([2, 3 + 1j], conv(2, 3, 4, 5)))
        self.assertIsInstance(u, np.ndarray)
        self.assertNumpyArrayEqual([300000, 450000 + 150000j], u)

    def testCopy(self):
        a = val(2, conv(3, 4, 5, 6), mps, kph)
        self.assertIs(a, copy.copy(a))
        self.assertIs(a, copy.deepcopy(a))

        b = val(1 + 2j, conv(3, 4, 5, 6), mps, kph)
        self.assertIs(b, copy.copy(b))
        self.assertIs(b, copy.deepcopy(b))

        c = val([10, 11], conv(3, 4, 5, 6), mps, kph)
        self.assertIsNot(c, copy.copy(c))
        self.assertIsNot(c, copy.deepcopy(c))
        self.assertNumpyArrayEqual(c, copy.copy(c))
        self.assertNumpyArrayEqual(c, copy.deepcopy(c))

        # Copy can be edited independently.
        c2 = copy.copy(c)
        c[1] = val(42, units=mps)
        self.assertNotNumpyArrayEqual(c, c2)

    def testPickle(self):
        examples = [
            val(1),
            val(2, conv(3, 4, 5, 6), mps, kph)
        ]
        for e in examples:
            self.assertDeepEqual(e, pickle.loads(pickle.dumps(e)))

    def testAddition(self):
        with self.assertRaises(UnitMismatchError):
            _ = val(2, units=m) + val(3, units=s)
        with self.assertRaises(UnitMismatchError):
            _ = val(2, units=m) + val(3, units=m * m)
        with self.assertRaises(UnitMismatchError):
            _ = val(2, units=m) + 3

        self.assertEqual(val(2) + val(3 + 1j), 5 + 1j)
        self.assertNumpyArrayEqual(val(2) + val([2, 3]), val([4, 5]))

        a = val(7, conv(5), units=s)
        b = val(3, units=s)
        self.assertEqual(a + b, val(38, units=s))

        # Prefers using the finer-grained conversion in the result.
        a = val(7, conv(5), units=s, display_units=kg)
        b = val(3, conv(4), units=s, display_units=m)
        c = val(11.75, conv(4), units=s, display_units=kg)
        self.assertDeepEqual(a + b, c)
        self.assertDeepEqual(b + a, c)

        self.assertEqual(val(2, conv(3, 4, 5, 6)) + val(7), 4800007)

        v1 = val(1, units=m)
        v2 = val(1, conv(exp10=3), units=m, display_units=s)
        self.assertEqual(1.0 * v2 / v1 + 5.0, 1005)

        # Tricky precision.
        a = val(1, conv(denom=101))
        b = val(1, conv(denom=101 * 103))
        self.assertDeepEqual(a + b, val(104, conv(denom=101 * 103)))
        self.assertDeepEqual(b + a, val(104, conv(denom=101 * 103)))

        # Adding dimensionless zero is fine, even if units don't match.
        self.assertDeepEqual(val(3, units=s) + 0, val(3, units=s))
        self.assertDeepEqual(0.0 + val(3, units=s), val(3, units=s))
        with self.assertRaises(UnitMismatchError):
            _ = val(3, units=s) + val(0, units=m)
            _ = val(0, units=s) + val(3, units=m)

    def testSubtraction(self):
        with self.assertRaises(UnitMismatchError):
            _ = val(2, units=m) - val(3, units=s)
        with self.assertRaises(UnitMismatchError):
            _ = val(2, units=m) - val(3, units=m * m)
        with self.assertRaises(UnitMismatchError):
            _ = val(2, units=m) - 3

        self.assertEqual(val(2) - val(5 + 1j), -3 - 1j)
        self.assertNumpyArrayEqual(val(2) - val([2, 3]), val([0, -1]))

        a = val(7, conv(5), units=s)
        b = val(3, units=s)
        self.assertEqual(a - b, val(32, units=s))

        # Subtracting dimensionless zero is fine, even if units don't match.
        self.assertDeepEqual(val(3, units=s) - 0, val(3, units=s))
        self.assertDeepEqual(0.0 - val(3, units=s), val(-3, units=s))
        with self.assertRaises(UnitMismatchError):
            _ = val(3, units=s) - val(0, units=m)
            _ = val(0, units=s) - val(3, units=m)

    def testMultiplication(self):
        self.assertEqual(val(2) * val(5), 10)

        self.assertDeepEqual(
            val(2, units=m) * val(3, units=s),
            val(6, units=m * s))

        self.assertDeepEqual(
            (val(2, conv(3, 4, 5, 6), m, s) *
                val(7, conv(8, 9, 10, 11), mps, kph)),
            val(14, conv(24, 18, 25, 17), m * mps, s * kph))

    def testDivision(self):
        self.assertEqual(val(5) / val(2), 2.5)

        self.assertDeepEqual(
            val(7, units=m) / val(4, units=s),
            val(1.75, units=m / s))

        self.assertDeepEqual(
            (val(7, conv(3, 9, 10, 11), mps, kph) /
                val(2, conv(8, 4, 5, 6), m, s)),
            val(3.5, conv(0.375, 9, 8, 5), mps / m, kph / s))

    def testIntDivision(self):
        with self.assertRaises(UnitMismatchError):
            _ = val(1, units=m) // val(1, units=s)

        self.assertIsInstance(val(5) // val(2), float)
        self.assertIsInstance(val(7, units=m) // val(4, units=m), float)

        self.assertEqual(val(5) // val(2), 2)
        self.assertEqual(val(-5) // val(-2), 2)
        self.assertEqual(val(-5) // val(2), -3)
        self.assertEqual(val(5) // val(-2), -3)

        self.assertEqual(val(7, units=m) // val(4, units=m), 1)

        self.assertEqual(
            val(7, conv(2), m, s) // val(4, units=m, display_units=h), 3)

    def testMod(self):
        with self.assertRaises(UnitMismatchError):
            _ = val(1, units=m) % val(1, units=s)

        self.assertDeepEqual(val(5) % val(3), val(2))
        self.assertDeepEqual(val(-5) % val(3), val(1))
        self.assertDeepEqual(
            val(7, units=m) % val(4, units=m), val(3, units=m))

        self.assertDeepEqual(
            val(7, conv(3), m, s) % val(4, conv(2), m, h),
            val(2.5, conv(2), m, h))

    def testDivMod(self):
        with self.assertRaises(UnitMismatchError):
            _ = divmod(val(1, units=m), val(1, units=s))

        q, r = divmod(val(7, conv(3), m, s), val(4, conv(2), m, h))
        self.assertEqual(q, 2)
        self.assertDeepEqual(r, val(2.5, conv(2), m, h))

    # noinspection PyTypeChecker
    def testPow(self):
        with self.assertRaises(TypeError):
            _ = 2 ** val(1, units=m)

        self.assertDeepEqual(val(2, units=m)**-2, val(0.25, units=m**-2))
        self.assertDeepEqual(val(2, units=m)**-1, val(0.5, units=m**-1))
        self.assertDeepEqual(val(2, units=m)**0, val(1))
        self.assertDeepEqual(val(2, units=m)**1, val(2, units=m))
        self.assertDeepEqual(val(2, units=m)**2, val(4, units=m**2))
        self.assertDeepEqual(val(2, units=m)**3, val(8, units=m**3))

        self.assertDeepEqual(val(4, units=m)**-0.5, val(0.5, units=m**-0.5))
        self.assertDeepEqual(val(4, units=m)**0.5, val(2, units=m**0.5))
        self.assertDeepEqual(val(4, units=m)**1.5, val(8, units=m**1.5))

        # Fractional powers that should work.
        for i in [1, 2, 3, 4, 6, 12]:
            self.assertDeepEqual(val(2**i, units=m**i)**(1.0 / i),
                                 val(2, units=m))

        # Conversion keeping/losing precision.
        self.assertDeepEqual(val(4, conv(numer=4))**0.5, val(2, conv(numer=2)))
        self.assertDeepEqual(val(4, conv(numer=2))**0.5,
                             val(2, conv(factor=2**0.5)))

    def testPos(self):
        self.assertDeepEqual(
            +val(2, conv(3, 5, 7, 11), mps, kph),
            val(2, conv(3, 5, 7, 11), mps, kph))

    def testNeg(self):
        self.assertDeepEqual(
            -val(2, conv(3, 5, 7, 11), mps, kph),
            val(-2, conv(3, 5, 7, 11), mps, kph))

    def testNonZero(self):
        self.assertFalse(bool(val(0, conv(3, 5, 7, 11), mps, kph)))
        self.assertFalse(bool(val(0)))
        self.assertTrue(bool(val(2, conv(3, 5, 7, 11), mps, kph)))
        self.assertTrue(bool(val(2)))

    def testNumpyMethod_isFinite(self):
        with self.assertRaises(TypeError):
            np.isfinite(val([], units=m))
        with self.assertRaises(TypeError):
            np.isfinite(val([1], units=m))

        v = val([2, 3, -2, float('nan'), float('inf')], conv(1, 2, 3, 4))
        self.assertNumpyArrayEqual(
            np.isfinite(v),
            [True, True, True, False, False])

        v = val([[2, 3], [-2, float('nan')]])
        self.assertNumpyArrayEqual(
            np.isfinite(v),
            [[True, True], [True, False]])

    def testGetItem(self):
        u = val(1, conv(exp10=1))
        v = val(2, conv(numer=3, denom=5, exp10=7), mps, kph)

        # Wrong kinds of index (unit array, slice).
        with self.assertRaises(TypeError):
            _ = u[mps]
        with self.assertRaises(TypeError):
            _ = u[1:2]

        # Safety against dimensionless unit ambiguity.
        with self.assertRaises(TypeError):
            _ = u[1.0]
        with self.assertRaises(TypeError):
            _ = u[1.0]
            _ = u[u]
        with self.assertRaises(TypeError):
            _ = u[1]
        with self.assertRaises(TypeError):
            _ = u[2 * v / v]
        self.assertEqual(u[v / v], 10)

        # Wrong unit.
        with self.assertRaises(UnitMismatchError):
            _ = u[v]

        self.assertEquals(u[''], 10)
        self.assertEquals(v[v], 1)
        self.assertEquals(v[val(2, conv(denom=5, exp10=7), mps, kph)], 3)
        self.assertEquals(v[val(2, conv(numer=3, exp10=7), mps, kph)], 0.2)
        self.assertEquals(val(2, conv(numer=3, exp10=7), mps, kph)[v], 5)

    def testIter(self):
        a = []
        for e in val([1, 2, 4], conv(numer=2), mps, kph):
            a.append(e)
        self.assertEquals(len(a), 3)
        self.assertDeepEqual(a[0], val(1, conv(numer=2), mps, kph))
        self.assertDeepEqual(a[1], val(2, conv(numer=2), mps, kph))
        self.assertDeepEqual(a[2], val(4, conv(numer=2), mps, kph))

    def testHash(self):
        d = dict()
        v = val(2, conv(denom=5), mps, kph)
        w = val(3, conv(exp10=7), mps, kph)
        d[v] = 5
        d[w] = "b"
        self.assertEquals(d[v], 5)
        self.assertEquals(d[w], "b")

    def testStr(self):
        self.assertEquals(str(val(2, conv(3, 4, 5, 6), s, m)), '2.0 m')
        self.assertEquals(str(val(2j, conv(3, 4, 5, 6), s)), '2j s')
        self.assertEquals(str(val(1, units=m, display_units=s)), 's')
        self.assertEquals(str(val(1, units=m)), 'm')
        self.assertEquals(str(val(1, units=h)), 's^3600')
        self.assertEquals(str(val([2, 3, 5], units=h, display_units=m)),
                          '[2 3 5] m')
        self.assertEquals(str(val([2, 3], units=h, display_units=mps)),
                          '[2 3] m/s')

    def testIsCompatible(self):
        equivalence_groups = [
            [val(0), val(1)],
            [val(1, units=m), val(5, conv(3), units=m, display_units=s)],
            [val(9, units=s)],
            [val(13, units=m**2), val(5, units=m**2)]
        ]
        for g1 in equivalence_groups:
            for g2 in equivalence_groups:
                for e1 in g1:
                    for e2 in g2:
                        self.assertEqual(e1.isCompatible(e2), g1 is g2)

    def testIsAngle(self):
        self.assertFalse(val(1).isAngle())
        self.assertFalse(val(1).is_angle)
        self.assertFalse(val(1, units=m).isAngle())
        self.assertFalse(val(1, units=m).is_angle)
        self.assertTrue(val(1, units=rad).isAngle())
        self.assertTrue(val(1, units=rad).is_angle)
        self.assertFalse(val(1, units=rad**2).isAngle())
        self.assertFalse(val(1, units=rad**2).is_angle)

    def testInUnitsOf(self):
        with self.assertRaises(UnitMismatchError):
            val(1, units=m).inUnitsOf(val(2, units=s))

        self.assertEqual(val(5).inUnitsOf(8), 0.625)
        self.assertDeepEqual(
            val(5, conv(3), units=m, display_units=s).inUnitsOf(
                val(8, conv(denom=7), units=m, display_units=kg)),
            val(13.125, conv(denom=7), units=m, display_units=kg))

    def testUnit(self):
        self.assertDeepEqual(val(7, conv(2, 3, 4, 5), m, s).unit,
                             val(1, conv(2, 3, 4, 5), m, s))

if __name__ == "__main__":
    unittest.main()
