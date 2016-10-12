import unittest
from pyfu import ValueArray, UnitMismatchError
from pyfu._all_cythonized import raw_WithUnit, raw_UnitArray
import numpy as np


class ValueArrayTests(unittest.TestCase):
    def assertNumpyArrayEqual(self, a, b):
        if len(a) != len(b) or not np.all(a == b):
            msg = 'not np.all(%s == %s)' % (repr(a), repr(b))
            msg = self._formatMessage(None, msg)
            raise self.failureException(msg)

    def testConstruction(self):
        from pyfu.units import ns, ps
        self.assertIsInstance([1, 2, 3] * ns, ValueArray)
        self.assertNumpyArrayEqual([1, 2, 3] * ns, [1000, 2000, 3000] * ps)

    def testSlicing(self):
        from pyfu.units import ms, ns
        self.assertNumpyArrayEqual(([0, 1, 2, 3, 4] * ms)[3:], [3, 4] * ms)
        self.assertNumpyArrayEqual(([0, 1, 2, 3, 4] * ns)[::2], [0, 2, 4] * ns)

    def testSetItem(self):
        from pyfu.units import km, m, s
        v = [1, 2, 3] * km

        with self.assertRaises(UnitMismatchError):
            v[0] = 2 * s

        v[0] = 2 * km
        v[1] = 16 * m

        self.assertNumpyArrayEqual(v, [2000, 16, 3000] * m)

    def testAddition(self):
        from pyfu.units import km, m
        self.assertNumpyArrayEqual([1, 2, 3] * km + [2, 3, 5] * m,
                                   [1002, 2003, 3005] * m)

        self.assertNumpyArrayEqual([1, 2, 3] * km + 5 * m,
                                   [1005, 2005, 3005] * m)

        with self.assertRaises(UnitMismatchError):
            _ = 1.0 + [1, 2, 3] * km

    def testMultiplication(self):
        from pyfu.units import km, m
        self.assertNumpyArrayEqual(([1, 2, 3] * km) * ([2, 3, 5] * m),
                                   [2, 6, 15] * (km * m))

        self.assertNumpyArrayEqual(([1, 2, 3] * km) * 5j,
                                   [5j, 10j, 15j] * km)

    def testPower(self):
        from pyfu.units import s
        self.assertNumpyArrayEqual(([1, 2, 3] * s)**2,
                                   [1, 4, 9] * s * s)

    def testRepr(self):
        from pyfu.units import km, kg, s
        self.assertEqual(repr([] * s),
                         "ValueArray(array([], dtype=float64), 's')")
        self.assertEqual(repr([2, 3] * km),
                         "ValueArray(array([ 2.,  3.]), 'km')")
        self.assertEqual(repr([3j] * km * kg),
                         "ValueArray(array([ 0.+3.j]), 'kg*km')")
        self.assertEqual(repr([-1] * km ** 2 / kg ** 3 * s),
                         "ValueArray(array([-1.]), 'km^2*s/kg^3')")
        self.assertEqual(repr([-1] * km**(2/3.0) / kg**3 * s),
                         "ValueArray(array([-1.]), 'km^(2/3)*s/kg^3')")

        # Numpy abbreviation is allowed.
        self.assertEqual(
            repr(list(range(50000)) * km),
            "ValueArray(array([  0.00000000e+00,   1.00000000e+00,   "
            "2.00000000e+00, ...,\n         4.99970000e+04,   "
            "4.99980000e+04,   4.99990000e+04]), 'km')")

        # Fallback case.
        v = raw_WithUnit([1, 2, 3],
                         {
                             'factor': 3.0,
                             'ratio': {
                                 'numer': 2,
                                 'denom': 5,
                             },
                             'exp10': 10
                         },
                         raw_UnitArray([('muffin', 1, 1)]),
                         raw_UnitArray([('cookie', 1, 1)]))
        self.assertEqual(repr(v),
                         "raw_WithUnit(array([1, 2, 3]), "
                         "{'exp10': 10, "
                         "'ratio': {'numer': 2, 'denom': 5}, "
                         "'factor': 3.0}, "
                         "raw_UnitArray([('muffin', 1, 1)]), "
                         "raw_UnitArray([('cookie', 1, 1)]))")

    def testStr(self):
        from pyfu.units import mm
        self.assertEqual(str([] * mm**3), '[] mm^3')
        self.assertEqual(str([2, 3, 5] * mm), '[ 2.  3.  5.] mm')

if __name__ == "__main__":
    unittest.main()
