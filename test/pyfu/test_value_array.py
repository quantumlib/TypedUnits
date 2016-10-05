import unittest
from pyfu import ValueArray, UnitMismatchError, UnitArray
from pyfu._all_cythonized import raw_UnitArray
import numpy as np


class FastUnitsTests(unittest.TestCase):
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

        # Fallback case.
        v = ValueArray.raw([1, 2, 3], 2, 5, 10,
                           raw_UnitArray([('muffin', 1, 1)]),
                           raw_UnitArray([('cookie', 1, 1)]))
        self.assertEqual(repr(v),
                         "raw_WithUnit(array([1, 2, 3]), 2, 5, 10, "
                         "raw_UnitArray([('muffin', 1, 1)]), "
                         "raw_UnitArray([('cookie', 1, 1)]))")

    def testStr(self):
        from pyfu.units import mm
        self.assertEqual(str([] * mm**3), '[] mm^3')
        self.assertEqual(str([2, 3, 5] * mm), '[ 2.  3.  5.] mm')

if __name__ == "__main__":
    unittest.main()
