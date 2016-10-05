#!/usr/bin/python

import unittest
from pyfu import Value, Unit, Complex, UnitMismatchError
import numpy as np
from pyfu.units import kilometer, meter, mm, second, us, ns


class FastUnitsTests(unittest.TestCase):
    def testConstruction(self):
        x = 2 * Unit('')
        y = Value(5, 'ns')
        self.assertIsInstance(x, Value)
        self.assertIsInstance(y, Value)
        self.assertTrue(x.isDimensionless())
        self.assertIsInstance(3j * x, Complex)

    def testDimensionless(self):
        """Test that dimensionless values act like floats"""
        x = Value(1.5, '')
        y = Value(1.5, 'us/ns')
        self.assertEqual(x, 1.5)
        self.assertEqual(np.ceil(x), 2.)
        self.assertEqual(np.floor(x), 1.)
        self.assertEqual(y, 1500.)

    def testAddition(self):
        n = Value(2, '')
        x = Value(1.0, kilometer)
        y = Value(3, 'meter')
        a = Value(20, 's')
        self.assertEqual(x + y, Value(1003, 'meter'))
        self.assertNotEqual(x, y)
        self.assertNotEqual(x, a)
        with self.assertRaises(UnitMismatchError):
            _ = y + a
        with self.assertRaises(UnitMismatchError):
            _ = x + 3.0
        _ = x + y
        self.assertEqual(x-y, Value(997, 'm'))
        self.assertIsInstance(x*1j + y, Complex)
        self.assertEqual(n+1, 3)

    def testMultiplication(self):
        n = Value(2, '')
        x = Value(1.0 + 2j, meter)
        y = Value(3, mm)
        a = Value(20, second)
        self.assertEqual(a * x, x * a)
        self.assertTrue((x / y).isDimensionless())

    def testPower(self):
        x = 2*mm
        y = 4*mm
        z = (x*y)**.5
        self.assertLess(abs(z**2- Value(8, 'mm^2')),  Value(1e-6, mm**2))

    def testRepr(self):
        from pyfu.units import km, kg
        self.assertEqual(repr(Value(1, mm)), "Value(1.0, 'mm')")
        self.assertEqual(repr(Value(4, mm)), "Value(4.0, 'mm')")
        self.assertEqual(repr(Value(1j+5, km * kg)), "Value((5+1j), 'kg*km')")

    def testStr(self):
        self.assertEqual(str(Value(1, mm)), 'mm')
        self.assertEqual(str(Value(4, mm)), '4.0 mm')

    def testDivmod(self):
        x = 4.001*us
        self.assertEquals(x//(4*ns), 1000)
        self.assertTrue(abs(x % (4*ns) - 1*ns) < .00001*ns)
        y = divmod(x, 2*ns)

    def testConversion(self):
        x = Value(3, 'm')
        self.assertEquals(x['mm'], 3000.0)
        with self.assertRaises(UnitMismatchError):
            x['s']
        y = Value(1000, 'Mg')
        self.assertEquals(y.inBaseUnits().value, 1000000.0)
        self.assertEquals(x.inUnitsOf('mm'), 3000 * mm)

    def testHash(self):
        x = Value(3, 'ks')
        y = Value(3000, 's')
        self.assertEquals(hash(x), hash(y))
        z = Value(3.1, '')
        self.assertEquals(hash(z), hash(3.1))
        hash(Value(4j, 'V'))

if __name__ == "__main__":
    unittest.main()
