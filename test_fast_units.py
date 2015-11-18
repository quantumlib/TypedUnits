#!/usr/bin/python

import unittest
import fast_units as U
from fast_units import Value, Unit, Complex, ValueArray
import numpy as np

class FastUnitsTests(unittest.TestCase):
    def testConstruction(self):
        x = 2*Unit('')
        y = Value(5, 'ns')
        self.assertIsInstance(x, Value)
        self.assertIsInstance(y, Value)
        self.assertTrue(x.isDimensionless())
        self.assertIsInstance(3j*x, Complex)
        self.assertIsInstance(np.arange(5)*U.ns, ValueArray)

    def testAddition(self):
        n = Value(2, '')
        x = Value(1.0, U.kilometer);
        y = Value(3, 'meter');
        a = Value(20, 's');
        self.assertEqual(x + y, Value(1003, 'meter'))
        self.assertNotEqual(x, y)
        self.assertNotEqual(x, a)
        with self.assertRaises(ValueError):
            _ = y + a 
        with self.assertRaises(ValueError):
            _ = x + 3.0
        _ = x + y
        self.assertEqual(x-y, Value(997, 'm'))
        self.assertIsInstance(x*1j + y, Complex)
        self.assertEqual(n+1, 3)

    def testMultiplication(self):
        n = Value(2, '')
        x = Value(1.0+2j, U.meter)
        y = Value(3, U.mm)
        a = Value(20, U.second)
        self.assertEqual(a*x, x*a)
        self.assertTrue((x/y).isDimensionless())
        
    def testPower(self):
        x = 2*U.mm
        y = 4*U.mm
        z = (x*y)**.5
        self.assertLess(abs(z**2- Value(8, 'mm^2')),  Value(1e-6, U.mm**2))

    def testStringification(self):
        x = Value(4, U.mm)
        self.assertEqual(repr(x), 'Value(4.0, "mm")')
        self.assertEqual(str(x), '4.0 mm');
        
    def testDivmod(self):
        x = 4.001*U.us
        self.assertEquals(x//(4*U.ns), 1000)
        self.assertTrue(abs(x % (4*U.ns) - 1*U.ns) < .00001*U.ns)
        y = divmod(x, 2*U.ns)

    def testConversion(self):
        x = Value(3, 'm')
        self.assertEquals(x['mm'], 3000.0)
        with self.assertRaises(TypeError):
            x['s']
        y = Value(1000, 'Mg')
        self.assertEquals(y.inBaseUnits().value, 1000000.0)
        self.assertEquals(x.inUnitsOf('mm'), 3000*U.mm)

if __name__ == "__main__":
    unittest.main()
