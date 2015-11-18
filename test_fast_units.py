#!/usr/bin/python

import unittest
import fast_units as U
from fast_units import Value, Unit
import numpy as np

class FastUnitsTests(unittest.TestCase):
    def testConstruction(self):
        x = 2*Unit('')
        y = Value(5, 'ns')
        self.assertIsInstance(x, Value)
        self.assertIsInstance(y, Value)
        self.assertTrue(x.isDimensionless())
        self.assertIsInstance(3j*x, Value)
        self.assertIsInstance(np.arange(5)*U.ns, Value)

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
        self.assertIsInstance(x*1j + y, Value)
        self.assertEqual(n+1, 3)

    def testMultiplication(self):
        n = Value(2, '')
        x = Value(1.0+2j, U.meter)
        y = Value(3, U.mm)
        a = Value(20, U.second)
        self.assertEqual(a*x, x*a)
        self.assertTrue((x/y).isDimensionless())
        
    def testStringification(self):
        x = Value(4, U.mm)
        self.assertEqual(repr(x), 'Value(4.0, "mm")')
        self.assertEqual(str(x), '4.0 mm');
        
    def testDivmod(self):
        x = 4.001*U.us
        self.assertEquals(x//(4*U.ns), 1000)
        self.assertTrue(abs(x % (4*U.ns) - 1*U.ns) < .00001*U.ns)
if __name__ == "__main__":
    unittest.main()
