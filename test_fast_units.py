#!/usr/bin/python

import unittest
import fast_units
from fast_units import Value, Complex
import numpy as np

dimensionless = (0,0,0,0,0,0,0,0,0)
meter = (1,0,0,0,0,0,0,0,0)
kg = (0,1,0,0,0,0,0,0,0)
second = (0,0,1,0,0,0,0,0,0)

class FastUnitsTests(unittest.TestCase):
    def testConstruction(self):
        self.assertIsInstance(Value(2, 0, dimensionless), Value)
        self.assertIsInstance(Complex(3, 0, meter), Complex)
        self.assertIsInstance(Value(1, -3, second), Value)
        self.assertIsInstance(Complex(1, 1, dimensionless), Complex)
        self.assertIsInstance(Value(np.int32(1), 2, meter), Value)
        self.assertIsInstance(Complex(np.float64(1), 2, meter), Complex)

    def testAddition(self):
        n = Value(2, 0, dimensionless)
        x = Value(1.0, 0, meter);
        y = Value(3, -3, meter);
        a = Value(2, 1, second);
        self.assertEqual(x + y, Value(1003, -3, meter))
        self.assertNotEqual(x, y)
        self.assertNotEqual(x, a)
        with self.assertRaises(ValueError):
            _ = y + a
        with self.assertRaises(ValueError):
            _ = x + 3.0
        _ = x + y
        self.assertIsInstance(x*1j + y, Complex)
        self.assertEqual(n+1, 3)

    def testMultiplication(self):
        n = Value(2, 0, dimensionless)
        x = Complex(1.0+2j, 0, meter);
        y = Value(3, -3, meter);
        a = Value(2, 1, second);
        self.assertEqual(a*x, x*a)
        self.assertEqual((a*x).get_unit_power(0), 1)
        self.assertEqual((a*x).get_unit_power(1), 0)
        self.assertEqual((a*x).get_unit_power(2), 1)
        self.assertEqual((a*x).get_unit_power(8), 0)
        self.assertEqual((x / y).get_unit_power(0), 0)
        self.assertEqual(n * a, Value(40, 0, second))
        self.assertNotEqual(n*a, Value(40, 0, dimensionless))
        self.assertNotEqual(n*a, Value(40, 0, meter))
        
        
    def testPromotion(self):
        x = Value(2, 0, dimensionless)
        y = Complex(3, -1, dimensionless)
        a = Value(1, 0, meter)
        b = Complex(2, 0, meter)
        self.assertIsInstance(x + 1j, Complex)
        self.assertIsInstance(x + 3.0, Value)
        self.assertIsInstance(-3 + x, Value)
        self.assertIsInstance(-1j + x, Complex)
        self.assertIsInstance(y * 3j, Complex)
        self.assertIsInstance(a *2j, Complex)

    def testBadCreate(self):
        with self.assertRaises(TypeError):
            Value(4+3j, 0, meter)
        with self.assertRaises(TypeError):
            Value([1,2,3], 0, meter)

    def testPosNegAbs(self):
        x = Value(2, 1, dimensionless);
        y = Complex(2j, 1, dimensionless);
        z = Value(-2, 1, dimensionless);
        self.assertEqual(-x, z)
        self.assertEqual(x, abs(y));
        self.assertEqual(y, +y);

if __name__ == "__main__":
    unittest.main()
