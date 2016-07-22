#!/usr/bin/python
import unittest
from fastunits import Value


class UsageTests(unittest.TestCase):
    def testParsingByComparison(self):
        self.assertLess(Value(1, 'in'), Value(1, 'm'))
        self.assertLess(Value(1, 'cm'), Value(1, 'in'))
        self.assertLess(Value(1, 'gauss'), Value(1, 'mT'))
        self.assertLess(Value(1, 'minute'), Value(100, 's'))
        self.assertEqual(Value(10, 'hertz'), Value(10, 'Hz'))
        self.assertEqual(Value(10, 'Mg'), Value(10000, 'kg'))
        self.assertEqual(Value(10, 'Mg'), Value(10000000, 'g'))
        self.assertNotEqual(Value(10, 'decibel'), Value(10, 'mol'))
        self.assertEqual(Value(1, 'millisecond'), Value(1, 'ms'))
        self.assertEqual(Value(1, ''), Value(1, 'm/m'))

    def testRadiansVsSteradians(self):
        self.assertNotEqual(Value(1, 'rad'), Value(1, 'sr'))
        self.assertEqual(Value(2, 'rad')**2, Value(4, 'sr'))
        self.assertEqual(Value(16, 'rad'), Value(256, 'sr')**0.5)
        self.assertEqual(Value(32, 'rad')*Value(2, 'sr'), Value(16, 'sr')**1.5)
        self.assertEqual(Value(1, 'rad')**(4/3.0), Value(1, 'sr')**(2/3.0))

if __name__ == "__main__":
    unittest.main()
