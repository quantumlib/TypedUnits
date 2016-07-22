#!/usr/bin/python
import unittest
from fastunits import Value


class UsageTests(unittest.TestCase):
    def testParsingByComparison(self):
        self.assertRaises(lambda: Value(1, 'junk'))
        self.assertRaises(lambda: Value(1, 'kminute'))

        self.assertLess(Value(1, 'in'), Value(1, 'm'))
        self.assertLess(Value(1, 'cm'), Value(1, 'in'))
        self.assertLess(Value(1, 'gauss'), Value(1, 'mT'))
        self.assertLess(Value(1, 'minute'), Value(100, 's'))
        self.assertEqual(Value(10, 'hertz'), Value(10, 'Hz'))
        self.assertEqual(Value(10, 'Mg'), Value(10000, 'kg'))
        self.assertEqual(Value(10, 'Mg'), Value(10000000, 'g'))
        self.assertNotEqual(Value(10, 'decibel'), Value(10, 'mol'))

if __name__ == "__main__":
    unittest.main()
