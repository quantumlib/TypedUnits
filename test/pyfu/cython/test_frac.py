import unittest
import pyfu.__all_cythonized as u


class FracUtilTests(unittest.TestCase):
    def testGcd(self):
        self.assertRaises(TypeError, lambda: u.gcd("a", "b"))

        self.assertEqual(u.gcd(0, -2), 2)
        self.assertEqual(u.gcd(0, -1), 1)
        self.assertEqual(u.gcd(0, 0), 0)
        self.assertEqual(u.gcd(0, 1), 1)
        self.assertEqual(u.gcd(0, 2), 2)

        self.assertEqual(u.gcd(1, -2), 1)
        self.assertEqual(u.gcd(1, -1), 1)
        self.assertEqual(u.gcd(1, 0), 1)
        self.assertEqual(u.gcd(1, 1), 1)
        self.assertEqual(u.gcd(1, 2), 1)

        self.assertEqual(u.gcd(2, -2), 2)
        self.assertEqual(u.gcd(2, -1), 1)
        self.assertEqual(u.gcd(2, 0), 2)
        self.assertEqual(u.gcd(2, 1), 1)
        self.assertEqual(u.gcd(2, 2), 2)

        self.assertEqual(u.gcd(3, -2), 1)
        self.assertEqual(u.gcd(3, -1), 1)
        self.assertEqual(u.gcd(3, 0), 3)
        self.assertEqual(u.gcd(3, 1), 1)
        self.assertEqual(u.gcd(3, 2), 1)

        self.assertEqual(u.gcd(2*3*5, 3*5*7), 3*5)
        self.assertEqual(u.gcd(2*3*5*5, 3*5*7), 3*5)
        self.assertEqual(u.gcd(2*3*5*5, 3*5*5*7), 3*5*5)
        self.assertEqual(u.gcd(945356, 633287), 1)
        self.assertEqual(u.gcd(+541838, +778063), 11)
        self.assertEqual(u.gcd(-541838, +778063), 11)
        self.assertEqual(u.gcd(+541838, -778063), 11)
        self.assertEqual(u.gcd(-541838, -778063), 11)

    def testFracLeastTerms(self):
        self.assertRaises(TypeError, lambda: u.frac_least_terms("a", "b"))
        self.assertRaises(ZeroDivisionError, lambda: u.frac_least_terms(0, 0))
        self.assertRaises(ZeroDivisionError, lambda: u.frac_least_terms(1, 0))

        self.assertEqual(u.frac_least_terms(0, 3), {'numer': 0, 'denom': 1})
        self.assertEqual(u.frac_least_terms(0, -3), {'numer': 0, 'denom': 1})
        self.assertEqual(u.frac_least_terms(2, 3), {'numer': 2, 'denom': 3})
        self.assertEqual(u.frac_least_terms(2, 4), {'numer': 1, 'denom': 2})

        self.assertEqual(u.frac_least_terms(+4, +6), {'numer': +2, 'denom': 3})
        self.assertEqual(u.frac_least_terms(-4, +6), {'numer': -2, 'denom': 3})
        self.assertEqual(u.frac_least_terms(+4, -6), {'numer': -2, 'denom': 3})
        self.assertEqual(u.frac_least_terms(-4, -6), {'numer': +2, 'denom': 3})

        self.assertEqual(u.frac_least_terms(1, 1), {'numer': 1, 'denom': 1})
        self.assertEqual(u.frac_least_terms(0, 1), {'numer': 0, 'denom': 1})
        self.assertEqual(u.frac_least_terms(121, 33), {'numer': 11, 'denom': 3})

    def testFracTimes(self):
        self.assertEqual(
            u.frac_times({'numer': 0, 'denom': 1}, {'numer': 5, 'denom': 7}),
            {'numer': 0, 'denom': 1})

        self.assertEqual(
            u.frac_times({'numer': 0, 'denom': 1}, {'numer': -5, 'denom': 7}),
            {'numer': 0, 'denom': 1})

        self.assertEqual(
            u.frac_times({'numer': 2, 'denom': 3}, {'numer': 0, 'denom': 1}),
            {'numer': 0, 'denom': 1})

        self.assertEqual(
            u.frac_times({'numer': 2, 'denom': 3}, {'numer': 5, 'denom': 7}),
            {'numer': 10, 'denom': 21})

        self.assertEqual(
            u.frac_times({'numer': 2, 'denom': 33}, {'numer': 55, 'denom': 7}),
            {'numer': 10, 'denom': 21})

        self.assertEqual(
            u.frac_times({'numer': 22, 'denom': 3}, {'numer': 5, 'denom': 77}),
            {'numer': 10, 'denom': 21})

        self.assertEqual(
            u.frac_times({'numer': -2, 'denom': 3}, {'numer': 5, 'denom': 7}),
            {'numer': -10, 'denom': 21})

        self.assertEqual(
            u.frac_times({'numer': 2, 'denom': 3}, {'numer': -5, 'denom': 7}),
            {'numer': -10, 'denom': 21})

        self.assertEqual(
            u.frac_times({'numer': -2, 'denom': 3}, {'numer': -5, 'denom': 7}),
            {'numer': 10, 'denom': 21})

    def testFracDiv(self):
        self.assertRaises(ZeroDivisionError, lambda: u.frac_div(
            {'numer': 2, 'denom': 3}, {'numer': 0, 'denom': 1}))

        self.assertEqual(
            u.frac_div({'numer': 0, 'denom': 1}, {'numer': 5, 'denom': 7}),
            {'numer': 0, 'denom': 1})

        self.assertEqual(
            u.frac_div({'numer': 0, 'denom': 1}, {'numer': -5, 'denom': 7}),
            {'numer': 0, 'denom': 1})

        self.assertEqual(
            u.frac_div({'numer': 2, 'denom': 3}, {'numer': 5, 'denom': 7}),
            {'numer': 14, 'denom': 15})

        self.assertEqual(
            u.frac_div({'numer': 22, 'denom': 3}, {'numer': 55, 'denom': 7}),
            {'numer': 14, 'denom': 15})

        self.assertEqual(
            u.frac_div({'numer': 2, 'denom': 33}, {'numer': 5, 'denom': 77}),
            {'numer': 14, 'denom': 15})

        self.assertEqual(
            u.frac_div({'numer': -2, 'denom': 3}, {'numer': 5, 'denom': 7}),
            {'numer': -14, 'denom': 15})

        self.assertEqual(
            u.frac_div({'numer': 2, 'denom': 3}, {'numer': -5, 'denom': 7}),
            {'numer': -14, 'denom': 15})

        self.assertEqual(
            u.frac_div({'numer': -2, 'denom': 3}, {'numer': -5, 'denom': 7}),
            {'numer': 14, 'denom': 15})

    def testFloatToTwelthsFrac(self):
        self.assertRaises(ValueError, lambda: u.float_to_twelths_frac(1.0/24))
        self.assertRaises(ValueError, lambda: u.float_to_twelths_frac(1.0/7))
        self.assertRaises(ValueError, lambda: u.float_to_twelths_frac(1.0/5))
        self.assertRaises(ValueError, lambda: u.float_to_twelths_frac(1.0/11))
        self.assertRaises(ValueError, lambda: u.float_to_twelths_frac(1.0/13))

        self.assertEqual(u.float_to_twelths_frac(0), {'numer': 0, 'denom': 1})

        self.assertEqual(
            u.float_to_twelths_frac(502),
            {'numer': 502, 'denom': 1})
        self.assertEqual(
            u.float_to_twelths_frac(1.0/12),
            {'numer': 1, 'denom': 12})
        self.assertEqual(
            u.float_to_twelths_frac(-1.0 / 12),
            {'numer': -1, 'denom': 12})
        self.assertEqual(
            u.float_to_twelths_frac(501.0 / 3),
            {'numer': 167, 'denom': 1})
        self.assertEqual(
            u.float_to_twelths_frac(502.0 / 3),
            {'numer': 502, 'denom': 3})

        # Precision.
        self.assertEqual(
            u.float_to_twelths_frac((1 << 55) + 1),
            {'numer': (1 << 55) + 1, 'denom': 1})
        self.assertEqual(
            u.float_to_twelths_frac(float(1 << 55) / 3.0),
            {'numer': 1 << 55, 'denom': 3})

    def testFracToDouble(self):
        self.assertEqual(u.frac_to_double({'numer': 0, 'denom': 1}), 0)
        self.assertEqual(u.frac_to_double({'numer': 2, 'denom': 3}), 2.0/3)

if __name__ == "__main__":
    unittest.main()
