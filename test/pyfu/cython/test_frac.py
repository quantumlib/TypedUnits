import unittest
import pyfu._all_cythonized as u


def frac(numer=1, denom=1):
    return {'numer': numer, 'denom': denom}


class FracTests(unittest.TestCase):
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

        self.assertEqual(u.gcd(2 * 3 * 5, 3 * 5 * 7), 3 * 5)
        self.assertEqual(u.gcd(2 * 3 * 5 * 5, 3 * 5 * 7), 3 * 5)
        self.assertEqual(u.gcd(2 * 3 * 5 * 5, 3 * 5 * 5 * 7), 3 * 5 * 5)
        self.assertEqual(u.gcd(945356, 633287), 1)
        self.assertEqual(u.gcd(+541838, +778063), 11)
        self.assertEqual(u.gcd(-541838, +778063), 11)
        self.assertEqual(u.gcd(+541838, -778063), 11)
        self.assertEqual(u.gcd(-541838, -778063), 11)

    def testFracLeastTerms(self):
        self.assertRaises(TypeError, lambda: u.frac_least_terms("a", "b"))
        self.assertRaises(ZeroDivisionError, lambda: u.frac_least_terms(0, 0))
        self.assertRaises(ZeroDivisionError, lambda: u.frac_least_terms(1, 0))

        self.assertEqual(u.frac_least_terms(0, 3), frac(0))
        self.assertEqual(u.frac_least_terms(0, -3), frac(0))
        self.assertEqual(u.frac_least_terms(2, 3), frac(2, 3))
        self.assertEqual(u.frac_least_terms(2, 4), frac(denom=2))

        self.assertEqual(u.frac_least_terms(+4, +6), frac(+2, 3))
        self.assertEqual(u.frac_least_terms(-4, +6), frac(-2, 3))
        self.assertEqual(u.frac_least_terms(+4, -6), frac(-2, 3))
        self.assertEqual(u.frac_least_terms(-4, -6), frac(+2, 3))

        self.assertEqual(u.frac_least_terms(1, 1), frac())
        self.assertEqual(u.frac_least_terms(0, 1), frac(0))
        self.assertEqual(u.frac_least_terms(121, 33), frac(11, 3))

    def testFracTimes(self):
        self.assertEqual(
            u.frac_times(frac(0), frac(5, 7)),
            frac(0))

        self.assertEqual(
            u.frac_times(frac(0), frac(-5, 7)),
            frac(0))

        self.assertEqual(
            u.frac_times(frac(2, 3), frac(0)),
            frac(0))

        self.assertEqual(
            u.frac_times(frac(2, 3), frac(5, 7)),
            frac(10, 21))

        self.assertEqual(
            u.frac_times(frac(2, 33), frac(55, 7)),
            frac(10, 21))

        self.assertEqual(
            u.frac_times(frac(22, 3), frac(5, 77)),
            frac(10, 21))

        self.assertEqual(
            u.frac_times(frac(-2, 3), frac(5, 7)),
            frac(-10, 21))

        self.assertEqual(
            u.frac_times(frac(2, 3), frac(-5, 7)),
            frac(-10, 21))

        self.assertEqual(
            u.frac_times(frac(-2, 3), frac(-5, 7)),
            frac(10, 21))

    def testFracDiv(self):
        self.assertRaises(ZeroDivisionError, lambda: u.frac_div(
            frac(2, 3), frac(0)))

        self.assertEqual(
            u.frac_div(frac(0), frac(5, 7)),
            frac(0))

        self.assertEqual(
            u.frac_div(frac(0), frac(-5, 7)),
            frac(0))

        self.assertEqual(
            u.frac_div(frac(2, 3), frac(5, 7)),
            frac(14, 15))

        self.assertEqual(
            u.frac_div(frac(22, 3), frac(55, 7)),
            frac(14, 15))

        self.assertEqual(
            u.frac_div(frac(2, 33), frac(5, 77)),
            frac(14, 15))

        self.assertEqual(
            u.frac_div(frac(-2, 3), frac(5, 7)),
            frac(-14, 15))

        self.assertEqual(
            u.frac_div(frac(2, 3), frac(-5, 7)),
            frac(-14, 15))

        self.assertEqual(
            u.frac_div(frac(-2, 3), frac(-5, 7)),
            frac(14, 15))

    def testFloatToTwelthsFrac(self):
        self.assertRaises(ValueError, lambda: u.float_to_twelths_frac(1.0 / 24))
        self.assertRaises(ValueError, lambda: u.float_to_twelths_frac(1.0 / 7))
        self.assertRaises(ValueError, lambda: u.float_to_twelths_frac(1.0 / 5))
        self.assertRaises(ValueError, lambda: u.float_to_twelths_frac(1.0 / 11))
        self.assertRaises(ValueError, lambda: u.float_to_twelths_frac(1.0 / 13))

        self.assertEqual(u.float_to_twelths_frac(0), frac(0))

        self.assertEqual(
            u.float_to_twelths_frac(502),
            frac(502))
        self.assertEqual(
            u.float_to_twelths_frac(1.0 / 12),
            frac(denom=12))
        self.assertEqual(
            u.float_to_twelths_frac(-1.0 / 12),
            frac(-1, 12))
        self.assertEqual(
            u.float_to_twelths_frac(501.0 / 3),
            frac(167))
        self.assertEqual(
            u.float_to_twelths_frac(502.0 / 3),
            frac(502, 3))

        # Precision.
        self.assertEqual(
            u.float_to_twelths_frac((1 << 55) + 1),
            frac((1 << 55) + 1))
        self.assertEqual(
            u.float_to_twelths_frac(float(1 << 55) / 3.0),
            frac(1 << 55, 3))

    def testFracToDouble(self):
        self.assertEqual(u.frac_to_double(frac(0)), 0)
        self.assertEqual(u.frac_to_double(frac(2, 3)), 2.0 / 3)


if __name__ == "__main__":
    unittest.main()
