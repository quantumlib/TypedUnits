import unittest
import pyfu._all_cythonized as u


def frac(numer=1, denom=1):
    return {'numer': numer, 'denom': denom}


def conv(factor=1.0, numer=1, denom=1, exp10=0):
    return {'factor': factor, 'ratio': frac(numer, denom), 'exp10': exp10}


class ConversionTests(unittest.TestCase):
    def testConversionToDouble(self):
        # Identity.
        self.assertEqual(u.conversion_to_double(conv()), 1)

        # Single elements.
        self.assertEqual(u.conversion_to_double(conv(factor=2)), 2)
        self.assertEqual(u.conversion_to_double(conv(numer=2)), 2)
        self.assertEqual(u.conversion_to_double(conv(denom=2)), 0.5)
        self.assertEqual(u.conversion_to_double(conv(exp10=2)), 100)

        # All elements.
        self.assertEqual(u.conversion_to_double(
            conv(factor=1.75, numer=5, denom=8, exp10=2)), 109.375)

    def testConversionTimes(self):
        # Identity.
        self.assertEqual(
            u.conversion_times(conv(), conv()),
            conv())

        # Single elements.
        self.assertEqual(
            u.conversion_times(conv(factor=2), conv(factor=3)),
            conv(factor=6))
        self.assertEqual(
            u.conversion_times(conv(numer=2), conv(numer=3)),
            conv(numer=6))
        self.assertEqual(
            u.conversion_times(conv(denom=2), conv(denom=3)),
            conv(denom=6))
        self.assertEqual(
            u.conversion_times(conv(exp10=2), conv(exp10=3)),
            conv(exp10=5))

        # All elements.
        self.assertEqual(
            u.conversion_times(
                conv(factor=3.14, numer=5, denom=12, exp10=2),
                conv(factor=2.71, numer=2, denom=15, exp10=-5)),
            conv(factor=3.14 * 2.71, numer=1, denom=18, exp10=-3))

    def testConversionDiv(self):
        # Identity.
        self.assertEqual(
            u.conversion_div(conv(), conv()),
            conv())

        # Single elements.
        self.assertEqual(
            u.conversion_div(conv(factor=3), conv(factor=2)),
            conv(factor=1.5))
        self.assertEqual(
            u.conversion_div(conv(numer=2), conv(numer=3)),
            conv(numer=2, denom=3))
        self.assertEqual(
            u.conversion_div(conv(denom=2), conv(denom=3)),
            conv(denom=2, numer=3))
        self.assertEqual(
            u.conversion_div(conv(exp10=2), conv(exp10=3)),
            conv(exp10=-1))

        # All elements.
        self.assertEqual(
            u.conversion_div(
                conv(factor=3.14, numer=5, denom=12, exp10=2),
                conv(factor=2.71, numer=2, denom=15, exp10=-5)),
            conv(factor=3.14 / 2.71, numer=25, denom=8, exp10=7))

    def testInverseConversion(self):
        # Identity.
        self.assertEqual(u.inverse_conversion(conv()), conv())

        # Single elements.
        self.assertEqual(
            u.inverse_conversion(conv(factor=2)),
            conv(factor=0.5))
        self.assertEqual(
            u.inverse_conversion(conv(numer=3)),
            conv(denom=3))
        self.assertEqual(
            u.inverse_conversion(conv(denom=5)),
            conv(numer=5))
        self.assertEqual(
            u.inverse_conversion(conv(exp10=7)),
            conv(exp10=-7))

        # All elements.
        self.assertEqual(
            u.inverse_conversion(conv(factor=3.14, numer=5, denom=12, exp10=2)),
            conv(factor=1 / 3.14, numer=12, denom=5, exp10=-2))

    def testConversionRaiseTo(self):
        # Identity.
        self.assertEqual(
            u.conversion_raise_to(conv(), frac()),
            conv())
        self.assertEqual(
            u.conversion_raise_to(conv(2, 3, 4, 5), frac()),
            conv(2, 3, 4, 5))
        self.assertEqual(
            u.conversion_raise_to(conv(), frac(500, 33)),
            conv())

        # Factor.
        self.assertEqual(
            u.conversion_raise_to(conv(factor=3), frac(2)),
            conv(factor=9))
        self.assertEqual(
            u.conversion_raise_to(conv(factor=4), frac(-2)),
            conv(factor=0.0625))
        self.assertEqual(
            u.conversion_raise_to(conv(factor=3), frac(denom=2)),
            conv(factor=1.7320508075688772))
        self.assertEqual(
            u.conversion_raise_to(conv(factor=3), frac(5, 9)),
            conv(factor=1.8410575470987482))
        self.assertEqual(
            u.conversion_raise_to(conv(factor=3), frac(-5, 9)),
            conv(factor=0.5431660740729487))

        # Numer.
        self.assertEqual(
            u.conversion_raise_to(conv(numer=3), frac(2)),
            conv(numer=9))
        self.assertEqual(
            u.conversion_raise_to(conv(numer=3), frac(-2)),
            conv(denom=9))
        self.assertEqual(
            u.conversion_raise_to(conv(numer=4), frac(denom=2)),
            conv(numer=2))
        self.assertEqual(
            u.conversion_raise_to(conv(numer=512), frac(5, 9)),
            conv(numer=32))
        # Lose precision.
        self.assertEqual(
            u.conversion_raise_to(conv(numer=3), frac(denom=2)),
            conv(factor=1.7320508075688772))
        self.assertEqual(
            u.conversion_raise_to(conv(numer=512), frac(5, 11)),
            conv(factor=17.040657431039403))
        self.assertEqual(
            u.conversion_raise_to(conv(numer=512), frac(-5, 11)),
            conv(factor=0.058683181916356644))

        # Denom.
        self.assertEqual(
            u.conversion_raise_to(conv(denom=3), frac(2)),
            conv(denom=9))
        self.assertEqual(
            u.conversion_raise_to(conv(denom=3), frac(-2)),
            conv(numer=9))
        self.assertEqual(
            u.conversion_raise_to(conv(denom=4), frac(denom=2)),
            conv(denom=2))
        self.assertEqual(
            u.conversion_raise_to(conv(denom=512), frac(5, 9)),
            conv(denom=32))
        # Lose precision.
        self.assertEqual(
            u.conversion_raise_to(conv(denom=3), frac(denom=2)),
            conv(factor=0.5773502691896258))
        self.assertEqual(
            u.conversion_raise_to(conv(denom=512), frac(5, 11)),
            conv(factor=0.058683181916356644))
        self.assertEqual(
            u.conversion_raise_to(conv(denom=512), frac(-5, 11)),
            conv(factor=17.040657431039403))

        # Exp10.
        self.assertEqual(
            u.conversion_raise_to(conv(exp10=6), frac(2)),
            conv(exp10=12))
        self.assertEqual(
            u.conversion_raise_to(conv(exp10=3), frac(-2)),
            conv(exp10=-6))
        self.assertEqual(
            u.conversion_raise_to(conv(exp10=6), frac(denom=2)),
            conv(exp10=3))
        self.assertEqual(
            u.conversion_raise_to(conv(exp10=18), frac(5, 9)),
            conv(exp10=10))
        # Lose precision.
        self.assertEqual(
            u.conversion_raise_to(conv(exp10=3), frac(denom=2)),
            conv(factor=31.622776601683793))
        self.assertEqual(
            u.conversion_raise_to(conv(exp10=18), frac(5, 11)),
            conv(factor=151991108.2952933))
        self.assertEqual(
            u.conversion_raise_to(conv(exp10=18), frac(-5, 11)),
            conv(factor=6.579332246575682e-9))

if __name__ == "__main__":
    unittest.main()
