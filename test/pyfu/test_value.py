import unittest
from pyfu import Value, Unit, Complex, UnitMismatchError
import numpy as np
from pyfu.units import kilometer, meter, mm, second, us, ns


class ValueTests(unittest.TestCase):
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
        x = Value(1.0 + 2j, meter)
        y = Value(3, mm)
        a = Value(20, second)
        self.assertEqual(a * x, x * a)
        self.assertTrue((x / y).isDimensionless())

    def testPower(self):
        from pyfu.units import km, m, minute, s, um
        x = 2 * mm
        y = 4 * mm
        z = (x * y)**.5
        self.assertLess(abs(z**2 - Value(8, 'mm^2')), Value(1e-6, mm**2))
        self.assertEqual(Value(16000000, 'um^2') ** 0.5, 4 * mm)
        self.assertEqual((16 * um * m)**0.5, 4 * mm)
        self.assertEqual((minute**2) ** 0.5, minute)
        self.assertEqual((1000 * m * km)**0.5, km)
        self.assertEqual((60 * s * minute) ** 0.5, minute)

    def testRepr(self):
        from pyfu.units import km, kg
        self.assertEqual(repr(Value(1, mm)), "Value(1.0, 'mm')")
        self.assertEqual(repr(Value(4, mm)), "Value(4.0, 'mm')")
        self.assertEqual(repr(Value(1j+5, km * kg)), "Value((5+1j), 'kg*km')")

    def testStr(self):
        self.assertEqual(str(Value(1, mm)), 'mm')
        self.assertEqual(str(Value(4, mm)), '4.0 mm')
        self.assertEqual(str(2 * meter * kilometer), '2.0 km*m')

    def testDivmod(self):
        x = 4.0009765625 * us
        self.assertEquals(x // (4 * ns), 1000)
        self.assertEquals(x % (4 * ns), 0.9765625 * ns)
        q, r = divmod(x, 2 * ns)
        self.assertEquals(q, 2000)
        self.assertEquals(r, x - 4 * us)

    def testConversion(self):
        x = Value(3, 'm')
        self.assertEquals(x['mm'], 3000.0)
        with self.assertRaises(UnitMismatchError):
            _ = x['s']
        y = Value(1000, 'Mg')
        self.assertEquals(y.inBaseUnits().value, 1000000.0)
        self.assertEquals(x.inUnitsOf('mm'), 3000 * mm)

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

    def testDivision(self):
        from pyfu.units import km, s, m

        self.assertEqual(5 * km / (2 * s), Value(2500, 'm/s'))
        with self.assertRaises(UnitMismatchError):
            _ = 5 * km // (2 * s)
        self.assertEqual((5 * km).__div__(2 * s), Value(2500, 'm/s'))
        self.assertEqual((5 * km).__truediv__(2 * s), Value(2500, 'm/s'))
        with self.assertRaises(UnitMismatchError):
            self.assertEqual((5 * km).__floordiv__(2 * s), Value(2500, 'm/s'))

        self.assertEqual((5 * km) / (64 * m), 78.125)
        self.assertEqual((5 * km) // (64 * m), 78)
        self.assertEqual((5 * km).__div__(64 * m), 78.125)
        self.assertEqual((5 * km).__truediv__(64 * m), 78.125)
        self.assertEqual((5 * km).__floordiv__(64 * m), 78)

    def testCycles(self):
        from pyfu.units import cyc, rad
        self.assertAlmostEquals((3.14159265*rad)[cyc], 0.5)
        self.assertAlmostEquals((1.0*rad)[cyc], 0.15915494309)
        self.assertAlmostEquals((1.0*cyc)[2*rad], 3.14159265)

    def testHash(self):
        x = Value(3, 'ks')
        y = Value(3000, 's')
        self.assertEquals(hash(x), hash(y))
        z = Value(3.1, '')
        self.assertEquals(hash(z), hash(3.1))
        hash(Value(4j, 'V'))

if __name__ == "__main__":
    unittest.main()
