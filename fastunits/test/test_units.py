# Copyright (C) 2007  Matthew Neeley
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy as np

import sys
import os
import cPickle
import fastunits as fu
ValueArray = fu.ValueArray
Value = fu.Value


class LabradUnitsTests(unittest.TestCase):
    def testParsing(self):
        # prefixes
        # multiplication
        # division
        # powers
        pass

    def testArithmetic(self):
        m = fu.Unit('m')
        kg = fu.Unit('kg')
        km = fu.Unit('km')

        self.assertEqual(fu.Value(5.0, None)*m, 5.0*m)

        # addition
        self.assertEqual(1.0*kg + 0.0*kg, 1.0*kg)
        with self.assertRaises(fu.UnitMismatchError): _ = 1.0*kg + 1.0*m
        with self.assertRaises(fu.UnitMismatchError): _ = 1.0*kg + 2.0
        self.assertAlmostEqual(1.0*km/m + 5.0, 1005)
        self.assertNotEqual(1.0*kg, None)

    def testValueArray(self):
        # Slicing
        self.assertTrue((ValueArray([1, 2, 3], 'm')[0:2] == ValueArray([1, 2], 'm')).all())
        # Cast to unit
        self.assertTrue((ValueArray([1.2, 4, 5], 'm')['m'] == np.array([1.2, 4, 5])).all())
        # Addition and subtraction of compatible units
        self.assertTrue((ValueArray([3, 4], 'm') + ValueArray([100, 200], 'cm') ==
                         ValueArray([4, 6], 'm')).all())
        self.assertTrue((ValueArray([2, 3, 4], 'm') - ValueArray([100, 200, 300], 'cm') ==
                         ValueArray([1, 1, 1], 'm')).all())
        # Division with units remaining
        self.assertTrue((ValueArray([3, 4, 5], 'm') / ValueArray([1, 2, 5], 's') ==
                         ValueArray([3, 2, 1], 'm/s')).all())
        # Division with no units remaining
        self.assertTrue((ValueArray([3, 4, 5], 'm') / ValueArray([1, 2, 5], 'm') ==
                         ValueArray([3, 2, 1], '')).all())
        # Powers
        self.assertTrue((ValueArray([2, 3], 'm')**2 == ValueArray([4, 9], 'm^2')).all())

        self.assertTrue((ValueArray([2, 3], 'GHz') * Value(3, 'ns')).dtype == np.float64)

    def testIsFinite(self):
        self.assertTrue(np.isfinite(ValueArray([1, 2], '')).all())
        self.assertTrue((np.isfinite(ValueArray([1, float('nan')], '')) ==
                         np.array([True, False])).all())

    def testNegativePowers(self):
        self.assertIn(str(fu.Unit('1/s')), ['s^-1', '1/s'])
        self.assertIn(str(fu.Unit('1/s^1/2')), ['s^-1/2', '1/s^(1/2)'])

    def testTypeConversions(self):
        m = fu.Unit('m')
        V = fu.Unit('V')
        GHz = fu.Unit('GHz')
        x1 = 1.0*m
        x2 = 5j*V
        a = np.arange(10)*1.0
        va = fu.ValueArray(np.arange(10)*1.0, 'GHz')

        # Unit times number
        self.assertIsInstance(1.0*m, fu.Value)
        self.assertIsInstance(1*m, fu.Value)
        self.assertIsInstance(m*1.0, fu.Value)
        self.assertIsInstance(m*1, fu.Value)

        # Value times value or number
        self.assertIsInstance(x1*x1, fu.Value)
        self.assertIsInstance(x1*5, fu.Value)
        self.assertIsInstance(0*x1, fu.Value)

        # Unit times complex
        self.assertIsInstance((1+1j)*V, fu.Complex)
        self.assertIsInstance(V*(1+1j), fu.Complex)

        # Value times Complex/complex
        self.assertIsInstance(x1*1j, fu.Complex)
        self.assertIsInstance(1j*x1, fu.Complex)
        self.assertIsInstance(x2*x1, fu.Complex)
        self.assertIsInstance(x1*x2, fu.Complex)

        # Unit/Value/ValueArray times array
        self.assertIsInstance(x1*a, fu.ValueArray)
        self.assertIsInstance(x2*a, fu.ValueArray)
        self.assertIsInstance(GHz*a, fu.ValueArray)
        self.assertIsInstance(va*a, fu.ValueArray)

        # Unit/Value/ValueArray times ValueArray
        self.assertIsInstance(x1*va, fu.ValueArray)
        self.assertIsInstance(x2*va, fu.ValueArray)
        self.assertIsInstance(GHz*va, fu.ValueArray)
        self.assertIsInstance(va*va, fu.ValueArray)

        # array times ?
        self.assertIsInstance(a*x1, fu.ValueArray)
        self.assertIsInstance(a*x2, fu.ValueArray)
        self.assertIsInstance(a*GHz, fu.ValueArray)
        self.assertIsInstance(a*va, fu.ValueArray)
        self.assertIsInstance(va*va, fu.ValueArray)

        # values
        self.assertEquals((a*x1)[2], 2*m)
        self.assertEquals((a*x2)[2], 10j*V)
        self.assertEquals((a*GHz)[2], 2*GHz)
        self.assertEquals((a*(GHz*GHz))[2], 2*GHz*GHz)
        self.assertEquals(((GHz*GHz)*a)[2], 2*GHz*GHz)
        self.assertEquals((a*va)[2], 4*GHz)
        self.assertEquals((va*va)[2], 4*GHz*GHz)

        # ValueArray times ?
        self.assertIsInstance(va*x1, fu.ValueArray)
        self.assertIsInstance(va*x2, fu.ValueArray)
        self.assertIsInstance(va*GHz, fu.ValueArray)
        self.assertIsInstance(va*a, fu.ValueArray)

    def testComparison(self):
        s = fu.Unit('s')
        ms = fu.Unit('ms')
        kg = fu.Unit('kg')
        self.assertTrue(1*s > 10*ms, '1*s > 10*ms')
        self.assertTrue(1*s >= 10*ms, '1*s >= 10*ms')
        self.assertTrue(1*s < 10000*ms, '1*s > 10000*ms')
        self.assertTrue(1*s <= 10000*ms, '1*s >= 10000*ms')
        self.assertTrue(10*ms < 1*s, '10*ms < 1*s')
        self.assertTrue(10*ms <= 1*s, '10*ms <= 1*s')
        self.assertTrue(10000*ms > 1*s, '10000*ms < 1*s')
        self.assertTrue(10000*ms >= 1*s, '10000*ms <= 1*s')
        with self.assertRaises(TypeError):
            nogood = 1*s > 1*kg

        self.assertFalse(1*s == 1*kg)
        self.assertTrue(0*s == 0*ms)
        self.assertTrue(4*s > 0*s)
        with self.assertRaises(TypeError): _ = 4*s > 1

    def testComplex(self):
        V = fu.Unit('V')

        self.assertTrue(1j*V != 1.0*V)
        self.assertTrue(1j*V == 1.0j*V)
        self.assertTrue(1.0*V == (1+0j)*V)
        with self.assertRaises(TypeError): _ = 1.0j*V < 2j*V

    def testDimensionless(self):
        ns = fu.Unit('ns')
        GHz = fu.Unit('GHz')

        self.assertEquals(float((5*ns)*(5*GHz)), 25.0)
        self.assertTrue(hasattr((5*ns)*(5*GHz), 'inUnitsOf'))
        self.assertTrue(((5*ns)*(5*GHz)).isDimensionless())
        self.assertTrue((5*ns)*(5*GHz) < 50)
        self.assertIsInstance(fu.WithUnit(1, ''), fu.WithUnit)
        self.assertIsInstance(5.0*fu.WithUnit(1, ''), fu.Value)

        self.assertTrue((5*ns*5j*GHz) == 25j)
        self.assertTrue((5*ns*5j*GHz).isDimensionless())

    def testAngle(self):
        rad = fu.Unit('rad')
        self.assertTrue(rad.is_angle)
        self.assertTrue(rad.isAngle())
        x = fu.Unit('rad*m/s')
        self.assertFalse(x.is_angle)

    def testInfNan(self):
        ms = fu.Unit('ms')
        GHz = fu.Unit('GHz')
        MHz = fu.Unit('MHz')

        self.assertEquals(float('inf')*GHz, float('inf')*MHz)
        self.assertNotEqual(float('inf')*GHz, float('inf')*ms)
        self.assertNotEqual(float('inf')*GHz, -float('inf')*GHz)
        self.assertNotEqual(float('nan')*GHz, float('nan')*GHz)
        self.assertNotEqual(float('nan')*GHz, float('nan')*ms)

    def testPickling(self):
        ns = fu.Unit('ns')
        GHz = fu.Unit('GHz')
        blank = fu.Unit('')

        def round_trip(obj):
            return cPickle.loads(cPickle.dumps(obj))
        self.assertEqual(round_trip(5*GHz), 5*GHz) # Value
        self.assertEqual(round_trip(GHz), GHz)     # Unit
        self.assertTrue((round_trip(np.arange(5)*ns) == np.arange(5)*ns).all()) # array
        self.assertEqual(round_trip(5*GHz*ns), 5)  # Dimensionless
        self.assertIsInstance(round_trip(3*blank), type(3*blank)) # Don't loose dimensionless type

    def testUnitCreation(self):
        # Unit creation is different in fastuntis, need to fix this
        pass
        #test0 = fu.Unit._new_derived_unit('test0', fu.hplanck/(2*fu.e))
        #self.assertIsInstance(test0, fu.Unit)
        #self.assertTrue((fu.Unit('phi0')**2).isCompatible(fu.Unit('phi0^2')))

    def testInUnitsOf(self):
        s = fu.Unit('s')
        ms = fu.Unit('ms')
        self.assertTrue((1*s).inUnitsOf(ms) == 1000*ms)
        self.assertTrue((1*s).inUnitsOf('ms') == 1000*ms)

    def testBaseUnitPowers(self):
        x = Value(1, 'ns^2')

        self.assertTrue(x.unit.base_unit == fu.Unit('s^2'))
        self.assertTrue(x.inBaseUnits() == Value(1e-18, 's^2'))

    def testUnitPowers(self):
        self.assertTrue(fu.Unit('ns')**2 == fu.Unit('ns^2'))

    def test_array_priority(self):
        """numpy issue 6133

        DimensionlessX needs to support all arithmetic operations when the
        other side is an array.  Numpy's __array_priority__ machinery doesn't
        handle NotImplemented results correctly, so the higher priority element
        *must* be able to handle all operations.

        In numpy 1.9 this becomes more critical because numpy scalars like np.float64
        get converted to arrays before being passed to binary arithmetic operations.
        """
        x = np.float64(1)
        y = fu.Value(2)
        self.assertTrue(x < y)
        z = np.arange(5)
        self.assertTrue(((x<z) == [False, False, True, True, True]).all())

    def testNone(self):
        with self.assertRaises(Exception):
            fu.Unit(None)
        with self.assertRaises(TypeError):
            None * fu.Unit('MHz')

    def test_non_SI(self):
        fu.addNonSI('count', True)
        x = 5 * fu.Unit('kcount')
        self.assertTrue(x['count'] == 5000.0)
        self.assertTrue(x.inBaseUnits() == 5000.0*fu.Unit('count'))
        self.assertTrue((x**2).unit == fu.Unit('kcount^2'))

    def test_string_unit(self):
        ts = fu.Unit('tshirt/s')
        self.assertEqual((1*ts)['tshirt/h'], 3600.0)
        self.assertEqual(str(ts), 'tshirt/s')

    def testIter(self):
        data = np.arange(5) * fu.ns
        for x in data:
            self.assertIsInstance(x, fu.Value)
        with self.assertRaises(TypeError):
            for _ in 5*fu.kg:
                pass

    def testIsCompatible(self):
        x = 5*fu.ns
        self.assertTrue(x.isCompatible('s'))
        self.assertFalse(x.isCompatible(fu.kg))
        self.assertTrue(fu.ns.isCompatible(fu.s))
        self.assertTrue(fu.ns.isCompatible(fu.ns))
        self.assertFalse(fu.ns.isCompatible(fu.kg))
        with self.assertRaises(Exception):
            x.isCompatible(4)

    def testScaledGetItem(self):
        from fastunits import ns, s
        v = s*1.0
        self.assertEquals(v[ns], 10**9)
        self.assertEquals(v[ns*2], 10**9/2)
        self.assertEquals((v*3)[(ns*3)], 10 ** 9)

if __name__ == "__main__":
    unittest.main()
