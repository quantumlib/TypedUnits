import math
import unittest


class UnitDataTests(unittest.TestCase):
    def testEnergyStoredInCapacitor(self):
        from pyfu.units import uF, V, uJ
        capacitance = 2 * uF
        voltage = 5 * V
        stored = capacitance * voltage**2 / 2
        self.assertEqual(stored, 25 * uJ)

    def testDurations(self):
        from pyfu.units import week, year, day, hour, minute, second
        a = week + year + day + hour + minute
        self.assertTrue(a.isCompatible(second))
        self.assertEqual(round(year / week), 52)
        self.assertAlmostEqual(year / second, 31557600)

    def testLengths(self):
        from pyfu.units import inch, foot, yard, nautical_mile, angstrom, meter
        a = inch + foot + yard + nautical_mile + angstrom
        self.assertTrue(a.isCompatible(meter))
        self.assertEqual((foot + inch + yard) * 5000, 6223 * meter)
        self.assertAlmostEqual(nautical_mile / angstrom, 1.852e13)

    def testAngles(self):
        from pyfu.units import deg, rad, cyc
        self.assertTrue((deg + cyc).isCompatible(rad))
        self.assertAlmostEqual((math.pi * rad)[deg], 180)
        self.assertAlmostEqual((math.pi * rad)[cyc], 0.5)

    def testVolumes(self):
        from pyfu.units import (teaspoon,
                                tablespoon,
                                fluid_ounce,
                                cup,
                                pint,
                                quart,
                                us_gallon,
                                british_gallon,
                                liter)
        a = (teaspoon + tablespoon + fluid_ounce + cup + pint + quart +
             us_gallon + british_gallon)
        self.assertTrue(a.isCompatible(liter))
        self.assertAlmostEqual(british_gallon / us_gallon, 1.20095, places=6)
        self.assertEqual(quart - pint - cup - tablespoon, 45 * teaspoon)
        self.assertAlmostEqual(33.814 * fluid_ounce / liter, 1, places=5)


if __name__ == "__main__":
    unittest.main()
