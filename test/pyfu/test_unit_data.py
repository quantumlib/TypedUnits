import unittest


class UnitDataTests(unittest.TestCase):
    def testEnergyStoredInCapacitor(self):
        from pyfu.units import uF, V, uJ
        capacitance = 2 * uF
        voltage = 5 * V
        stored = capacitance * voltage**2 / 2
        self.assertEqual(stored, 25 * uJ)


if __name__ == "__main__":
    unittest.main()
