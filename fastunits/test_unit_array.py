#!/usr/bin/python
import unittest
import unit_array

class UnitsArrayTests(unittest.TestCase):
    def testConstruction(self):
        x = unit_array.UnitArray('km')
        y = unit_array.UnitArray('m')
        self.assertEquals(repr(x/y), {'km': 1, 's': -1})

def perf_unit_array(N=10000):
    x = unit_array.UnitArray('km')
    y = unit_array.UnitArray('m')
    for j in range(N):
        z = x*y
    
if __name__ == "__main__":
    unittest.main()

