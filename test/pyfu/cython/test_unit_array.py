import pickle
import unittest
from pyfu import DimensionlessUnit as du
from pyfu._all_cythonized import raw_UnitArray, UnitArray


class UnitArrayTests(unittest.TestCase):
    def testConstructionVersusItems(self):
        empty = UnitArray()
        self.assertEqual(len(empty), 0)
        self.assertEqual(list(empty), [])

        singleton = UnitArray('arbitrary')
        self.assertEqual(len(singleton), 1)
        self.assertEqual(singleton[0], ('arbitrary', 1, 1))
        self.assertEqual(list(singleton), [('arbitrary', 1, 1)])

        self.assertRaises(TypeError, lambda: raw_UnitArray(1))
        self.assertRaises(TypeError, lambda: raw_UnitArray((2, 'a', 'c')))

        raw0 = raw_UnitArray([])
        self.assertEqual(len(raw0), 0)
        self.assertEqual(list(raw0), [])

        raw1 = raw_UnitArray([('a', 2, 3)])
        self.assertEqual(len(raw1), 1)
        self.assertEqual(raw1[0], ('a', 2, 3))
        self.assertEqual(list(raw1), [('a', 2, 3)])

        raw2 = raw_UnitArray([('a', 3, 7), ('b', 6, 15)])
        self.assertEqual(len(raw2), 2)
        self.assertEqual(raw2[0], ('a', 3, 7))
        self.assertEqual(raw2[1], ('b', 2, 5))
        self.assertEqual(list(raw2), [('a', 3, 7), ('b', 2, 5)])

    def testRepr(self):
        self.assertEqual(repr(du), 'raw_UnitArray([])')
        self.assertEqual(repr(UnitArray('a')), "raw_UnitArray([('a', 1, 1)])")

        self.assertEqual(
            repr(raw_UnitArray([])),
            "raw_UnitArray([])")
        self.assertEqual(
            repr(raw_UnitArray([('a', 2, 3)])),
            "raw_UnitArray([('a', 2, 3)])")
        self.assertEqual(
            repr(raw_UnitArray([('a', 2, 3), ('b', -5, 7)])),
            "raw_UnitArray([('a', 2, 3), ('b', -5, 7)])")

    def testStr(self):
        self.assertEqual(str(du), '')
        self.assertEqual(str(UnitArray('a')), 'a')

        self.assertEqual(
            str(raw_UnitArray([('b', -1, 1)])),
            '1/b')
        self.assertEqual(
            str(raw_UnitArray([('a', 2, 3), ('b', -5, 7)])),
            'a^(2/3)/b^(5/7)')
        self.assertEqual(
            str(raw_UnitArray([
                ('a', 1, 1), ('b', -1, 1), ('c', 1, 1), ('d', -1, 1)])),
            'a*c/b/d')
        self.assertEqual(
            str(raw_UnitArray([
                ('a', 2, 1), ('b', -1, 2), ('c', 1, 1), ('d', -1, 1)])),
            'a^2*c/b^(1/2)/d')

    def testEquality(self):
        equivalence_groups = [
            [0],
            [[]],
            [""],
            ["other types"],
            [UnitArrayTests],
            [None],

            [du, UnitArray(), raw_UnitArray([])],
            [UnitArray('a'), raw_UnitArray([('a', 1, 1)])],
            [raw_UnitArray([('a', 2, 1)]), raw_UnitArray([('a', 6, 3)])],
            [raw_UnitArray([('b', 2, 1)]), raw_UnitArray([('b', -6, -3)])],
            [raw_UnitArray([('b', -2, 1)]), raw_UnitArray([('b', 2, -1)])],
            [raw_UnitArray([('a', 2, 1), ('a', 2, 1)])],
            [raw_UnitArray([('a', 2, 1), ('b', 2, 1)])],
            [raw_UnitArray([('b', 2, 1), ('a', 2, 1)])],
            [raw_UnitArray([('a', 1, 1), ('b', 1, 1), ('c', 1, 1)])]*2,
        ]
        for g1 in equivalence_groups:
            for g2 in equivalence_groups:
                for e1 in g1:
                    for e2 in g2:
                        match = g1 is g2
                        if match:
                            self.assertEqual(e1, e2)
                        else:
                            self.assertNotEqual(e1, e2)
                        self.assertEqual(e1 == e2, match)
                        self.assertEqual(e1 != e2, not match)

    def testMultiplicativeIdentity(self):
        various = [
            UnitArray('a'),
            raw_UnitArray([('a', 2, 3), ('b', 1, 1)]),
            du
        ]
        for e in various:
            self.assertEqual(du * e, e)
            self.assertEqual(e * du, e)
            self.assertEqual(e / du, e)

    def testMultiplication(self):
        self.assertEqual(UnitArray('a') * UnitArray('b'),
                         raw_UnitArray([('a', 1, 1), ('b', 1, 1)]))
        self.assertEqual(UnitArray('b') * UnitArray('a'),
                         raw_UnitArray([('a', 1, 1), ('b', 1, 1)]))
        self.assertEqual(
            raw_UnitArray([('a', 2, 7)]) * raw_UnitArray([('a', 3, 5)]),
            raw_UnitArray([('a', 31, 35)]))
        self.assertEqual(
            raw_UnitArray([('a', 1, 1), ('b', 3, 5)]) * UnitArray('b'),
            raw_UnitArray([('a', 1, 1), ('b', 8, 5)]))
        self.assertEqual(
            raw_UnitArray([('b', -3, 5), ('a', 1, 1)]) * UnitArray('b'),
            raw_UnitArray([('b', 2, 5), ('a', 1, 1)]))

    def testDivision(self):
        self.assertEqual(du / UnitArray('b'),
                         raw_UnitArray([('b', -1, 1)]))
        self.assertEqual(UnitArray('a') / UnitArray('b'),
                         raw_UnitArray([('a', 1, 1), ('b', -1, 1)]))
        self.assertEqual(UnitArray('b') / UnitArray('a'),
                         raw_UnitArray([('a', -1, 1), ('b', 1, 1)]))
        self.assertEqual(
            raw_UnitArray([('a', 2, 7)]) / raw_UnitArray([('a', 3, 5)]),
            raw_UnitArray([('a', -11, 35)]))
        self.assertEqual(
            raw_UnitArray([('a', 1, 1), ('b', 3, 5)]) / UnitArray('b'),
            raw_UnitArray([('a', 1, 1), ('b', -2, 5)]))
        self.assertEqual(
            raw_UnitArray([('b', -3, 5), ('a', 1, 1)]) / UnitArray('b'),
            raw_UnitArray([('b', -8, 5), ('a', 1, 1)]))

    def testPow(self):
        self.assertEqual(du**2, du)
        self.assertEqual(UnitArray('a')**0, du)
        self.assertEqual(UnitArray('a')**2, raw_UnitArray([('a', 2, 1)]))
        self.assertEqual(UnitArray('a')**-1, raw_UnitArray([('a', -1, 1)]))
        self.assertEqual(UnitArray('a')**(1.0/3), raw_UnitArray([('a', 1, 3)]))
        self.assertEqual(UnitArray('a')**(7.0/12), raw_UnitArray([('a', 7, 12)]))
        self.assertEqual(UnitArray('a')**(1.0/12), raw_UnitArray([('a', 1, 12)]))

        self.assertEqual(raw_UnitArray([('a', 2, 3), ('b', -5, 7)])**(37.0/12),
                         raw_UnitArray([('a', 37, 18), ('b', -5*37, 7*12)]))

    def testPickling(self):
        examples = [
            du,
            raw_UnitArray([('a', 2, 7)]),
            raw_UnitArray([('a', 2, 7), ('b', 1, 3)])
        ]
        for e in examples:
            self.assertEqual(e, pickle.loads(pickle.dumps(e)))

if __name__ == "__main__":
    unittest.main()
