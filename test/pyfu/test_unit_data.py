import math

import numpy as np


def test_all_default_units_and_simple_variations_thereof_are_parseable():
    import pyfu.unit
    db = pyfu.unit.default_unit_database
    for k, u in db.known_units.items():
        assert db.parse_unit_formula(k) == u
        for v in [u, 1 / u, 5 * u, 1.1 * u, u**2]:
            assert db.parse_unit_formula(str(v)) == v

def test_unit_relationship_energy_stored_in_capacity():
    from pyfu.units import uF, V, uJ
    capacitance = 2 * uF
    voltage = 5 * V
    stored = capacitance * voltage**2 / 2
    assert stored == 25 * uJ

def test_durations():
    from pyfu.units import week, year, day, hour, minute, second
    a = week + year + day + hour + minute
    assert a.isCompatible(second)
    assert round(year / week) == 52
    assert np.isclose(year / second, 31557600)

def test_lengths():
    from pyfu.units import inch, foot, yard, nautical_mile, angstrom, meter
    a = inch + foot + yard + nautical_mile + angstrom
    assert a.isCompatible(meter)
    assert (foot + inch + yard) * 5000 == 6223 * meter
    assert np.isclose(nautical_mile / angstrom, 1.852e13)

def test_angles():
    from pyfu.units import deg, rad, cyc
    assert (deg + cyc).isCompatible(rad)
    assert np.isclose((math.pi * rad)[deg], 180)
    assert np.isclose((math.pi * rad)[cyc], 0.5)

def test_volumes():
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
    assert a.isCompatible(liter)
    assert np.isclose(british_gallon / us_gallon, 1.20095, atol=1e-6)
    assert quart - pint - cup - tablespoon == 45 * teaspoon
    assert np.isclose(33.814 * fluid_ounce / liter, 1, atol=1e-5)

def test_masses():
    from pyfu.units import (ounce,
                            pound,
                            ton,
                            megagram)
    assert np.isclose((ounce + pound + ton) / megagram, 0.9077, atol=1e-4)
