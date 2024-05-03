# Copyright 2024 The TUnits Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np


def test_all_default_units_and_simple_variations_thereof_are_parseable() -> None:
    import tunits.api.unit

    db = tunits.api.unit.default_unit_database
    for k, u in db.known_units.items():
        assert db.parse_unit_formula(k) == u
        for v in [u, 1 / u, 5 * u, 1.1 * u, u**2]:
            assert db.parse_unit_formula(str(v)) == v


def test_unit_relationship_energy_stored_in_capacity() -> None:
    from tunits.units import uF, V, uJ

    capacitance = 2 * uF
    voltage = 5 * V
    stored = capacitance * voltage**2 / 2
    assert stored == 25 * uJ


def test_durations() -> None:
    from tunits.units import week, year, day, hour, minute, second

    a = week + year + day + hour + minute
    assert a.isCompatible(second)
    assert round(year / week) == 52
    assert np.isclose(year / second, 31557600)


def test_lengths() -> None:
    from tunits.units import inch, foot, yard, nautical_mile, angstrom, light_year, meter

    a = inch + foot + yard + nautical_mile + angstrom + light_year
    assert a.isCompatible(meter)
    assert (foot + inch + yard) * 5000 == 6223 * meter
    assert np.isclose(nautical_mile / angstrom, 1.852e13)
    from tunits import Value

    assert light_year == Value(1, 'c*yr')


def test_areas() -> None:
    from tunits.units import hectare, barn, meter

    assert (hectare + barn).isCompatible(meter**2)

    # *Obviously* a hectare of land can hold a couple barns.
    assert hectare > barn * 2
    # But not *too* many. ;)
    assert hectare < barn * 10**33


def test_angles() -> None:
    from tunits.units import deg, rad, cyc

    assert (deg + cyc).isCompatible(rad)
    assert np.isclose((math.pi * rad)[deg], 180)
    assert np.isclose((math.pi * rad)[cyc], 0.5)


def test_volumes() -> None:
    from tunits.units import (
        teaspoon,
        tablespoon,
        fluid_ounce,
        cup,
        pint,
        quart,
        us_gallon,
        british_gallon,
        liter,
    )

    a = teaspoon + tablespoon + fluid_ounce + cup + pint + quart + us_gallon + british_gallon
    assert a.isCompatible(liter)
    assert np.isclose(british_gallon / us_gallon, 1.20095, atol=1e-6)
    assert quart - pint - cup - tablespoon == 45 * teaspoon
    assert np.isclose(33.814 * fluid_ounce / liter, 1, atol=1e-5)


def test_masses() -> None:
    from tunits.units import ounce, pound, ton, megagram

    assert np.isclose((ounce + pound + ton) / megagram, 0.9077, atol=1e-4)


def test_pressures() -> None:
    from tunits.units import psi, Pa

    assert psi.isCompatible(Pa)


def test_basic_constants() -> None:
    from tunits.units import (
        c,
        mu0,
        eps0,
        G,
        hplanck,
        hbar,
        e,
        me,
        mp,
        Nav,
        k,
        m,
        s,
        kg,
        A,
        K,
        mol,
    )

    # Just some random products compared against results from Wolfram Alpha.

    u = c * mu0 * eps0 * G * hplanck
    v = 1.475e-52 * m**4 / s**2
    assert np.isclose(u / v, 1, atol=1e-3)

    u = hbar * e * me * mp * Nav * k
    v = 2.14046e-109 * kg**4 * m**4 * A / (s**2 * K * mol)
    assert np.isclose(u / v, 1, atol=1e-3)


def test_other_constants() -> None:
    from tunits.units import bohr_magneton, Bohr, degR, Hartree, rootHz, amu, kg, m, s, A, K

    u = Hartree * rootHz * amu
    v = 7.239526e-45 * kg**2 * m**2 / s ** (2.5)
    assert np.isclose(u / v, 1, atol=1e-3)

    u = bohr_magneton * Bohr * degR
    v = 2.7264415e-34 * m**3 * A * K
    assert np.isclose(u / v, 1, atol=1e-3)
