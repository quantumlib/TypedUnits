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
import pytest

import numpy as np

from tunits import units
import tunits.core as core


def test_all_default_units_and_simple_variations_thereof_are_parseable() -> None:
    """
    parsing unit formulas recovers that same default unit
    even after doing basic arithmetic, the printed expression
    remains parseable and parses to the given value
    """
    db = core.default_unit_database
    for k, u in db.known_units.items():
        assert db.parse_unit_formula(k) == u
        for v in [u, 1 / u, 5 * u, 1.1 * u, u**2, 1e-20 * u]:
            assert db.parse_unit_formula(str(v)) == v


def test_unit_relationship_energy_stored_in_capacity() -> None:
    """
    unit relationship of volts, farads and joules
    """
    capacitance = 2 * units.uF
    voltage = 5 * units.V
    stored = capacitance * voltage**2 / 2
    assert stored == 25 * units.uJ


def test_durations() -> None:
    """
    time units are compatible
    - can be added to give another time
    - explicit conversion factors for some of the specified durations
    - can divide times and get something unitless
    """
    a = units.week + units.year + units.day + units.hour + units.minute
    assert a.is_compatible(units.second)
    assert round(units.year / units.week) == 52
    assert np.isclose(units.year / units.second, 31557600)


def test_lengths() -> None:
    """
    length units are compatible
    - can be added to give another length
    - explicit conversion factors for some of the specified lengths (mix of imperial and SI)
    - can divide lengths and get something unitless
    """
    a = (
        units.inch
        + units.foot
        + units.yard
        + units.nautical_mile
        + units.angstrom
        + units.light_year
    )
    assert a.is_compatible(units.meter)
    assert (units.foot + units.inch + units.yard) * 5000 == 6223 * units.meter
    assert np.isclose(units.nautical_mile / units.angstrom, 1.852e13)

    assert units.light_year == core.Value(1, 'c*yr')


def test_areas() -> None:
    """
    area units are compatible
    - can be added to give another area
    - can compare areas even if they have different units
    """
    assert (units.hectare + units.barn).is_compatible(units.meter**2)

    # *Obviously* a hectare of land can hold a couple barns.
    assert units.hectare > units.barn * 2
    # But not *too* many. ;)
    assert units.hectare < units.barn * 10**33


def test_angles() -> None:
    """
    angle units are compatible
    - can be added to give another angle
    - explicit conversion factors for the different units
    """
    assert (units.deg + units.cyc).is_compatible(units.rad)
    assert np.isclose((math.pi * units.rad)[units.deg], 180)
    assert np.isclose((math.pi * units.rad)[units.cyc], 0.5)


def test_volumes() -> None:
    """
    volume units are compatible
    - can be added to give another volume
    - explicit conversion factors for some of the specified volumes (mix of imperial and SI)
    - can divide volumes and get something unitless
    """
    a = (
        units.teaspoon
        + units.tablespoon
        + units.fluid_ounce
        + units.cup
        + units.pint
        + units.quart
        + units.us_gallon
        + units.british_gallon
    )
    assert a.is_compatible(units.liter)
    assert np.isclose(units.british_gallon / units.us_gallon, 1.20095, atol=1e-6)
    assert units.quart - units.pint - units.cup - units.tablespoon == 45 * units.teaspoon
    assert np.isclose(33.814 * units.fluid_ounce / units.liter, 1, atol=1e-5)


def test_masses() -> None:
    """
    mass units are compatible
    - can be added to give another mass
    - explicit conversion factors for some of the specified masses (mix of imperial and SI)
    - can divide masses and get something unitless
    """
    assert np.isclose((units.ounce + units.pound + units.ton) / units.megagram, 0.9077, atol=1e-4)


def test_pressures() -> None:
    assert units.psi.is_compatible(units.Pa)


def test_basic_constants() -> None:
    """
    Just some random products compared against results from Wolfram Alpha.
    """

    u = units.c * units.mu0 * units.eps0 * units.G * units.hplanck
    v = 1.475e-52 * units.m**4 / units.s**2
    assert np.isclose(u / v, 1, atol=1e-3)

    u = units.hbar * units.e * units.me * units.mp * units.Nav * units.k
    v = 2.14046e-109 * units.kg**4 * units.m**4 * units.A / (units.s**2 * units.K * units.mol)
    assert np.isclose(u / v, 1, atol=1e-3)


@pytest.mark.parametrize(
    ['lhs', 'rhs', 'ratio'],
    [
        [
            units.Hartree * units.rootHz * units.amu,
            units.kg**2 * units.m**2 / units.s**2.5,
            7.239526e-45,
        ],
        [
            units.bohr_magneton * units.Bohr * units.degR,
            units.m**3 * units.A * units.K,
            2.7264415e-34,
        ],
    ],
)
def test_other_constants(lhs: core.Value, rhs: core.Value, ratio: float) -> None:
    """
    more physically relevant unitful constants with parameterized pytest
    """
    r = lhs / rhs
    assert np.isclose(r, ratio, atol=1e-3)
