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

"""
Defines named physical units, specified in terms of other units, exposed in the
default unit database and as members of tunits.units (and
tunits.api.like_pylabrad_units).

Derived units are defined by conversion parameters and unit formulas. The
formulas can mention base units and earlier derived units (but not later derived
units or physical constants).
"""

from typing import Any

from attrs import frozen

import math
import numpy as np


@frozen
class DerivedUnitData:
    """Describes the properties of a derived unit.

    Attributes:
        symbol: The short name for the unit (e.g. 'm' for meter).
        name: A full name for the unit (e.g. 'meter').
        formula: A formula defining the unit in terms of others.
        value: A floating-point scale factor.
        exp10: An integer power-of-10 exponent scale factor.
        numerator: A small integer scale factor.
        denominator: A small integer inverse scale factor.
        use_prefixes: Should there be 'kiloUNIT', 'milliUNIT', etc.
    """

    symbol: str
    name: str | None
    formula: str
    value: int | float | complex | np.number[Any] = 1.0
    exp10: int = 0
    numerator: int = 1
    denominator: int = 1
    use_prefixes: bool = False


__SI_REDUNDANT_BASE_UNITS = [
    DerivedUnitData('sr', 'steradian', 'rad^2', use_prefixes=True),
]

__SI_DERIVED_UNITS = [
    DerivedUnitData('Hz', 'hertz', '1/s', use_prefixes=True),
    DerivedUnitData('N', 'newton', 'kg*m/s^2', use_prefixes=True),
    DerivedUnitData('Pa', 'pascal', 'N/m^2', use_prefixes=True),
    DerivedUnitData('J', 'joule', 'N*m', use_prefixes=True),
    DerivedUnitData('W', 'watt', 'J/s', use_prefixes=True),
    DerivedUnitData('C', 'coulomb', 'A*s', use_prefixes=True),
    DerivedUnitData('V', 'volt', 'W/A', use_prefixes=True),
    DerivedUnitData('F', 'farad', 'C/V', use_prefixes=True),
    DerivedUnitData('Ohm', 'ohm', 'V/A', use_prefixes=True),
    DerivedUnitData('S', 'siemens', 'A/V', use_prefixes=True),
    DerivedUnitData('Wb', 'weber', 'V*s', use_prefixes=True),
    DerivedUnitData('T', 'tesla', 'Wb/m^2', use_prefixes=True),
    DerivedUnitData('Gauss', 'gauss', 'T', exp10=-4, use_prefixes=True),
    DerivedUnitData('H', 'henry', 'Wb/A', use_prefixes=True),
    DerivedUnitData('lm', 'lumen', 'cd*sr', use_prefixes=True),
    DerivedUnitData('lx', 'lux', 'lm/m^2', use_prefixes=True),
    DerivedUnitData('Bq', 'becquerel', 'Hz', use_prefixes=True),
    DerivedUnitData('l', 'liter', 'm^3', exp10=-3, use_prefixes=True),
    DerivedUnitData('phi0', 'magnetic_flux_quantum', 'J*s/C', value=2.067833831170082e-15, use_prefixes=True),
    DerivedUnitData('eV', 'electron_volt', 'N*m', value=1.602176634e-19, use_prefixes=False),
]

__OTHER_UNITS = [
    # Lengths.
    DerivedUnitData('in', 'inch', 'cm', 2, numerator=127, exp10=-2),
    DerivedUnitData('ft', 'foot', 'in', 4, numerator=3),
    DerivedUnitData('yd', 'yard', 'ft', numerator=3),
    DerivedUnitData('nmi', 'nautical_mile', 'm', 1852),
    DerivedUnitData('Ang', 'angstrom', 'm', exp10=-10),
    DerivedUnitData('ly', 'light_year', 'm', 94607304725808, exp10=2),
    DerivedUnitData('lyr', None, 'ly'),
    # Durations.
    DerivedUnitData('h', 'hour', 's', 4, numerator=9, exp10=2),
    DerivedUnitData('min', 'minute', 's', 2, numerator=3, exp10=1),
    # Angles.
    DerivedUnitData('cyc', 'cycle', 'rad', 2 * math.pi),
    DerivedUnitData('deg', None, 'rad', math.pi / 180),
    # Areas.
    DerivedUnitData('ha', 'hectare', 'm^2', exp10=4),
    DerivedUnitData('b', 'barn', 'm^2', exp10=-28),
    # Volumes.
    DerivedUnitData('tsp', 'teaspoon', 'ml', 4.92892159375),
    DerivedUnitData('tbsp', 'tablespoon', 'tsp', numerator=3),
    DerivedUnitData('floz', 'fluid_ounce', 'tbsp', 2),
    DerivedUnitData('cup', None, 'floz', 8),
    DerivedUnitData('pint', None, 'floz', 16),
    DerivedUnitData('qt', 'quart', 'pint', 2),
    DerivedUnitData('galUS', 'us_gallon', 'qt', 4),
    DerivedUnitData('galUK', 'british_gallon', 'l', 4.54609),
    # Mass.
    DerivedUnitData('oz', 'ounce', 'g', 28.349523125),
    DerivedUnitData('lb', 'pound', 'oz', 16),
    DerivedUnitData('ton', None, 'lb', 2000),
    # Pressure.
    DerivedUnitData('psi', 'pounds_per_square_inch', 'Pa', 6894.75729317),
    DerivedUnitData('bar', 'barometric_pressure', 'Pa', 1e5),
]

# Units that aren't technically exact, but close enough for our purposes.
__APPROXIMATE_CIVIL_UNITS = [
    DerivedUnitData('d', 'day', 's', 32, numerator=27, exp10=2),
    DerivedUnitData('wk', 'week', 'day', numerator=7),
    DerivedUnitData('yr', 'year', 'day', 365.25),
]

ALL_DERIVED_UNITS = (
    __SI_REDUNDANT_BASE_UNITS + __SI_DERIVED_UNITS + __OTHER_UNITS + __APPROXIMATE_CIVIL_UNITS
)
