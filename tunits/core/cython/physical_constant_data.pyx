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
Defines named physical constants exposed in the default unit database and as
members of tunits.units (and tunits.api.like_pylabrad_units).

Constants differ from derived units in that they don't affect the displayed
units of values. For example, `str(2 * c)` will return '599584916.0 m/s' instead
of '2.0 c'.

Constants are defined by scaling factors and unit formulas. The formulas can
mention base units, derived units, and earlier constants (but not later
constants).
"""
from attrs import frozen

from math import pi


@frozen
class PhysicalConstantData:
    """Describes a physical constant.

    Attributes:
        symbol: The constant's symbol (e.g. 'c' for speed of light).
        name: The constant's name (e.g. 'speed_of_light').
        factor: A scaling factor on the constant's unit value.
        formula: The constant's unit value in string form.
    """

    symbol: str
    name: str | None
    factor: float
    formula: str


_PHYSICAL_CONSTANTS = [
    PhysicalConstantData('c', 'speed_of_light', 299792458, 'm/s'),
    PhysicalConstantData('mu0', 'vacuum_permeability', 4.0e-7 * pi, 'N/A^2'),
    PhysicalConstantData('eps0', 'vacuum_permittivity', 1, '1/mu0/c^2'),
    PhysicalConstantData('G', 'gravitational_constant', 6.67259e-11, 'm^3/kg/s^2'),
    PhysicalConstantData('hplanck', 'planck_constant', 6.62606957e-34, 'J*s'),
    PhysicalConstantData('hbar', 'reduced_planck_constant', 0.5 / pi, 'hplanck'),
    PhysicalConstantData('e', 'elementary_charge', 1.60217733e-19, 'C'),
    PhysicalConstantData('me', 'electron_mass', 9.1093897e-31, 'kg'),
    PhysicalConstantData('mp', 'proton_mass', 1.6726231e-27, 'kg'),
    PhysicalConstantData('Nav', 'avogadro_constant', 6.0221367e23, '1/mol'),
    PhysicalConstantData('k', 'boltzmann_constant', 1.380658e-23, 'J/K'),
]

_DERIVED_CONSTANTS = [
    PhysicalConstantData('Bohr', 'bohr_radius', 4 * pi, 'eps0*hbar^2/me/e^2'),
    # Wavenumbers/inverse cm
    PhysicalConstantData('Hartree', None, 1.0 / 16 / pi**2, 'me*e^4/eps0^2/hbar^2'),
    PhysicalConstantData('rootHz', 'sqrtHz', 1, 'Hz^(1/2)'),
    PhysicalConstantData('amu', 'atomic_mass_unit', 1.6605402e-27, 'kg'),
    # degrees Rankine
    PhysicalConstantData('degR', None, 5.0 / 9.0, 'K'),
    PhysicalConstantData('bohr_magneton', None, 9.2740096820e-24, 'J/T'),
    PhysicalConstantData('R_k', 'resistance_quantum', 25812.78277321444, 'ohm'),  # h/e^2
]

ALL_PHYSICAL_CONSTANT_DATA = _PHYSICAL_CONSTANTS + _DERIVED_CONSTANTS
