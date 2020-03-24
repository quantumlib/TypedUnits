"""
Defines named physical constants exposed in the default unit database and as
members of pyfu.units (and pyfu.like_pylabrad_units).

Constants differ from derived units in that they don't affect the displayed
units of values. For example, `str(2 * c)` will return '599584916.0 m/s' instead
of '2.0 c'.

Constants are defined by scaling factors and unit formulas. The formulas can
mention base units, derived units, and earlier constants (but not later
constants).
"""

from math import pi


class PhysicalConstantData:
    """
    Describes a physical constant.
    """
    def __init__(self,
                 symbol,
                 name,
                 factor,
                 formula):
        """
        :param str symbol: The constant's symbol (e.g. 'c' for speed of light).
        :param None|str name: The constant's name (e.g. 'speed_of_light').
        :param float factor: A scaling factor on the constant's unit value.
        :param str formula: The constant's unit value in string form.
        """
        self.symbol = symbol
        self.name = name
        self.factor = factor
        self.formula = formula

_data = PhysicalConstantData

_PHYSICAL_CONSTANTS = [
    _data('c', 'speed_of_light', 299792458, 'm/s'),
    _data('mu0', 'vacuum_permeability', 4.e-7 * pi, 'N/A^2'),
    _data('eps0', 'vacuum_permittivity', 1, '1/mu0/c^2'),
    _data('G', 'gravitational_constant', 6.67259e-11, 'm^3/kg/s^2'),
    _data('hplanck', 'planck_constant', 6.62606957e-34, 'J*s'),
    _data('hbar', 'reduced_planck_constant', 0.5 / pi, 'hplanck'),
    _data('e', 'elementary_charge', 1.60217733e-19, 'C'),
    _data('me', 'electron_mass', 9.1093897e-31, 'kg'),
    _data('mp', 'proton_mass', 1.6726231e-27, 'kg'),
    _data('Nav', 'avogadro_constant', 6.0221367e23, '1/mol'),
    _data('k', 'boltzmann_constant', 1.380658e-23, 'J/K'),
]

_DERIVED_CONSTANTS = [
    _data('Bohr', 'bohr_radius', 4 * pi, 'eps0*hbar^2/me/e^2'),
    # Wavenumbers/inverse cm
    _data('Hartree', None, 1.0 / 16 / pi ** 2, 'me*e^4/eps0^2/hbar^2'),
    _data('rootHz', 'sqrtHz', 1, 'Hz^(1/2)'),
    _data('amu', 'atomic_mass_unit', 1.6605402e-27, 'kg'),
    # degrees Rankine
    _data('degR', None, 5. / 9., 'K'),
    _data('bohr_magneton', None, 9.2740096820e-24, 'J/T'),
]

ALL_PHYSICAL_CONSTANT_DATA = _PHYSICAL_CONSTANTS + _DERIVED_CONSTANTS
