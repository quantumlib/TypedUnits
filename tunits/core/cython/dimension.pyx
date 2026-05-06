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

"""Dimension abstraction.

This module creates the dimension abstraction which allows the creation of values
that belong to a dimension (e.g. t: Time, x: Length, ...etc). This allows us to
use static types to check code correctness (e.g. time_method(t: Time)).

To add a new dimension, create 3 classes:
- `class _NewDimension(Dimension):` which implements the abstract methods from
    the abstract `Dimension` class.
- `class NewDimension(_NewDimension, ValueWithDimension)` which represents scalar
    values and doesn't need to implement any methods.
- `class AccelerationArray(_Acceleration, ArrayWithDimension)` which represents
    an array of values sharing the same dimension and
"""

import abc

from functools import cache


class Dimension(abc.ABC):
    """Dimension abstraction.

    This abstract class allows the creation of values that belong to a dimension
    (e.g. t: Time, x: Length, ...etc). This allows us to use static types to check
    code correctness (e.g. time_method(t: Time)).

    To add a new dimension, create 3 classes:
        - `class _NewDimension(Dimension):` which implements the abstract methods from
            this class.
        - `class NewDimension(_NewDimension, ValueWithDimension)` which represents scalar
            values and doesn't need to implement any methods.
        - `class AccelerationArray(_Acceleration, ArrayWithDimension)` which represents
            an array of values sharing the same dimension and unit.
    """

    @staticmethod
    def valid_base_units() -> tuple[Value, ...]:
        """Returns a tuple of valid base units (e.g. (dB, dBm) for LogPower)."""

    @classmethod
    def is_valid(cls, v: WithUnit) -> bool:
        return any(v.base_units == u.base_units for u in cls.valid_base_units())


class _Acceleration(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (
            default_unit_database.known_units['m'] / default_unit_database.known_units['s'] ** 2,
        )

    def _value_class(self) -> type[Value]:
        return Acceleration

    def _array_class(self) -> type[ValueArray]:
        return AccelerationArray


class ValueWithDimension(Dimension, Value):
    def __init__(self, val, unit=None, validate: bool = True):
        super().__init__(val, unit=unit)
        if validate and not type(self).is_valid(self):
            raise ValueError(f'{self.unit} is not a valid unit for dimension {type(self)}')


class ArrayWithDimension(Dimension, ValueArray):
    def __init__(self, val, unit=None, validate: bool = True):
        super().__init__(val, unit=unit)
        if validate and not type(self).is_valid(self):
            raise ValueError(f'{self.unit} is not a valid unit for dimension {type(self)}')


class Acceleration(_Acceleration, ValueWithDimension):
    """A scalar value representing acceleration. Standard unit: meters per second squared (m/s²)."""


class AccelerationArray(_Acceleration, ArrayWithDimension):
    """An array of acceleration values. Standard unit: meters per second squared (m/s²)."""


class _Angle(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (
            default_unit_database.known_units['rad'],
            default_unit_database.known_units['sr'],
        )

    def _value_class(self) -> type[Value]:
        return Angle

    def _array_class(self) -> type[ValueArray]:
        return AngleArray


class Angle(_Angle, ValueWithDimension):
    """A scalar value representing a planar or solid angle. Base units: radian (rad), steradian (sr)."""


class AngleArray(_Angle, ArrayWithDimension):
    """An array of angle values. Base units: radian (rad), steradian (sr)."""


class _AngularFrequency(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (
            default_unit_database.known_units['rad'] * default_unit_database.known_units['Hz'] * 2,
        )

    def _value_class(self) -> type[Value]:
        return AngularFrequency

    def _array_class(self) -> type[ValueArray]:
        return AngularFrequencyArray


class AngularFrequency(_AngularFrequency, ValueWithDimension):
    """A scalar value representing angular frequency. Standard unit: radians per second (rad/s)."""


class AngularFrequencyArray(_AngularFrequency, ArrayWithDimension):
    """An array of angular frequency values. Standard unit: radians per second (rad/s)."""


class _Area(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['m'] ** 2,)

    def _value_class(self) -> type[Value]:
        return Area

    def _array_class(self) -> type[ValueArray]:
        return AreaArray


class Area(_Area, ValueWithDimension):
    """A scalar value representing area. Standard unit: square meter (m²)."""


class AreaArray(_Area, ArrayWithDimension):
    """An array of area values. Standard unit: square meter (m²)."""


class _Capacitance(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['farad'],)

    def _value_class(self) -> type[Value]:
        return Capacitance

    def _array_class(self) -> type[ValueArray]:
        return CapacitanceArray


class Capacitance(_Capacitance, ValueWithDimension):
    """A scalar value representing electrical capacitance. Standard unit: farad (F)."""


class CapacitanceArray(_Capacitance, ArrayWithDimension):
    """An array of capacitance values. Standard unit: farad (F)."""


class _Charge(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['coulomb'],)

    def _value_class(self) -> type[Value]:
        return Charge

    def _array_class(self) -> type[ValueArray]:
        return ChargeArray


class Charge(_Charge, ValueWithDimension):
    """A scalar value representing electrical charge. Standard unit: coulomb (C)."""


class ChargeArray(_Charge, ArrayWithDimension):
    """An array of charge values. Standard unit: coulomb (C)."""


class _CurrentDensity(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (
            default_unit_database.known_units['ampere']
            / default_unit_database.known_units['m'] ** 2,
        )

    def _value_class(self) -> type[Value]:
        return CurrentDensity

    def _array_class(self) -> type[ValueArray]:
        return CurrentDensityArray


class CurrentDensity(_CurrentDensity, ValueWithDimension):
    """A scalar value representing electrical current density. Standard unit: amperes per square meter (A/m²)."""


class CurrentDensityArray(_CurrentDensity, ArrayWithDimension):
    """An array of current density values. Standard unit: amperes per square meter (A/m²)."""


class _Density(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (
            default_unit_database.known_units['kg'] / default_unit_database.known_units['m'] ** 3,
        )

    def _value_class(self) -> type[Value]:
        return Density

    def _array_class(self) -> type[ValueArray]:
        return DensityArray


class Density(_Density, ValueWithDimension):
    """A scalar value representing mass density. Standard unit: kilograms per cubic meter (kg/m³)."""


class DensityArray(_Density, ArrayWithDimension):
    """An array of density values. Standard unit: kilograms per cubic meter (kg/m³)."""


class _ElectricCurrent(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['ampere'],)

    def _value_class(self) -> type[Value]:
        return ElectricCurrent

    def _array_class(self) -> type[ValueArray]:
        return ElectricCurrentArray


class ElectricCurrent(_ElectricCurrent, ValueWithDimension):
    """A scalar value representing electrical current. Standard unit: ampere (A)."""


class ElectricCurrentArray(_ElectricCurrent, ArrayWithDimension):
    """An array of electrical current values. Standard unit: ampere (A)."""


class _ElectricPotential(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['V'],)

    def _value_class(self) -> type[Value]:
        return ElectricPotential

    def _array_class(self) -> type[ValueArray]:
        return ElectricPotentialArray


class ElectricPotential(_ElectricPotential, ValueWithDimension):
    """A scalar value representing electrical potential (voltage). Standard unit: volt (V)."""


class ElectricPotentialArray(_ElectricPotential, ArrayWithDimension):
    """An array of electrical potential values. Standard unit: volt (V)."""


class _ElectricalConductance(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['siemens'],)

    def _value_class(self) -> type[Value]:
        return ElectricalConductance

    def _array_class(self) -> type[ValueArray]:
        return ElectricalConductanceArray


class ElectricalConductance(_ElectricalConductance, ValueWithDimension):
    """A scalar value representing electrical conductance. Standard unit: siemens (S)."""


class ElectricalConductanceArray(_ElectricalConductance, ArrayWithDimension):
    """An array of electrical conductance values. Standard unit: siemens (S)."""


class _Energy(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['joule'],)

    def _value_class(self) -> type[Value]:
        return Energy

    def _array_class(self) -> type[ValueArray]:
        return EnergyArray


class Energy(_Energy, ValueWithDimension):
    """A scalar value representing energy. Standard unit: joule (J)."""


class EnergyArray(_Energy, ArrayWithDimension):
    """An array of energy values. Standard unit: joule (J)."""


class _Force(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['newton'],)

    def _value_class(self) -> type[Value]:
        return Force

    def _array_class(self) -> type[ValueArray]:
        return ForceArray


class Force(_Force, ValueWithDimension):
    """A scalar value representing force. Standard unit: newton (N)."""


class ForceArray(_Force, ArrayWithDimension):
    """An array of force values. Standard unit: newton (N)."""


class _Frequency(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['Hz'],)

    def _value_class(self) -> type[Value]:
        return Frequency

    def _array_class(self) -> type[ValueArray]:
        return FrequencyArray


class Frequency(_Frequency, ValueWithDimension):
    """A scalar value representing frequency. Standard unit: hertz (Hz)."""


class FrequencyArray(_Frequency, ArrayWithDimension):
    """An array of frequency values. Standard unit: hertz (Hz)."""


class _Illuminance(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['lux'],)

    def _value_class(self) -> type[Value]:
        return Illuminance

    def _array_class(self) -> type[ValueArray]:
        return IlluminanceArray


class Illuminance(_Illuminance, ValueWithDimension):
    """A scalar value representing illuminance. Standard unit: lux (lx)."""


class IlluminanceArray(_Illuminance, ArrayWithDimension):
    """An array of illuminance values. Standard unit: lux (lx)."""


class _Inductance(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['henry'],)

    def _value_class(self) -> type[Value]:
        return Inductance

    def _array_class(self) -> type[ValueArray]:
        return InductanceArray


class Inductance(_Inductance, ValueWithDimension):
    """A scalar value representing electrical inductance. Standard unit: henry (H)."""


class InductanceArray(_Inductance, ArrayWithDimension):
    """An array of inductance values. Standard unit: henry (H)."""


class _Length(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['m'],)

    def _value_class(self) -> type[Value]:
        return Length

    def _array_class(self) -> type[ValueArray]:
        return LengthArray


class Length(_Length, ValueWithDimension):
    """A scalar value representing length or distance. Standard unit: meter (m)."""


class LengthArray(_Length, ArrayWithDimension):
    """An array of length values. Standard unit: meter (m)."""


class _LogPower(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (
            default_unit_database.known_units['dBm'],
            default_unit_database.known_units['dB'],
        )

    def _value_class(self) -> type[Value]:
        return LogPower

    def _array_class(self) -> type[ValueArray]:
        return LogPowerArray


class LogPower(_LogPower, ValueWithDimension):
    """A scalar value representing logarithmic power or power ratio. Supported units: decibel-milliwatts (dBm), decibel (dB)."""


class LogPowerArray(_LogPower, ArrayWithDimension):
    """An array of logarithmic power values. Supported units: decibel-milliwatts (dBm), decibel (dB)."""


class _LuminousFlux(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['lumen'],)

    def _value_class(self) -> type[Value]:
        return LuminousFlux

    def _array_class(self) -> type[ValueArray]:
        return LuminousFluxArray


class LuminousFlux(_LuminousFlux, ValueWithDimension):
    """A scalar value representing luminous flux. Standard unit: lumen (lm)."""


class LuminousFluxArray(_LuminousFlux, ArrayWithDimension):
    """An array of luminous flux values. Standard unit: lumen (lm)."""


class _LuminousIntensity(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['candela'],)

    def _value_class(self) -> type[Value]:
        return LuminousIntensity

    def _array_class(self) -> type[ValueArray]:
        return LuminousIntensityArray


class LuminousIntensity(_LuminousIntensity, ValueWithDimension):
    """A scalar value representing luminous intensity. Standard unit: candela (cd)."""


class LuminousIntensityArray(_LuminousIntensity, ArrayWithDimension):
    """An array of luminous intensity values. Standard unit: candela (cd)."""


class _MagneticFlux(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['weber'],)

    def _value_class(self) -> type[Value]:
        return MagneticFlux

    def _array_class(self) -> type[ValueArray]:
        return MagneticFluxArray


class MagneticFlux(_MagneticFlux, ValueWithDimension):
    """A scalar value representing magnetic flux. Standard unit: weber (Wb)."""


class MagneticFluxArray(_MagneticFlux, ArrayWithDimension):
    """An array of magnetic flux values. Standard unit: weber (Wb)."""


class _MagneticFluxDensity(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['tesla'],)

    def _value_class(self) -> type[Value]:
        return MagneticFluxDensity

    def _array_class(self) -> type[ValueArray]:
        return MagneticFluxDensityArray


class MagneticFluxDensity(_MagneticFluxDensity, ValueWithDimension):
    """A scalar value representing magnetic flux density. Standard unit: tesla (T)."""


class MagneticFluxDensityArray(_MagneticFluxDensity, ArrayWithDimension):
    """An array of magnetic flux density values. Standard unit: tesla (T)."""


class _Mass(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['kg'],)

    def _value_class(self) -> type[Value]:
        return Mass

    def _array_class(self) -> type[ValueArray]:
        return MassArray


class Mass(_Mass, ValueWithDimension):
    """A scalar value representing mass. Standard unit: kilogram (kg)."""


class MassArray(_Mass, ArrayWithDimension):
    """An array of mass values. Standard unit: kilogram (kg)."""


class _Noise(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (
            default_unit_database.known_units['V'] / default_unit_database.known_units['Hz'] ** 0.5,
            default_unit_database.known_units['watt'] / default_unit_database.known_units['Hz'],
        )

    def _value_class(self) -> type[Value]:
        return Noise

    def _array_class(self) -> type[ValueArray]:
        return NoiseArray


class Noise(_Noise, ValueWithDimension):
    """A scalar value representing spectral noise density. Base units: V/√Hz, W/Hz."""


class NoiseArray(_Noise, ArrayWithDimension):
    """An array of spectral noise density values. Base units: V/√Hz, W/Hz."""


class _Power(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['watt'],)

    def _value_class(self) -> type[Value]:
        return Power

    def _array_class(self) -> type[ValueArray]:
        return PowerArray


class Power(_Power, ValueWithDimension):
    """A scalar value representing power. Standard unit: watt (W)."""


class PowerArray(_Power, ArrayWithDimension):
    """An array of power values. Standard unit: watt (W)."""


class _Pressure(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['pascal'],)

    def _value_class(self) -> type[Value]:
        return Pressure

    def _array_class(self) -> type[ValueArray]:
        return PressureArray


class Pressure(_Pressure, ValueWithDimension):
    """A scalar value representing pressure. Standard unit: pascal (Pa)."""


class PressureArray(_Pressure, ArrayWithDimension):
    """An array of pressure values. Standard unit: pascal (Pa)."""


class _Quantity(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['mole'],)

    def _value_class(self) -> type[Value]:
        return Quantity

    def _array_class(self) -> type[ValueArray]:
        return QuantityArray


class Quantity(_Quantity, ValueWithDimension):
    """A scalar value representing amount of substance. Standard unit: mole (mol)."""


class QuantityArray(_Quantity, ArrayWithDimension):
    """An array of quantity values. Standard unit: mole (mol)."""


class _Resistance(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['ohm'],)

    def _value_class(self) -> type[Value]:
        return Resistance

    def _array_class(self) -> type[ValueArray]:
        return ResistanceArray


class Resistance(_Resistance, ValueWithDimension):
    """A scalar value representing electrical resistance. Standard unit: ohm (Ω)."""


class ResistanceArray(_Resistance, ArrayWithDimension):
    """An array of resistance values. Standard unit: ohm (Ω)."""


class _Speed(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['m'] / default_unit_database.known_units['s'],)

    def _value_class(self) -> type[Value]:
        return Speed

    def _array_class(self) -> type[ValueArray]:
        return SpeedArray


class Speed(_Speed, ValueWithDimension):
    """A scalar value representing speed or velocity. Standard unit: meters per second (m/s)."""


class SpeedArray(_Speed, ArrayWithDimension):
    """An array of speed values. Standard unit: meters per second (m/s)."""


class _SurfaceDensity(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (
            default_unit_database.known_units['kg'] / default_unit_database.known_units['m'] ** 2,
        )

    def _value_class(self) -> type[Value]:
        return SurfaceDensity

    def _array_class(self) -> type[ValueArray]:
        return SurfaceDensityArray


class SurfaceDensity(_SurfaceDensity, ValueWithDimension):
    """A scalar value representing surface density. Standard unit: kilograms per square meter (kg/m²)."""


class SurfaceDensityArray(_SurfaceDensity, ArrayWithDimension):
    """An array of surface density values. Standard unit: kilograms per square meter (kg/m²)."""


class _Temperature(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (
            default_unit_database.known_units['kelvin'],
            default_unit_database.known_units['celsius'],
            default_unit_database.known_units['fahrenheit'],
        )

    def _value_class(self) -> type[Value]:
        return Temperature

    def _array_class(self) -> type[ValueArray]:
        return TemperatureArray


class Temperature(_Temperature, ValueWithDimension):
    """A scalar value representing temperature. Supported units: Kelvin (K), Celsius (°C), Fahrenheit (°F)."""


class TemperatureArray(_Temperature, ArrayWithDimension):
    """An array of temperature values. Supported units: Kelvin (K), Celsius (°C), Fahrenheit (°F)."""


class _Time(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['s'],)

    def _value_class(self) -> type[Value]:
        return Time

    def _array_class(self) -> type[ValueArray]:
        return TimeArray


class Time(_Time, ValueWithDimension):
    """A scalar value representing time duration. Standard unit: second (s)."""


class TimeArray(_Time, ArrayWithDimension):
    """An array of time duration values. Standard unit: second (s)."""


class _Torque(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (
            default_unit_database.known_units['newton'] * default_unit_database.known_units['m'],
        )

    def _value_class(self) -> type[Value]:
        return Torque

    def _array_class(self) -> type[ValueArray]:
        return TorqueArray


class Torque(_Torque, ValueWithDimension):
    """A scalar value representing torque. Standard unit: newton-meter (N·m)."""


class TorqueArray(_Torque, ArrayWithDimension):
    """An array of torque values. Standard unit: newton-meter (N·m)."""


class _Volume(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['m'] ** 3,)

    def _value_class(self) -> type[Value]:
        return Volume

    def _array_class(self) -> type[ValueArray]:
        return VolumeArray


class Volume(_Volume, ValueWithDimension):
    """A scalar value representing volume. Standard unit: cubic meter (m³)."""


class VolumeArray(_Volume, ArrayWithDimension):
    """An array of volume values. Standard unit: cubic meter (m³)."""


class _WaveNumber(Dimension):

    @staticmethod
    @cache
    def valid_base_units() -> tuple[Value, ...]:
        return (default_unit_database.known_units['m'] ** -1,)

    def _value_class(self) -> type[Value]:
        return WaveNumber

    def _array_class(self) -> type[ValueArray]:
        return WaveNumberArray


class WaveNumber(_WaveNumber, ValueWithDimension):
    """A scalar value representing wavenumber. Standard unit: reciprocal meter (m⁻¹)."""


class WaveNumberArray(_WaveNumber, ArrayWithDimension):
    """An array of wavenumber values. Standard unit: reciprocal meter (m⁻¹)."""
