from typing import ClassVar, Sequence, Any, Callable, Iterator, overload, Generic, SupportsIndex
import abc
from attrs import frozen
from typing_extensions import TypeVar
from numpy.typing import NDArray, DTypeLike
import numpy as np

from tunits.proto import tunits_pb2

SCALE_PREFIXES: dict[str, int]

_NUMERICAL_TYPE = int | float | complex | np.number[Any]
_NUMERICAL_TYPE_OR_ARRAY = (
    int
    | float
    | np.number[Any]
    | complex
    | list[_NUMERICAL_TYPE]
    | tuple[_NUMERICAL_TYPE]
    | NDArray[Any]
)
_NUMERICAL_TYPE_OR_ARRAY_OR_UNIT = (
    int
    | float
    | np.number[Any]
    | complex
    | list[_NUMERICAL_TYPE]
    | tuple[_NUMERICAL_TYPE]
    | NDArray[Any]
    | 'WithUnit'
    | 'Value'
    | 'ValueArray'
)
T = TypeVar('T', bound=WithUnit)
_NUMERICAL_TYPE_OR_ARRAY_OR_GENERIC_UNIT = (
    int
    | float
    | np.number[Any]
    | complex
    | list[_NUMERICAL_TYPE]
    | tuple[_NUMERICAL_TYPE]
    | NDArray[Any]
    | 'WithUnit'
)

class UnitMismatchError(TypeError):
    """Incompatible units."""

class NotTUnitsLikeError(Exception):
    """The value is not a tunits object and can't be converted to one."""

frac = dict[str, int]

def float_to_twelths_frac(x: float) -> frac:
    """Recognizes floats corresponding to twelths. Returns them as a fraction."""

def frac_div(a: frac, b: frac) -> frac:
    """Returns the quotient of the two given fractions, in least terms."""

def frac_times(a: frac, b: frac) -> frac:
    """Returns the product of the two given fractions, in least terms."""

def frac_least_terms(a: int, b: int) -> frac:
    """
    Returns an equivalent fraction, without common factors between numerator and
    denominator and with the negative sign on the numerator (if present).
    """

def frac_to_double(a: frac) -> float:
    """Converts a fraction to a double approximating its value."""

def gcd(a: int, b: int) -> int: ...

class UnitTerm:
    name: Any
    power: frac

conversion = dict[str, Any]

def conversion_div(a: conversion, b: conversion) -> conversion:
    """
    Returns a conversion equivalent to applying one conversion and un-applying
    another.
    """

def inverse_conversion(a: conversion) -> conversion: ...
def conversion_times(a: conversion, b: conversion) -> conversion:
    """Returns a conversion equivalent to applying both given conversions."""

def conversion_raise_to(base: conversion, exponent: frac) -> conversion:
    """
    Returns a conversion that, if applied several times, would be roughly
    equivalent to the given conversion.

    Precision lose may be unavoidable when performing roots, but we do try to
    use exact results when possible.
    """

def conversion_to_double(c: conversion) -> float:
    """Returns a double that approximates the given conversion."""

def raw_UnitArray(args: Sequence[tuple[str, int, int]]) -> 'UnitArray':
    """
    A factory method the creates and directly sets the items of a UnitArray.
    (__init__ couldn't play this role for backwards-compatibility reasons.)

    :param list((name, power.numer, power.denom)) name_numer_denom_tuples:
        The list of properties that units in the resulting list should have.
    :return UnitArray:
    """

class UnitArray:
    """
    A list of physical units raised to various powers.
    """

    __pyx_vtable__: ClassVar[Any] = ...
    @classmethod
    def __init__(cls, name: str | None = None) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getitem__(self, index: int) -> tuple[str, int, int]: ...
    def __gt__(self, other: object) -> bool: ...
    def __iter__(self) -> Iterator[tuple[str, int, int]]: ...
    def __len__(self) -> int: ...
    def __mul__(self, other: UnitArray) -> UnitArray: ...
    def __pow__(self, exponent: Any, modulo: Any = None) -> UnitArray: ...
    def __truediv__(self, other: UnitArray) -> UnitArray: ...

NumericalT = TypeVar('NumericalT', default=float, bound=float | int)
ValueType = TypeVar('ValueType', bound='Value')
ValueType2 = TypeVar('ValueType2', default='Value', bound='Value')
ArrayType = TypeVar('ArrayType', bound='ValueArray')

@overload
def raw_WithUnit(
    value: _NUMERICAL_TYPE,
    conv: conversion | dict[str, Any],
    base_units: UnitArray,
    display_units: UnitArray,
    value_class: type[ValueType] | None = None,
    array_class: type[ArrayType] | None = None,
) -> ValueType: ...
@overload
def raw_WithUnit(
    value: list[Any] | tuple[Any] | NDArray[Any],
    conv: conversion | dict[str, Any],
    base_units: UnitArray,
    display_units: UnitArray,
    value_class: type[ValueType] | None = None,
    array_class: type[ArrayType] | None = None,
) -> ArrayType: ...
@overload
def raw_WithUnit(
    value: Any,
    conv: conversion | dict[str, Any],
    base_units: UnitArray,
    display_units: UnitArray,
    value_class: type[ValueType] | None = None,
    array_class: type[ArrayType] | None = None,
) -> ArrayType | ValueType:
    """
    A factory method that directly sets the properties of a WithUnit.
    (__init__ couldn't play this role for backwards-compatibility reasons.)
    """

class WithUnit:
    """
    A value with associated physical units.
    """

    __array_priority__: ClassVar[int] = ...
    __pyx_vtable__: ClassVar[Any] = ...
    base_units: UnitArray
    display_units: UnitArray
    factor: float
    is_angle: bool
    numer: int
    denom: int
    exp10: int
    unit: 'Value'
    value: float | complex | NDArray[Any]
    real: 'Value' | 'ValueArray'
    imag: 'Value' | 'ValueArray'
    is_dimensionless: bool
    units: str
    base_unit: 'Value'
    name: str

    def __init__(
        self, value: Any, unit: 'WithUnit' | UnitTerm | UnitArray | str | None = None
    ) -> None: ...
    def in_base_units(self: T) -> T: ...
    def in_units_of(self: T, unit: Any, should_round: bool = False) -> T: ...
    def is_compatible(self, other: Any) -> bool: ...
    # def floor(self: T, unit: Any) -> T: ...
    # def ceil(self: T, unit: Any) -> T: ...
    def sqrt(self: T) -> T: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __abs__(self: 'WithUnit') -> 'WithUnit': ...
    def __getitem__(self, key: Any) -> NDArray[Any] | float: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...
    def __neg__(self: T) -> T: ...
    def __pos__(self: T) -> T: ...
    def __lt__(self, other: _NUMERICAL_TYPE_OR_ARRAY_OR_GENERIC_UNIT) -> bool | NDArray[Any]: ...
    def __gt__(self, other: _NUMERICAL_TYPE_OR_ARRAY_OR_GENERIC_UNIT) -> bool | NDArray[Any]: ...
    def __le__(self, other: _NUMERICAL_TYPE_OR_ARRAY_OR_GENERIC_UNIT) -> bool | NDArray[Any]: ...
    def __ge__(self, other: _NUMERICAL_TYPE_OR_ARRAY_OR_GENERIC_UNIT) -> bool | NDArray[Any]: ...
    def __eq__(self, other: Any) -> bool | NDArray[Any]: ...  # type: ignore[override]
    def __neq__(self, other: Any) -> bool | NDArray[Any]: ...
    def __round__(self) -> int: ...
    def __divmod__(self, other: Any) -> tuple[_NUMERICAL_TYPE_OR_ARRAY, 'WithUnit']: ...
    def _value_class(self) -> type['Value']: ...
    def _array_class(self) -> type['ValueArray']: ...
    def _json_dict_(self) -> dict[str, Any]: ...
    @classmethod
    def _json_namespace_(cls) -> str: ...
    @classmethod
    def _from_json_dict_(cls: type[T], **kwargs: Any) -> T: ...

class Value(Generic[NumericalT], WithUnit, np.generic, SupportsIndex):
    """A floating-point value with associated units."""

    __array_interface__ = {'typestr': '<c16', 'shape': ()}
    __array_priority__ = -1000000
    value: NumericalT
    real: 'Value'
    imag: 'Value'

    @classmethod
    def from_proto(cls: type[T], msg: tunits_pb2.Value) -> T: ...
    def to_proto(self, msg: tunits_pb2.Value | None = None) -> tunits_pb2.Value: ...
    def __abs__(self: ValueType) -> ValueType: ...
    def __divmod__(self: ValueType, other: Any) -> tuple[_NUMERICAL_TYPE_OR_ARRAY, ValueType]: ...
    @overload
    def __add__(self: ValueType, other: _NUMERICAL_TYPE) -> ValueType: ...
    @overload
    def __add__(
        self: ValueType, other: ValueArray | list[Any] | tuple[Any] | NDArray[Any]
    ) -> ValueArray[ValueType]: ...
    @overload
    def __add__(self: T, other: T) -> T: ...
    @overload
    def __radd__(self: ValueType, other: int | float | complex) -> ValueType: ...
    @overload
    def __radd__(
        self: ValueType, other: list[Any] | tuple[Any] | NDArray[Any]
    ) -> ValueArray[ValueType]: ...
    @overload
    def __sub__(self: ValueType, other: _NUMERICAL_TYPE) -> ValueType: ...
    @overload
    def __sub__(
        self: ValueType, other: ValueArray | list[Any] | tuple[Any] | NDArray[Any]
    ) -> ValueArray[ValueType]: ...
    @overload
    def __sub__(self: T, other: T) -> T: ...
    @overload
    def __rsub__(self: ValueType, other: int | float | complex) -> ValueType: ...
    @overload
    def __rsub__(
        self: ValueType, other: list[Any] | tuple[Any] | NDArray[Any]
    ) -> ValueArray[ValueType]: ...
    @overload
    def __truediv__(self: ValueType, other: _NUMERICAL_TYPE) -> ValueType: ...
    @overload
    def __truediv__(self, other: list[Any] | tuple[Any] | NDArray[Any]) -> ValueArray: ...
    @overload
    def __truediv__(self, other: Value) -> Value: ...
    @overload
    def __truediv__(self, other: ValueArray) -> ValueArray: ...
    @overload
    def __rtruediv__(self, other: int | float | complex) -> Value: ...
    @overload
    def __rtruediv__(self, other: list[Any] | tuple[Any] | NDArray[Any]) -> ValueArray: ...
    @overload
    def __mul__(self: ValueType, other: _NUMERICAL_TYPE) -> ValueType: ...
    @overload
    def __mul__(self, other: list[Any] | tuple[Any] | NDArray[Any]) -> ValueArray: ...
    @overload
    def __mul__(self, other: Value) -> Value: ...
    @overload
    def __mul__(self, other: ValueArray) -> ValueArray: ...
    @overload
    def __rmul__(self: ValueType, other: int | float | complex) -> ValueType: ...
    @overload
    def __rmul__(self, other: list[Any] | tuple[Any] | NDArray[Any]) -> ValueArray: ...
    @overload
    def __mod__(self: ValueType, other: _NUMERICAL_TYPE) -> ValueType: ...
    @overload
    def __mod__(self, other: list[Any] | tuple[Any] | NDArray[Any]) -> ValueArray: ...
    @overload
    def __mod__(self, other: 'Value') -> Value: ...
    @overload
    def __mod__(self, other: 'ValueArray') -> ValueArray: ...
    @overload
    def __mod__(self, other: 'WithUnit') -> Any: ...
    @overload
    def __rmod__(self: ValueType, other: _NUMERICAL_TYPE) -> ValueType: ...
    @overload
    def __rmod__(self, other: list[Any] | tuple[Any] | NDArray[Any]) -> ValueArray: ...
    @overload
    def __floordiv__(self: ValueType, other: _NUMERICAL_TYPE) -> ValueType: ...
    @overload
    def __floordiv__(self, other: list[Any] | tuple[Any] | NDArray[Any]) -> ValueArray: ...
    @overload
    def __floordiv__(self, other: 'Value') -> Value: ...
    @overload
    def __floordiv__(self, other: 'ValueArray') -> ValueArray: ...
    @overload
    def __floordiv__(self, other: 'WithUnit') -> Any: ...
    def __rfloordiv__(self, other: list[Any] | tuple[Any] | NDArray[Any]) -> ValueArray: ...
    def __lt__(self, other: _NUMERICAL_TYPE_OR_ARRAY_OR_GENERIC_UNIT) -> bool: ...
    def __gt__(self, other: _NUMERICAL_TYPE_OR_ARRAY_OR_GENERIC_UNIT) -> bool: ...
    def __le__(self, other: _NUMERICAL_TYPE_OR_ARRAY_OR_GENERIC_UNIT) -> bool: ...
    def __ge__(self, other: _NUMERICAL_TYPE_OR_ARRAY_OR_GENERIC_UNIT) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...
    def __neq__(self, other: Any) -> bool: ...
    def __getitem__(self, key: Any) -> NumericalT: ...
    def value_in_base_units(self) -> NumericalT: ...
    def __index__(self) -> int: ...
    def __pow__(self, other: Any, modulus: Any = None) -> Value: ...
    def sign(self) -> int: ...
    def dimensionless(self) -> float: ...

class ValueArray(Generic[ValueType2], WithUnit):
    value: NDArray[Any]
    real: 'ValueArray'
    imag: 'ValueArray'

    @classmethod
    def from_proto(cls: type[T], msg: tunits_pb2.ValueArray) -> T: ...
    def to_proto(self, msg: tunits_pb2.ValueArray | None = None) -> tunits_pb2.ValueArray: ...
    def __init__(self, data: Any, unit: Any = None) -> None: ...
    def allclose(self, other: ValueArray | Value, *args: Any, **kwargs: float) -> bool: ...
    def __array__(self, dtype: DTypeLike = None) -> NDArray[Any]: ...
    def __array_wrap__(self, out_arr: NDArray[Any], context: Any = None) -> NDArray[Any]: ...
    def __copy__(self: ArrayType) -> ArrayType: ...
    def __deepcopy__(self: ArrayType) -> ArrayType: ...
    def __iter__(self) -> Iterator[ValueType2]: ...
    def __len__(self) -> int: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    @property
    def dtype(self) -> NDArray[Any]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    def __abs__(self: ArrayType) -> ArrayType: ...
    def __divmod__(self: ArrayType, other: Any) -> tuple[_NUMERICAL_TYPE_OR_ARRAY, ArrayType]: ...
    def __add__(self: ArrayType, other: Any) -> ArrayType: ...
    def __radd__(self: ArrayType, other: Any) -> ArrayType: ...
    def __sub__(self: ArrayType, other: Any) -> ArrayType: ...
    def __rsub__(self: ArrayType, other: Any) -> ArrayType: ...
    @overload
    def __truediv__(self: ArrayType, other: _NUMERICAL_TYPE_OR_ARRAY) -> ArrayType: ...
    @overload
    def __truediv__(self: ArrayType, other: WithUnit) -> ValueArray: ...
    def __rtruediv__(self: ArrayType, other: _NUMERICAL_TYPE_OR_ARRAY) -> ValueArray: ...
    def __floordiv__(self, other: Any) -> NDArray[Any]: ...
    @overload
    def __mul__(self: ArrayType, other: _NUMERICAL_TYPE_OR_ARRAY) -> ArrayType: ...
    @overload
    def __mul__(self: ArrayType, other: WithUnit) -> ValueArray: ...
    def __rmul__(self: ArrayType, other: _NUMERICAL_TYPE_OR_ARRAY) -> ArrayType: ...
    def __mod__(self: ArrayType, other: Any) -> ArrayType: ...
    def __rmod__(self: ArrayType, other: Any) -> ArrayType: ...
    def __lt__(self, other: _NUMERICAL_TYPE_OR_ARRAY_OR_GENERIC_UNIT) -> NDArray[Any]: ...
    def __gt__(self, other: _NUMERICAL_TYPE_OR_ARRAY_OR_GENERIC_UNIT) -> NDArray[Any]: ...
    def __le__(self, other: _NUMERICAL_TYPE_OR_ARRAY_OR_GENERIC_UNIT) -> NDArray[Any]: ...
    def __ge__(self, other: _NUMERICAL_TYPE_OR_ARRAY_OR_GENERIC_UNIT) -> NDArray[Any]: ...
    def __eq__(self, other: Any) -> NDArray[Any]: ...  # type: ignore[override]
    def __neq__(self, other: Any) -> NDArray[Any]: ...
    @overload
    def __getitem__(self, key: str | Value) -> NDArray[Any]: ...
    @overload
    def __getitem__(self, key: int | np.integer) -> ValueType2: ...
    @overload
    def __getitem__(self: ArrayType, key: slice | tuple[Any, ...] | NDArray[Any]) -> ArrayType: ...
    def value_in_base_units(self) -> NDArray[Any]: ...
    def __array_ufunc__(
        self, ufunc: Callable[..., Any], method: str, *inputs: Any, **kwargs: Any
    ) -> ValueArray: ...
    def __pow__(self, other: Any, modulus: Any = None) -> ValueArray: ...
    def sign(self) -> NDArray[np.integer]: ...
    def dimensionless(self) -> NDArray[Any]: ...

def init_base_unit_functions(
    try_interpret_as_with_unit: Callable[[Any, bool], WithUnit | None],
    is_value_consistent_with_default_unit_database: Callable[[Any], bool],
) -> None: ...

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
    @abc.abstractmethod
    def valid_base_units() -> tuple[Value, ...]:
        """Returns a tuple of valid base units (e.g. (dB, dBm) for LogPower)."""

    @classmethod
    def is_valid(cls, v: WithUnit) -> bool: ...

class ValueWithDimension(abc.ABC, Value):
    @staticmethod
    @abc.abstractmethod
    def valid_base_units() -> tuple[Value, ...]:
        """Returns a tuple of valid base units (e.g. (dB, dBm) for LogPower)."""

    @classmethod
    def is_valid(cls, v: WithUnit) -> bool: ...
    def __getitem__(self: ValueType2, unit: ValueType2 | str) -> float: ...

class ArrayWithDimension(abc.ABC, ValueArray[ValueType2]):
    @staticmethod
    @abc.abstractmethod
    def valid_base_units() -> tuple[Value, ...]:
        """Returns a tuple of valid base units (e.g. (dB, dBm) for LogPower)."""

    @classmethod
    def is_valid(cls, v: WithUnit) -> bool: ...
    def __iter__(self) -> Iterator[ValueType2]: ...

class _Acceleration(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Acceleration(_Acceleration, ValueWithDimension): ...
class AccelerationArray(_Acceleration, ArrayWithDimension[Acceleration]): ...

class _Angle(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Angle(_Angle, ValueWithDimension): ...
class AngleArray(_Angle, ArrayWithDimension[Angle]): ...

class _AngularFrequency(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class AngularFrequency(_AngularFrequency, ValueWithDimension): ...
class AngularFrequencyArray(_AngularFrequency, ArrayWithDimension[AngularFrequency]): ...

class _Area(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Area(_Area, ValueWithDimension): ...
class AreaArray(_Area, ArrayWithDimension[Area]): ...

class _Capacitance(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Capacitance(_Capacitance, ValueWithDimension): ...
class CapacitanceArray(_Capacitance, ArrayWithDimension[Capacitance]): ...

class _Charge(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Charge(_Charge, ValueWithDimension): ...
class ChargeArray(_Charge, ArrayWithDimension[Charge]): ...

class _CurrentDensity(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class CurrentDensity(_CurrentDensity, ValueWithDimension): ...
class CurrentDensityArray(_CurrentDensity, ArrayWithDimension[CurrentDensity]): ...

class _Density(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Density(_Density, ValueWithDimension): ...
class DensityArray(_Density, ArrayWithDimension[Density]): ...

class _ElectricCurrent(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class ElectricCurrent(_ElectricCurrent, ValueWithDimension): ...
class ElectricCurrentArray(_ElectricCurrent, ArrayWithDimension[ElectricCurrent]): ...

class _ElectricPotential(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class ElectricPotential(_ElectricPotential, ValueWithDimension): ...
class ElectricPotentialArray(_ElectricPotential, ArrayWithDimension[ElectricPotential]): ...

class _ElectricalConductance(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class ElectricalConductance(_ElectricalConductance, ValueWithDimension): ...
class ElectricalConductanceArray(
    _ElectricalConductance, ArrayWithDimension[ElectricalConductance]
): ...

class _Energy(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Energy(_Energy, ValueWithDimension): ...
class EnergyArray(_Energy, ArrayWithDimension[Energy]): ...

class _Force(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Force(_Force, ValueWithDimension): ...
class ForceArray(_Force, ArrayWithDimension[Force]): ...

class _Frequency(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Frequency(_Frequency, ValueWithDimension): ...
class FrequencyArray(_Frequency, ArrayWithDimension[Frequency]): ...

class _Illuminance(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Illuminance(_Illuminance, ValueWithDimension): ...
class IlluminanceArray(_Illuminance, ArrayWithDimension[Illuminance]): ...

class _Inductance(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Inductance(_Inductance, ValueWithDimension): ...
class InductanceArray(_Inductance, ArrayWithDimension[Inductance]): ...

class _Length(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Length(_Length, ValueWithDimension): ...
class LengthArray(_Length, ArrayWithDimension[Length]): ...

class _LogPower(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class LogPower(_LogPower, ValueWithDimension): ...
class LogPowerArray(_LogPower, ArrayWithDimension[LogPower]): ...

class _LuminousFlux(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class LuminousFlux(_LuminousFlux, ValueWithDimension): ...
class LuminousFluxArray(_LuminousFlux, ArrayWithDimension[LuminousFlux]): ...

class _LuminousIntensity(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class LuminousIntensity(_LuminousIntensity, ValueWithDimension): ...
class LuminousIntensityArray(_LuminousIntensity, ArrayWithDimension[LuminousIntensity]): ...

class _MagneticFlux(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class MagneticFlux(_MagneticFlux, ValueWithDimension): ...
class MagneticFluxArray(_MagneticFlux, ArrayWithDimension[MagneticFlux]): ...

class _MagneticFluxDensity(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class MagneticFluxDensity(_MagneticFluxDensity, ValueWithDimension): ...
class MagneticFluxDensityArray(_MagneticFluxDensity, ArrayWithDimension[MagneticFluxDensity]): ...

class _Mass(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Mass(_Mass, ValueWithDimension): ...
class MassArray(_Mass, ArrayWithDimension[Mass]): ...

class _Noise(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Noise(_Noise, ValueWithDimension): ...
class NoiseArray(_Noise, ArrayWithDimension[Noise]): ...

class _Power(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Power(_Power, ValueWithDimension): ...
class PowerArray(_Power, ArrayWithDimension[Power]): ...

class _Pressure(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Pressure(_Pressure, ValueWithDimension): ...
class PressureArray(_Pressure, ArrayWithDimension[Pressure]): ...

class _Quantity(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Quantity(_Quantity, ValueWithDimension): ...
class QuantityArray(_Quantity, ArrayWithDimension[Quantity]): ...

class _Resistance(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Resistance(_Resistance, ValueWithDimension): ...
class ResistanceArray(_Resistance, ArrayWithDimension[Resistance]): ...

class _Speed(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Speed(_Speed, ValueWithDimension): ...
class SpeedArray(_Speed, ArrayWithDimension[Speed]): ...

class _SurfaceDensity(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class SurfaceDensity(_SurfaceDensity, ValueWithDimension): ...
class SurfaceDensityArray(_SurfaceDensity, ArrayWithDimension[SurfaceDensity]): ...

class _Temperature(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Temperature(_Temperature, ValueWithDimension): ...
class TemperatureArray(_Temperature, ArrayWithDimension[Temperature]): ...

class _Time(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Time(_Time, ValueWithDimension): ...
class TimeArray(_Time, ArrayWithDimension[Time]): ...

class _Torque(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Torque(_Torque, ValueWithDimension): ...
class TorqueArray(_Torque, ArrayWithDimension[Torque]): ...

class _Volume(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class Volume(_Volume, ValueWithDimension): ...
class VolumeArray(_Volume, ArrayWithDimension[Volume]): ...

class _WaveNumber(Dimension):
    @staticmethod
    def valid_base_units() -> tuple[Value, ...]: ...

class WaveNumber(_WaveNumber, ValueWithDimension): ...
class WaveNumberArray(_WaveNumber, ArrayWithDimension[WaveNumber]): ...

@frozen
class BaseUnitData:
    """Describes the properties of a base unit.

    Attributes:
        symbol: The short name for the unit (e.g. 'm' for meter).
        name: The full name of the unit (e.g. 'meter').
        use_prefixes: Should there be 'kiloUNIT', 'milliUNIT', etc.
    """

    symbol: str
    name: str
    use_prefixes: bool = True

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

@frozen
class PrefixData:
    """Describes the properties of a unit prefix.

    Attributes:
        symbol: The short name for the prefix (e.g. 'G' for giga).
        name: The full name of the prefix (e.g. 'giga').
        exp10: The power of 10 the prefix corresponds to.
    """

    symbol: str
    name: str
    exp10: int

class UnitDatabase:
    """
    Values defined in unit_array do not actually store a unit object, the unit
    names and powers are stored within the value object itself.  However, when
    constructing new values or converting between units, we need a database of
    known units.
    """

    known_units: dict[str, Value]
    auto_create_units: bool

    def __init__(self, auto_create_units: bool = True):
        """
        :param auto_create_units: Determines if unrecognized strings are
        interpreted as new units or not by default.
        """

    def get_unit(self, unit_name: str, auto_create: bool | None = None) -> Value:
        """
        :param str unit_name:
        :param None|bool auto_create: If this is set, a missing unit will be
        created and returned instead of causing an error. If not specified,
        defaults to the 'auto_create_units' attribute of the receiving instance.
        :return Value: The unit with the given name.
        """

    def parse_unit_formula(self, formula: str, auto_create: bool | None = None) -> Value:
        """
        :param str formula: Describes a combination of units.
        :param None|bool auto_create: If this is set, missing unit strings will
        cause new units to be created and returned instead of causing an error.
        If not specified, defaults to the 'auto_create_units' attribute of the
        receiving instance.
        :return Value: The value described by the formula.
        """

    def add_unit(self, unit_name: str, unit_base_value: Value) -> None:
        """
        Adds a unit to the database, pointing it at the given value.
        :param str unit_name: Key for the new unit.
        :param Value unit_base_value: The unit's value.
        """

    def add_root_unit(self, unit_name: str) -> None:
        """
        Adds a plain unit, not defined in terms of anything else, to the database.
        :param str unit_name: Key and unit array entry for the new unit.
        """

    def add_alternate_unit_name(self, alternate_name: str, unit_name: str) -> None:
        """
        Adds an alternate name for a unit, mapping to exactly the same value.
        :param str alternate_name: The new alternate name for the unit.
        :param str unit_name: The existing name for the unit.
        """

    def add_scaled_unit(
        self,
        unit_name: str,
        formula: str,
        factor: int | float | complex | np.number[Any] = 1.0,
        numer: int = 1,
        denom: int = 1,
        exp10: int = 0,
    ) -> None:
        """
        Creates and adds a derived unit to the database. The unit's value is
        computed by parsing the given formula (in terms of existing units) and
        applying the given scaling parameters.
        :param str unit_name: Name of the derived unit.
        :param str formula: Math expression containing a unit combination.
        :param float factor: A lossy factor for converting to the base unit.
        :param int numer: An exact factor for converting to the base unit.
        :param int denom: An exact divisor for converting to the base unit.
        :param int exp10: An exact power-of-10 for converting to the base unit.
        """

    def add_base_unit_data(self, data: BaseUnitData, prefixes: list[PrefixData]) -> None:
        """
        Adds a unit, with alternate names and prefixes, defined by a
        BaseUnitData and some PrefixData.
        :param BaseUnitData data:
        :param list[PrefixData] prefixes:
        """

    def add_derived_unit_data(self, data: DerivedUnitData, prefixes: list[PrefixData]) -> None:
        """
        Adds a unit, with alternate names and prefixes, defined by a
        DerivedUnitData and some PrefixData.
        :param DerivedUnitData data:
        :param list[PrefixData] prefixes:
        """

    def add_physical_constant_data(self, data: PhysicalConstantData) -> None:
        """
        Adds a physical constant, i.e. a unit that doesn't override the display
        units of a value, defined by a PhysicalConstantData.
        :param PhysicalConstantData data:
        """

    def is_value_consistent_with_database(self, value: Value) -> bool:
        """
        Determines if the value's base and display units are known and that
        the conversion factor between them is consistent with the known unit
        scales.

        :param Value value:
        :return bool:
        """

default_unit_database: UnitDatabase
SI_PREFIXES: list[PrefixData]
