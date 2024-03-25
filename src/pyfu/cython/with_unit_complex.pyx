from typing import TypeVar
import numpy as np


T = TypeVar('Complex', bound='Complex')


class Complex(WithUnit):
    """A complex value with associated units."""

    @classmethod
    def from_proto(cls: type[T], msg: tunits_pb2.Value) -> T:
        if not msg.HasField('complex_value'):
            raise ValueError(f"{msg=} doesn't have a value.")
        val = msg.complex_value.real + 1j * msg.complex_value.imaginary
        return cls(val, _proto_to_units(msg.units))

    def to_proto(self, msg: Optiona[tunits_pb2.Value] = None) -> tunits_pb2.Value:
        if msg is None:
            msg = tunits_pb2.Value()
        msg.complex_value.real = self.value.real
        msg.complex_value.imaginary = self.value.imag
        msg.units.extend(_units_to_proto(self.display_units))
        return msg
