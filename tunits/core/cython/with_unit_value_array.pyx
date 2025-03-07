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

from typing import TypeVar

import numpy as np

T = TypeVar('ValueArray', bound='ValueArray')

def _canonize_data_and_unit(data, unit=None):
    first_item = next(data.flat, None)
    if isinstance(first_item, WithUnit):
        shared_unit = first_item.unit
        scalar = first_item[shared_unit]
        inferred_dtype = np.array([scalar]).dtype

        it = np.nditer([data, None],
                        op_dtypes=[data.dtype, inferred_dtype],
                        flags=['refs_ok'],
                        op_flags=[['readonly'], ['writeonly', 'allocate']])
        for inp, out in it:
            out[()] = inp[()][shared_unit]

        data = it.operands[1]
        unit = shared_unit if unit is None else unit * shared_unit
    return data, unit

class ValueArray(WithUnit):

    def __init__(WithUnit self, data, unit=None):
        """
        Initializes an array of values with an associated unit.

        If the first item has units, a shared unit will be extracted from all
        the items.
        """
        if unit is not None:
            parsed_unit = _try_interpret_as_with_unit(unit)
            if parsed_unit is None:
                raise ValueError("Bad WithUnit scaling value: " + repr(unit))
            unit = parsed_unit

        # If the items have units, we're supposed to extract a shared unit.
        data = np.asarray(data)
        data, unit = _canonize_data_and_unit(data, unit)

        super().__init__(data, unit)

    """ A numpy array with associated units. """
    def __setitem__(WithUnit self, key, val):
        cdef WithUnit right = _in_WithUnit(val)
        if self.base_units != right.base_units:
            raise UnitMismatchError(
                "'%s' can't be put in an array of '%s'." %
                    (val, self.display_units))
        cdef conversion conv = conversion_div(right.conv, self.conv)
        self.value[key] = right.value * conversion_to_double(conv)

    def __copy__(WithUnit self):
        return self.__with_value(copy.copy(self.value))

    def __deepcopy__(WithUnit self, memo):
        return self.__with_value(copy.deepcopy(self.value))

    def __iter__(WithUnit self):
        for e in self.value:
            yield self.__with_value(e)

    def __len__(WithUnit self):
        return len(self.value)

    def __array__(WithUnit self, dtype=None, copy: bool=False):
        if self.is_dimensionless:
            # TODO: pass copy to np.asarray.
            return np.asarray(conversion_to_double(self.conv) * self.value,
                              dtype=dtype)

        unit_array = np.full_like(self.value, self.unit, dtype=object)
        result = self.value * unit_array
        return np.asarray(result)

    def __array_wrap__(WithUnit self, out_arr, context=None, return_scalar: bool=False):
        return np.ndarray.__array_wrap__(self.value, out_arr, return_scalar)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            if ufunc == np.add:
                return inputs[0] + inputs[1]
            if ufunc == np.subtract:
                return inputs[0] - inputs[1]
            if ufunc == np.multiply:
                if not isinstance(inputs[0], np.ndarray) and not isinstance(inputs[1], np.ndarray):
                    return inputs[0] * inputs[1]
                elif isinstance(inputs[0], np.ndarray):
                    return inputs[1] * inputs[0]
                elif isinstance(inputs[1], np.ndarray):
                    return inputs[0] * inputs[1]
                else:
                    raise NotImplementedError(
                        f"multiply not implemented for types {type(inputs[0])}, {type(inputs[1])}"
                    )
            if ufunc == np.divide:
                if not isinstance(inputs[0], np.ndarray) and not isinstance(inputs[1], np.ndarray):
                    return inputs[0] / inputs[1]
                elif isinstance(inputs[0], np.ndarray):
                    return inputs[1].__rtruediv__(inputs[0])
                elif isinstance(inputs[1], np.ndarray):
                    return inputs[0] / inputs[1]
                else:
                    raise NotImplementedError(
                        f"divide not implemented for types {type(inputs[0])}, {type(inputs[1])}"
                    )
            if ufunc == np.power:
                return inputs[0] ** inputs[1]
            if ufunc in [np.positive, np.negative, np.abs, np.fabs, np.conj]:
                return ufunc(self.value) * self.unit
            if ufunc in [np.sign, np.isfinite, np.isinf, np.isnan]:
                return ufunc(self.value)
            if ufunc == np.sqrt:
                return self.sqrt()
            if ufunc == np.square:
                return self ** 2
            if ufunc == np.reciprocal:
                return self.__rtruediv__(1)

        if ufunc in [
            np.greater,
            np.greater_equal,
            np.less,
            np.less_equal,
            np.not_equal,
            np.equal,
            np.maximum,
            np.minimum,
            np.fmax,
        ]:
            return getattr(ufunc, method)(*(np.asarray(x) for x in inputs), **kwargs)

        if self._is_dimensionless():
            return getattr(ufunc, method)(*(np.asarray(x) for x in inputs), **kwargs)

        raise NotImplemented

    @property
    def dtype(WithUnit self) -> np.dtype:
        return self.value.dtype

    @property
    def ndim(WithUnit self) -> int:
        return self.value.ndim

    @property
    def shape(WithUnit self) -> tuple[int, ...]:
        return self.value.shape

    def allclose(WithUnit self, other, *args, **kw) -> bool:
        return np.allclose(self.value, other[self.unit], *args, **kw)

    @classmethod
    def from_proto(cls: type[T], msg: 'tunits_pb2.ValueArray') -> T:
        return cls(_ndarray_from_proto(msg), _proto_to_units(msg.units))

    def to_proto(self, msg: Optional['tunits_pb2.ValueArray'] = None) -> 'tunits_pb2.ValueArray':
        ret = _ndarray_to_proto(self.value, msg)
        ret.units.extend(_units_to_proto(self.display_units))
        return ret