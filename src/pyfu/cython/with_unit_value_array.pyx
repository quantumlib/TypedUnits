import numpy as np


class ValueArray(WithUnit):

    def __init__(WithUnit self, data, unit=None):
        """
        Initializes an array of values with an associated unit.

        If the first item has units, a shared unit will be extracted from all
        the items.
        """
        if unit is not None:
            unit = __try_interpret_as_with_unit(unit)

        # If the items have units, we're supposed to extract a shared unit.
        data = np.asarray(data)
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

    def __array__(WithUnit self, dtype=None):
        if self.isDimensionless():
            return np.asarray(conversion_to_double(self.conv) * self.value,
                              dtype=dtype)

        unit_array = np.full_like(self.value, self.unit, dtype=np.object)
        result = self.value * unit_array
        return np.asarray(result)

    def __array_wrap__(WithUnit self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self.value, out_arr)

    @property
    def dtype(WithUnit self):
        return self.value.dtype

    @property
    def ndim(WithUnit self):
        return self.value.ndim

    @property
    def shape(WithUnit self):
        return self.value.shape

    def allclose(WithUnit self, other, *args, **kw):
        return np.allclose(self.value, other[self.unit], *args, **kw)