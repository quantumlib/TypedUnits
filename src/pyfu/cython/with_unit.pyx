import cython
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.object cimport Py_EQ, Py_NE, Py_LE, Py_GE, Py_LT, Py_GT
import copy
import copy_reg
import numpy as np

from libc.math cimport pow as c_pow


def isOrAllTrue(x):
    return np.all(x) if isinstance(x, np.ndarray) else x


cpdef raw_WithUnit(value,
                   conversion conv,
                   UnitArray base_units,
                   UnitArray display_units):
    """
    A factory method that directly sets the properties of a WithUnit.
    (__init__ couldn't play this role for backwards-compatibility reasons.)

    (Python-visible for testing and unit database bootstrapping.)
    """

    # Choose derived class type.
    if isinstance(value, complex):
        val = value
        target_type = Complex
    elif isinstance(value, list) or isinstance(value, np.ndarray):
        val = np.array(value)
        target_type = ValueArray
    elif (isinstance(value, int) or
          isinstance(value, float) or
          isinstance(value, long)):
        val = float(value)
        target_type = Value
    else:
        raise TypeError("Unrecognized value type: {}".format(type(value)))

    cdef WithUnit result = target_type(val)
    result.conv = conv
    result.base_units = base_units
    result.display_units = display_units
    return result


def _in_WithUnit(obj):
    """
    Wraps the given object into a WithUnit instance, unless it's already
    a WithUnit.
    """
    if isinstance(obj, WithUnit):
        return obj
    return raw_WithUnit(obj, identity_conversion(), _EmptyUnit, _EmptyUnit)


def _is_dimensionless_zero(WithUnit u):
    return (u.isDimensionless() and
            not isinstance(u.value, np.ndarray) and
            u.value == 0)

@cython.auto_pickle(True)
cdef class WithUnit:
    """
    A value with associated physical units.
    """

    """Floating point value"""
    cdef readonly value
    """Conversion details to go from display units to base units."""
    cdef conversion conv
    """Units in base units"""
    cdef readonly UnitArray base_units
    """Units for display"""
    cdef readonly UnitArray display_units

    property numer:
        def __get__(self):
            return self.conv.ratio.numer
    property denom:
        def __get__(self):
            return self.conv.ratio.denom
    property factor:
        def __get__(self):
            return self.conv.factor
    property exp10:
        def __get__(self):
            return self.conv.exp10

    def __init__(WithUnit self, value, unit=None):
        """
        Creates a value with associated units.
        :param value: The value. An int, float, complex, list, string, Unit,
                      WithUnit, or ndarray.
        :param unit: A representation of the physical units. Could be an
            instance of Unit, or UnitArray, or a string to be parsed.
        """
        if isinstance(value, list):
            value = np.array(value)
        if unit is None and not isinstance(value, WithUnit):
            self.value = value
            self.conv = identity_conversion()
            self.base_units = _EmptyUnit
            self.display_units = _EmptyUnit
            return

        cdef WithUnit unit_val = WithUnit(1) if unit is None else \
            __try_interpret_as_with_unit(unit)
        if unit_val is None:
            raise ValueError("Bad WithUnit scaling value: " + repr(value))
        unit_val *= value
        self.value = unit_val.value
        self.conv = unit_val.conv
        self.base_units = unit_val.base_units
        self.display_units = unit_val.display_units

    cdef __with_value(self, new_value):
        return raw_WithUnit(
            new_value,
            self.conv,
            self.base_units,
            self.display_units)

    def __neg__(WithUnit self):
        return self.__with_value(-self.value)

    def __pos__(self):
        return self

    def __abs__(WithUnit self):
        return self.__with_value(abs(self.value))

    def __nonzero__(self):
        return bool(self.value)

    def __add__(a, b):
        cdef WithUnit left = _in_WithUnit(a)
        cdef WithUnit right = _in_WithUnit(b)

        # Adding dimensionless zero is always fine (but watch out for arrays).
        if _is_dimensionless_zero(left):
            return right
        if _is_dimensionless_zero(right):
            return left

        if left.base_units != right.base_units:
            raise UnitMismatchError("Can't add '%s' and '%s'." % (left, right))

        cdef conversion left_to_right = conversion_div(left.conv, right.conv)
        cdef double c = conversion_to_double(left_to_right)

        # Prefer scaling up, not down.
        if c > -1 and c < 1:
            c = conversion_to_double(inverse_conversion(left_to_right))
            return left.__with_value(left.value + right.value * c)

        return right.__with_value(left.value * c + right.value)

    def __sub__(a, b):
        return a + -b

    def __mul__(a, b):
        cdef WithUnit left = _in_WithUnit(a)
        cdef WithUnit right = _in_WithUnit(b)
        return raw_WithUnit(left.value * right.value,
                            conversion_times(left.conv, right.conv),
                            left.base_units * right.base_units,
                            left.display_units * right.display_units)

    def __div__(a, b):
        cdef WithUnit left = _in_WithUnit(a)
        cdef WithUnit right = _in_WithUnit(b)
        return raw_WithUnit(left.value / right.value,
                            conversion_div(left.conv, right.conv),
                            left.base_units / right.base_units,
                            left.display_units / right.display_units)

    def __truediv__(a, b):
        cdef WithUnit left = _in_WithUnit(a)
        cdef WithUnit right = _in_WithUnit(b)
        return left.__div__(right)

    def __divmod__(a, b):
        cdef WithUnit left = _in_WithUnit(a)
        cdef WithUnit right = _in_WithUnit(b)
        if left.base_units != right.base_units:
            raise UnitMismatchError("Can't divmod '%s' by '%s'." %
                (left, right))

        cdef double c = conversion_to_double(conversion_div(left.conv,
                                                            right.conv))

        q, r = divmod(left.value * c, right.value)
        return q, right.__with_value(r)

    def __floordiv__(a, b):
        return divmod(a, b)[0]

    def __mod__(a, b):
        return divmod(a, b)[1]

    def __pow__(WithUnit self not None, exponent, modulo):
        """
        Raises the given value to the given power, assuming the exponent can
        be broken down into twelths.
        """
        if modulo is not None:
            raise ValueError("WithUnit.__pow__ doesn't support modulo argument")

        cdef frac exponent_frac = float_to_twelths_frac(exponent)

        return raw_WithUnit(self.value ** exponent,
                            conversion_raise_to(self.conv, exponent_frac),
                            self.base_units.pow_frac(exponent_frac),
                            self.display_units.pow_frac(exponent_frac))

    def sqrt(WithUnit self):
        return self ** 0.5

    def __int__(self):
        if self.base_units.unit_count != 0:
            raise UnitMismatchError(
                "'%s' can't be stripped into an int; not dimensionless." % self)
        return int(conversion_to_double(self.conv) * self.value)

    def __float__(self):
        if self.base_units.unit_count != 0:
            raise UnitMismatchError(
                "'%s' can't be stripped into a float; not dimensionless." %
                    self)
        return conversion_to_double(self.conv) * float(self.value)

    def __complex__(self):
        if self.base_units.unit_count != 0:
            raise UnitMismatchError(
                "'%s' can't be stripped into a complex; not dimensionless." %
                    self)
        return conversion_to_double(self.conv) * complex(self.value)

    def __richcmp__(a, b, int op):
        cdef WithUnit left
        cdef WithUnit right
        try:
            left = _in_WithUnit(a)
            right = _in_WithUnit(b)
        except:
            return NotImplemented

        # Check units.
        if left.base_units != right.base_units:
            # For arrays we want a list of true/false comparisons.
            shaped_false = (left.value == right.value) & False
            if op == Py_EQ:
                return shaped_false
            if op == Py_NE:
                return not shaped_false
            raise UnitMismatchError("Can't compare '%s' to '%s'." %
                (left, right))

        # Compute scaled comparand values, without dividing.
        cdef frac f = frac_div(left.conv.ratio, right.conv.ratio)
        u = left.value * left.conv.factor * f.numer
        v = right.value * right.conv.factor * f.denom
        cdef int e = left.exp10 - right.exp10
        if e > 0:
            # Note: don't use *=. Numpy will do an inplace modification.
            u = u * c_pow(10, e)
        else:
            v = v * c_pow(10, -e)

        # Delegate to value comparison.
        if op == Py_EQ:
            return u == v
        elif op == Py_NE:
            return u != v
        elif op == Py_LT:
            return u < v
        elif op == Py_GT:
            return u > v
        elif op == Py_LE:
            return u <= v
        elif op == Py_GE:
            return u >= v

        return NotImplemented

    def __str__(self):
        unit_str = str(self.display_units)
        if not isinstance(self.value, np.ndarray) \
                and self.value == 1 and unit_str != '':
            return unit_str
        val_str = (repr if isinstance(self.value, float) else str)(self.value)
        return (val_str + " " + unit_str).strip()

    def __repr__(WithUnit self):
        if __is_value_consistent_with_default_unit_database(self):
            return "%s(%s, '%s')" % (
                type(self).__name__,
                repr(self.value),
                str(self.display_units))

        return "raw_WithUnit(%s)" % ', '.join(
            repr(e) for e in _pickle_WithUnit(self)[1])

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def inBaseUnits(WithUnit self):
        return raw_WithUnit(
            self.value * conversion_to_double(self.conv),
            identity_conversion(),
            self.base_units,
            self.base_units)

    def isDimensionless(self):
        return self.base_units.unit_count == 0

    def isAngle(self):
        if self.base_units.unit_count != 1:
            return False
        cdef UnitTerm unit = self.base_units.units[0]
        return unit.power.numer == 1 \
            and unit.power.denom == 1 \
            and <str>unit.name == "rad"

    property is_angle:
        def __get__(self):
            return self.isAngle()

    def __getitem__(WithUnit self, key):
        """
        Returns the number of given units needed to make up the receiving value,
         or else returns the wrapped result of forwarding the key into the
         receiving value's inner value's own __getitem__.

        :param str|WithUnit|* key: The unit, or formula representing a unit,
        to compare the receiving unit against. Or else an index or slice or
        other __getitem__ key to forward.
        """
        cdef WithUnit unit_val
        if isinstance(key, WithUnit) or isinstance(key, str):
            unit_val = __try_interpret_as_with_unit(key, True)
            if unit_val is None:
                raise TypeError("Bad unit key: " + repr(key))
            if self.base_units != unit_val.base_units:
                raise UnitMismatchError("'%s' doesn't match '%s'." %
                    (self, key))
            return (self.value
                * conversion_to_double(conversion_div(self.conv, unit_val.conv))
                / unit_val.value)

        return self.__with_value(self.value[key])

    def __iter__(self):
        # Hack: We want calls to 'iter' to see that __iter__ exists and try to
        # use it, instead of falling back to checking if __getitem__ exists,
        # assuming scalars are iterable, and returning an iterator that only
        # blows up later. So we define this do-nothing method.
        raise TypeError("'WithUnit' object is not iterable")

    def isCompatible(self, unit):
        cdef WithUnit other = __try_interpret_as_with_unit(unit)
        if other is None:
            raise ValueError("Bad unit key: " + repr(unit))
        return self.base_units == other.base_units

    def inUnitsOf(WithUnit self, unit):
        cdef WithUnit unit_val = __try_interpret_as_with_unit(unit)
        if unit_val is None:
            raise ValueError("Bad unit key: " + repr(unit))
        if self.base_units != unit_val.base_units:
            raise UnitMismatchError("'%s' doesn't have units matching '%s'." %
                (self, unit))

        return unit_val.__with_value(self.value
             * conversion_to_double(conversion_div(self.conv, unit_val.conv))
             / unit_val.value)

    def __hash__(self):
        # Note: Anyone calling this, except in the case where they're using a
        # single unchanging value as a key, likely has a bug.
        return hash(self.inBaseUnits().value)

    property unit:
        def __get__(self):
            return self.__with_value(1)

    def __array__(self, dtype=None):
        if self.isDimensionless():
            # Unwrap into raw numbers.
            return np.array(
                conversion_to_double(self.conv) * self.value,
                dtype=dtype)

        result = np.empty(dtype=np.object, shape=())
        result[()] = self
        return result

    def __array_wrap__(WithUnit self, out_arr):
        if out_arr.shape == ():
            return out_arr[()]
        return np.ndarray.__array_wrap__(self.value, out_arr)

    __array_priority__ = 15

__try_interpret_as_with_unit = None
__is_value_consistent_with_default_unit_database = None
def init_base_unit_functions(
        try_interpret_as_with_unit,
        is_value_consistent_with_default_unit_database):
    global __try_interpret_as_with_unit
    global __is_value_consistent_with_default_unit_database
    __try_interpret_as_with_unit = try_interpret_as_with_unit
    __is_value_consistent_with_default_unit_database = \
            is_value_consistent_with_default_unit_database

def _pickle_WithUnit(WithUnit e):
    return raw_WithUnit, (
        e.value,
        {
            'factor': e.conv.factor,
            'ratio': {'numer': e.conv.ratio.numer, 'denom': e.conv.ratio.denom},
            'exp10': e.conv.exp10
        },
        e.base_units,
        e.display_units)

copy_reg.pickle(WithUnit, _pickle_WithUnit)
