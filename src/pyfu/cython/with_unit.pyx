from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.object cimport Py_EQ, Py_NE, Py_LE, Py_GE, Py_LT, Py_GT
import copy
import copy_reg
import numpy as np

from libc.math cimport pow as c_pow


def isOrAllTrue(x):
    return np.all(x) if isinstance(x, np.ndarray) else x


cdef long long inv_root(long long x, int exponent_denom):
    cdef long long tmp = <long long>c_pow(x, 1.0/exponent_denom)
    if c_pow(tmp, exponent_denom) != x:
        raise ValueError("%s root of %s not an integer" % (exponent_denom, x))
    return tmp


cpdef raw_WithUnit(value,
                   long long numer,
                   long long denom,
                   int exp10,
                   UnitArray base_units,
                   UnitArray display_units):
    """
    A factory method the creates and directly sets the properties of a WithUnit.
    (__init__ couldn't play this role for backwards-compatibility reasons.)
    """

    if not (numer > 0 and denom > 0):
        raise ValueError("Numerator and denominator must both be positive.")

    # Choose derived class type.
    if isinstance(value, complex):
        val = value
        target_type = Complex
    elif isinstance(value, np.ndarray):
        val = value
        target_type = ValueArray
    elif isinstance(value, list) or isinstance(value, np.ndarray):
        val = np.array(value)
        target_type = ValueArray
    elif isinstance(value, int) or isinstance(value, float):
        val = float(value)
        target_type = Value
    else:
        raise ValueError("Unrecognized value type: " + type(value))

    cdef WithUnit result = target_type(0)
    result.value = val
    result.ratio = frac_least_terms(numer, denom)
    result.exp10 = exp10
    result.base_units = base_units
    result.display_units = display_units
    return result


cdef class WithUnit:
    """
    A value with associated physical units.
    """

    """Floating point value"""
    cdef readonly value
    """Fractional part of ratio between base and display units"""
    cdef frac ratio
    """Power of 10 ratio between base and display units"""
    cdef readonly int exp10
    """Units in base units"""
    cdef readonly UnitArray base_units
    """Units for display"""
    cdef readonly UnitArray display_units

    property numer:
        def __get__(self):
            return self.ratio.numer
    property denom:
        def __get__(self):
            return self.ratio.denom

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
            self.base_units = _EmptyUnit
            self.display_units = _EmptyUnit
            self.exp10 = 0
            self.ratio.numer = 1
            self.ratio.denom = 1
            return

        cdef WithUnit unit_val = WithUnit(1) if unit is None else \
            __try_interpret_as_with_unit(unit)
        if unit_val is None:
            raise ValueError("Bad WithUnit scaling value: " + repr(value))
        unit_val *= value
        self.value = unit_val.value
        self.base_units = unit_val.base_units
        self.display_units = unit_val.display_units
        self.exp10 = unit_val.exp10
        self.ratio = unit_val.ratio

    @staticmethod
    def raw(value,
            long long numer,
            long long denom,
            int exp10,
            UnitArray base_units not None,
            UnitArray display_units not None):
        """
        Creates a WithUnit instance, of the appropriate type for the given
        value, with the given properties.
        """
        return raw_WithUnit(value, numer, denom, exp10, base_units,
                            display_units)

    @staticmethod
    def wrap(obj):
        """
        Wraps the given object into a WithUnit instance, unless it's already
        a WithUnit.
        """
        if isinstance(obj, WithUnit):
            return obj
        return raw_WithUnit(obj, 1, 1, 0, _EmptyUnit, _EmptyUnit)

    cdef __with_value(self, new_value):
        return raw_WithUnit(
            new_value,
            self.ratio.numer,
            self.ratio.denom,
            self.exp10,
            self.base_units,
            self.display_units)

    cdef double _scale_to_double(self):
        return frac_to_double(self.ratio) * c_pow(10.0, self.exp10)

    def __neg__(self):
        return self.__with_value(-self.value)

    def __pos__(self):
        return self

    def __abs__(self):
        return self.__with_value(abs(self.value))

    def __nonzero__(self):
        return bool(self.value)

    def __add__(a, b):
        cdef WithUnit left = WithUnit.wrap(a)
        cdef WithUnit right = WithUnit.wrap(b)

        if left.base_units != right.base_units:
            raise UnitMismatchError()

        cdef double fL = left._scale_to_double()
        cdef double fR = right._scale_to_double()
        # Prefer finer grained display units.
        if fL < fR:
            return left.__with_value(left.value + right.value * (fR/fL))
        return right.__with_value(right.value + left.value * (fL/fR))

    def __sub__(a, b):
        return a + -b

    def __mul__(a, b):
        cdef WithUnit left = WithUnit.wrap(a)
        cdef WithUnit right = WithUnit.wrap(b)
        cdef frac ratio = frac_times(left.ratio, right.ratio)
        return raw_WithUnit(left.value * right.value,
                            ratio.numer,
                            ratio.denom,
                            left.exp10 + right.exp10,
                            left.base_units * right.base_units,
                            left.display_units * right.display_units)

    def __div__(a, b):
        cdef WithUnit left = WithUnit.wrap(a)
        cdef WithUnit right = WithUnit.wrap(b)
        cdef frac ratio = frac_div(left.ratio, right.ratio)
        return raw_WithUnit(left.value / right.value,
                            ratio.numer,
                            ratio.denom,
                            left.exp10 - right.exp10,
                            left.base_units / right.base_units,
                            left.display_units / right.display_units)

    def __divmod__(a, b):
        cdef WithUnit left = WithUnit.wrap(a)
        cdef WithUnit right = WithUnit.wrap(b)
        if left.base_units != right.base_units:
            raise UnitMismatchError()

        cdef double f = left._scale_to_double() / right._scale_to_double()
        divmod_result = divmod(left.value * f, right.value)
        remainder = right.__with_value(divmod_result[1])
        return divmod_result[0], right.__with_value(divmod_result[1])

    def __floordiv__(a, b):
        return divmod(a, b)[0]

    def __mod__(a, b):
        return divmod(a, b)[1]

    def __pow__(WithUnit left not None, exponent, modulo):
        """
        Raises the given value to the given power, assuming the exponent can
        be broken down into twelths.
        """
        if modulo is not None:
            raise ValueError("WithUnit power does not support third argument")

        cdef frac pow_frac = float_to_twelths_frac(exponent)
        cdef int abs_numer = pow_frac.numer
        cdef int pow_sign = 1
        if abs_numer < 0:
            abs_numer *= -1
            pow_sign = -1

        if (left.exp10 * pow_frac.numer) % pow_frac.denom:
            raise RuntimeError("Unable to take root of specified unit")

        cdef long long numer = <long long>c_pow(
            inv_root(left.ratio.numer, pow_frac.denom), abs_numer)
        cdef long long denom = <long long>c_pow(
            inv_root(left.ratio.denom, pow_frac.denom), abs_numer)
        if pow_sign == -1:
            numer, denom = denom, numer

        val = left.value ** exponent
        cdef int exp10 = left.exp10 * pow_frac.numer / pow_frac.denom

        cdef UnitArray base_units = left.base_units.pow_frac(pow_frac)
        cdef UnitArray display_units = left.display_units.pow_frac(pow_frac)
        return raw_WithUnit(val,
                            numer,
                            denom,
                            exp10,
                            base_units,
                            display_units)

    def __float__(self):
        if self.base_units.unit_count != 0:
            raise UnitMismatchError(
                "Only dimensionless values can be stripped into a float.")
        return self._scale_to_double() * float(self.value)

    def __complex__(self):
        if self.base_units.unit_count != 0:
            raise UnitMismatchError(
                "Only dimensionless values can be stripped into a complex.")
        return self._scale_to_double() * complex(self.value)

    def __richcmp__(a, b, int op):
        cdef WithUnit left
        cdef WithUnit right
        try:
            left = WithUnit.wrap(a)
            right = WithUnit.wrap(b)
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
            raise UnitMismatchError("Comparands have different units.")

        # Compute scaled comparand values, without dividing.
        u = left.value
        v = right.value
        cdef frac f = frac_div(left.ratio, right.ratio)
        cdef int e = left.exp10 - right.exp10
        if e > 0:
            u = u * (f.numer * c_pow(10, e))
            v = v * f.denom
        else:
            u = u * f.numer
            v = v * (f.denom * c_pow(10, -e))

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
        return ("%s %s" % (str(self.value), unit_str)).strip()

    def __repr__(self):
        # If the default unit database is capable of correctly parsing our
        # units, use a nice output. Else use a gross but correct output.

        cdef WithUnit parse_attempt
        try:
            parse_attempt = type(self)(self.value, str(self.display_units))
            if (parse_attempt.base_units == self.base_units
                    and parse_attempt.display_units == self.display_units
                    and parse_attempt.ratio.numer == self.ratio.numer
                    and parse_attempt.ratio.denom == self.ratio.denom
                    and parse_attempt.exp10 == self.exp10
                    and isOrAllTrue(parse_attempt.value == self.value)):
                return "%s(%s, '%s')" % (
                    type(self).__name__,
                    repr(self.value),
                    str(self.display_units))
        except:
            # Some kind of non-standard unit? Fall back to raw output.
            pass

        return "raw_WithUnit(%s)" % ', '.join(repr(e) for e in [
            self.value,
            self.ratio.numer,
            self.ratio.denom,
            self.exp10,
            self.base_units,
            self.display_units
        ])

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def inBaseUnits(self):
        factor = self._scale_to_double()
        new_value = self.value * factor
        return raw_WithUnit(
            new_value, 1, 1, 0, self.base_units, self.base_units)

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

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            return self.__with_value(self.value[key])

        cdef WithUnit unit_val = __try_interpret_as_with_unit(key)
        if unit_val is None:
            raise TypeError("Bad unit key: " + repr(key))

        # We could interpret the dimensionless value like the others, but x[1.0]
        # is uncomfortably close to x[1] yet acts differently so for now we will
        # disallow it.
        if unit_val.isDimensionless() and not isinstance(key, str):
            raise TypeError("Ambiguous unit key: " + repr(key))

        if self.base_units != unit_val.base_units:
            raise UnitMismatchError("Unit key doesn't match value's units.")

        return (self.value
            * frac_to_double(frac_div(self.ratio, unit_val.ratio))
            * c_pow(10.0, self.exp10 - unit_val.exp10)
            / unit_val.value)

    def isCompatible(self, unit):
        cdef WithUnit other = __try_interpret_as_with_unit(unit)
        if other is None:
            raise ValueError("Bad unit key: " + repr(unit))
        return self.base_units == other.base_units

    def inUnitsOf(WithUnit self, unit):
        cdef WithUnit unit_val = __try_interpret_as_with_unit(unit)
        if unit_val is None:
            raise ValueError("Bad unit key: " + repr(unit))
        return unit_val.__with_value(self[unit_val])

    def __hash__(self):
        # Note: Anyone calling this, except in the case where they're using a
        # single unchanging value as a key, likely has a bug.
        return hash(self.inBaseUnits().value)

    property unit:
        def __get__(self):
            return self.__with_value(1)

    def __array__(self):
        if self.base_units.unit_count != 0:
            raise UnitMismatchError(
                "Only dimensionless values can be stripped into an array.")
        return np.array(self._scale_to_double() * self.value)

    __array_priority__ = 15

__try_interpret_as_with_unit = None
def init_base_unit_functions(try_interpret_as_with_unit):
    global __try_interpret_as_with_unit
    __try_interpret_as_with_unit = try_interpret_as_with_unit


copy_reg.pickle(
    WithUnit,
    lambda e: (raw_WithUnit, (
        e.value,
        e.numer,
        e.denom,
        e.exp10,
        e.base_units,
        e.display_units)))
