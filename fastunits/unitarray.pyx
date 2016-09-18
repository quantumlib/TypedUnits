from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.object cimport Py_EQ, Py_NE, Py_LE, Py_GE, Py_LT, Py_GT
import copy
import copy_reg
import math
import numpy as np

cimport cython
from libc.math cimport floor as c_floor, pow as c_pow


# A ratio that should always be canonicalized into least terms with the sign
# on the numerator.
cdef struct frac:
    long long numer
    long long denom


# Returns the greatest divisor that both inputs share.
@cython.cdivision(True)
cpdef long long gcd(long long a, long long b):
    if a < 0 or b < 0:
       return gcd(abs(a), abs(b))
    if a == 0:
        return b
    return gcd(b % a, a)


# Returns an equivalent fraction, without common factors between numerator and
# denominator and with the negative sign on the numerator (if present).
@cython.cdivision(True)
cpdef frac frac_least_terms(long long numer, long long denom) except *:
    if denom < 0:
        return frac_least_terms(-numer, -denom)
    if denom == 0:
        raise ZeroDivisionError()
    cdef long long d = gcd(numer, denom)
    cdef frac f
    f.numer = numer / d
    f.denom = denom / d
    return f


# Returns the product of the two given fractions, in least terms.
@cython.cdivision(True)
cpdef frac frac_times(frac a, frac b):
    cdef long long d1 = gcd(a.numer, b.denom)
    cdef long long d2 = gcd(b.numer, a.denom)
    cdef frac f
    f.numer = (a.numer/d1) * (b.numer/d2)
    f.denom = (a.denom/d2) * (b.denom/d1)
    return f


# Returns the quotient of the two given fractions, in least terms.
@cython.cdivision(True)
cpdef frac frac_div(frac a, frac b) except *:
    if b.numer == 0:
        raise ZeroDivisionError()
    cdef long long d1 = gcd(a.numer, b.numer)
    cdef long long d2 = gcd(b.denom, a.denom)
    cdef frac f
    f.numer = (a.numer/d1) * (b.denom/d2)
    f.denom = (a.denom/d2) * (b.numer/d1)
    if f.denom < 0:
        f.numer *= -1
        f.denom *= -1
    return f


# Recognizes floats corresponding to twelths. Returns them as a fraction.
cpdef frac float_to_twelths_frac(a) except *:
    if isinstance(a, int):
        return frac_least_terms(a, 1)

    cdef double d = float(a)
    cdef long long x = <long long>c_floor(12*d + 0.5)
    if abs(12*d - x) > 1e-5:
        raise ValueError("Not a twelfth.")

    return frac_least_terms(x, 12)


# Converts a fraction to a double approximating its value.
cpdef double frac_to_double(frac f):
    return <double>f.numer / <double>f.denom


cdef long long iroot(long long x, int exponent_denom):
    cdef long long tmp = <long long>c_pow(x, 1.0/exponent_denom)
    if c_pow(tmp, exponent_denom) != x:
        raise ValueError("%s root of %s not an integer" % (exponent_denom, x))
    return tmp


# A symbol raised to a power.
cdef struct UnitTerm:
    PyObject *name
    frac power


cdef class UnitArray:
    """
    A list of physical units raised to various powers.
    """
    cdef UnitTerm *units
    cdef int unit_count

    def __cinit__(self, str name = None):
        if name is not None:
            # Singleton unit array.
            self.unit_count = 1
            self.units = <UnitTerm *>PyMem_Malloc(sizeof(UnitTerm))
            if self.units == NULL:
                raise RuntimeError("Malloc failed")
            Py_INCREF(name)
            self.units[0].name = <PyObject *>name
            self.units[0].power.numer = 1
            self.units[0].power.denom = 1
        # else default to empty unit array
        # (the calling Cython code may do some non-empty initialization)

    @staticmethod
    def raw(name_numer_denom_tuples):
        """
        :param list((name, power.numer, power.denom)) name_numer_denom_tuples:
            The list of properties that units in the resulting list should have.
        :return UnitArray:
        """
        cdef int n = len(name_numer_denom_tuples)
        cdef UnitArray result = UnitArray()
        result.units = <UnitTerm *>PyMem_Malloc(sizeof(UnitTerm) * n)
        if result.units == NULL:
            raise RuntimeError("Malloc failed")

        cdef str name
        cdef long long numer
        cdef long long denom
        cdef UnitTerm* dst
        for name, numer, denom in name_numer_denom_tuples:
            dst = result.units + result.unit_count
            dst.power = frac_least_terms(numer, denom)
            dst.name = <PyObject *>name
            Py_INCREF(name)
            result.unit_count += 1

        return result

    def __dealloc__(self):
        cdef int i
        for i in range(self.unit_count):
            Py_DECREF(<str>self.units[i].name)
        if self.units:
            PyMem_Free(self.units)

    def __len__(UnitArray self):
        return self.unit_count

    def __getitem__(UnitArray self, int index):
        if index < 0 or index >= self.unit_count:
            raise IndexError()
        cdef UnitTerm unit = self.units[index]
        return <str>unit.name, unit.power.numer, unit.power.denom

    def __iter__(self):
        cdef int i
        for i in range(self.unit_count):
            yield self[i]

    def __repr__(self):
        return 'UnitArray.raw(%s)' % repr(list(self))

    def __str__(self):
        def tup_str(tup):
            name, numer, denom = tup
            numer = abs(numer)
            if numer == 1 and denom == 1:
                return name
            if denom == 1:
                return "%s^%d" % (name, numer)
            return "%s^(%d/%d)" % (name, numer, denom)

        times = '*'.join(tup_str(e) for e in self if e[1] > 0)
        divisions = ''.join('/' + tup_str(e) for e in self if e[1] < 0)
        if not divisions:
            return times
        return (times or '1') + divisions

    def __richcmp__(a, b, int op):
        if op != Py_EQ and op != Py_NE:
            return NotImplemented
        match = op == Py_EQ
        if not isinstance(a, UnitArray) or not isinstance(b, UnitArray):
            return not match
        cdef UnitArray left = a
        cdef UnitArray right = b
        if left.unit_count != right.unit_count:
            return not match
        cdef int i
        for i in range(left.unit_count):
            if <str>left.units[i].name != <str>right.units[i].name:
                return not match
            if left.units[i].power.numer != right.units[i].power.numer:
                return not match
            if left.units[i].power.denom != right.units[i].power.denom:
                return not match
        return match

    def __mul__(UnitArray a, UnitArray b):
        return a.__times_div(b, +1)

    def __div__(UnitArray a, UnitArray b):
        return a.__times_div(b, -1)

    def __times_div(UnitArray left, UnitArray right, int sign_r):
        # Compute the needed array size
        cdef UnitTerm *a = left.units
        cdef UnitTerm *b = right.units
        cdef int out_count = 0
        cdef UnitTerm *a_end = left.units + left.unit_count
        cdef UnitTerm *b_end = right.units + right.unit_count
        while a != a_end or b != b_end:
            a_name = None if a == a_end else <str>a.name
            b_name = None if b == b_end else <str>b.name

            if a_name == b_name:
                if a.power.numer * b.power.denom \
                        + sign_r * b.power.numer * a.power.denom != 0:
                    out_count += 1
                a += 1
                b += 1
            elif b_name is None or (a_name is not None and a_name < b_name):
                a += 1
                out_count += 1
            else:
                b += 1
                out_count += 1

        cdef UnitArray out = UnitArray()
        out.units = <UnitTerm *>PyMem_Malloc(sizeof(UnitTerm) * out_count)
        if out.units == NULL:
            raise RuntimeError("Malloc failed")

        a = left.units
        b = right.units
        a_end = left.units + left.unit_count
        b_end = right.units + right.unit_count
        cdef int i = 0
        cdef long long new_numer
        cdef long long new_denom
        while a != a_end or b != b_end:
            a_name = None if a == a_end else <str>a.name
            b_name = None if b == b_end else <str>b.name

            if a_name == b_name:
                new_numer = a.power.numer * b.power.denom \
                        + sign_r * b.power.numer * a.power.denom
                if new_numer != 0:
                    out.units[i].name = a.name
                    Py_INCREF(a_name)
                    out.unit_count += 1
                    new_denom = a.power.denom * b.power.denom
                    out.units[i].power = frac_least_terms(new_numer, new_denom)
                    i += 1
                a += 1
                b += 1
            elif b_name is None or (a_name is not None and a_name < b_name):
                out.units[i] = a[0]
                Py_INCREF(a_name)
                out.unit_count += 1
                a += 1
                i += 1
            else:
                out.units[i] = b[0]
                Py_INCREF(b_name)
                out.unit_count += 1
                out.units[i].power.numer *= sign_r
                b += 1
                i += 1

        return out

    cdef pow_frac(UnitArray self, frac exponent):
        if exponent.numer == 0:
            return DimensionlessUnit
        cdef UnitArray result = UnitArray()
        result.units = <UnitTerm*>PyMem_Malloc(sizeof(UnitTerm)*self.unit_count)
        if result.units == NULL:
            raise RuntimeError("Malloc failed")
        cdef UnitTerm *p
        cdef int i
        for i in range(self.unit_count):
            p = result.units + i
            p[0] = self.units[i]
            Py_INCREF(<str>p[0].name)
            result.unit_count += 1
            p[0].power = frac_times(p[0].power, exponent)
        return result

    def __pow__(UnitArray self, exponent, modulo):
        if modulo is not None:
            raise ValueError("UnitArray power does not support third argument")
        return self.pow_frac(float_to_twelths_frac(exponent));

DimensionlessUnit = UnitArray()


cdef raw_WithUnit(value,
                  long long numer,
                  long long denom,
                  int exp10,
                  UnitArray base_units,
                  UnitArray display_units):
    if not (numer > 0 and denom > 0):
        raise ValueError("Numerator and denominator must both be positive.")

    # Unwrap a Unit instance, which we can't talk about directly here
    # because it's in python code that depends on us, into a WithUnit value.
    if hasattr(value, '_value') and isinstance(value._value, WithUnit):
        return value._value

    # Choose derived class type.
    if isinstance(value, WithUnit):
        return value
    elif isinstance(value, complex):
        val = value
        target_type = Complex
    elif isinstance(value, np.ndarray):
        val = value
        target_type = ValueArray
    elif isinstance(value, list):
        val = np.array(value)
        target_type = ValueArray
    else:  # int or float or other
        val = float(value)
        target_type = Value

    cdef WithUnit result = target_type(None)
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
        :param value: The value. An int, float, complex, or ndarray.
        :param unit: A representation of the physical units. Could be an
            instance of Unit, or UnitArray, or a string to be parsed.
        """
        if isinstance(value, list):
            value = np.array(value)
        if unit is None and not isinstance(value, WithUnit):
            self.value = value
            self.base_units = DimensionlessUnit
            self.display_units = DimensionlessUnit
            self.exp10 = 0
            self.ratio.numer = 1
            self.ratio.denom = 1
            return

        cdef WithUnit unit_val
        if unit is None:
            unit_val = WithUnit(1)
        elif isinstance(unit, WithUnit):
            unit_val = unit
        elif isinstance(unit, str):
            unit_val = __unit_val_from_str(unit)
        else:
            unit_val = unit._value
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
        return raw_WithUnit(obj, 1, 1, 0, DimensionlessUnit, DimensionlessUnit)

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

        cdef long long numer = <long long>c_pow(iroot(left.ratio.numer, pow_frac.denom), abs_numer)
        cdef long long denom = <long long>c_pow(iroot(left.ratio.denom, pow_frac.denom), abs_numer)
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
        return "%s %s" % (str(self.value), str(self.display_units))

    def __repr__(self):
        return "WithUnit.raw(%s)" % ', '.join(repr(e) for e in [
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
        return WithUnit.raw(new_value, 1, 1, 0, self.base_units, self.base_units)

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
        cdef WithUnit unit_val
        if isinstance(key, int) or isinstance(key, slice):
            return self.__with_value(self.value[key])
        if isinstance(key, str):
            unit_val = __unit_val_from_str(key)
        elif isinstance(key, WithUnit):
            unit_val = key
        else:
            unit_val = key._value
#            raise ValueError("Bad unit key")

        if self.base_units != unit_val.base_units:
            raise UnitMismatchError("Value doesn't match specified units.")


        return (self.value
            * frac_to_double(frac_div(self.ratio, unit_val.ratio))
            * c_pow(10.0, self.exp10 - unit_val.exp10)
            / unit_val.value)

    def isCompatible(self, unit):
        cdef WithUnit other
        if isinstance(unit, str):
            other = __unit_val_from_str(unit)
        elif isinstance(unit, WithUnit):
            other = unit
        else:
            other = unit._value
        return self.base_units == other.base_units

    def inUnitsOf(WithUnit self, unit):
        cdef WithUnit unit_val
        if isinstance(unit, str):
            unit_val = __unit_val_from_str(unit)
        else:
            unit_val = unit._value
        return unit_val.__with_value(self[unit_val])

    def __hash__(self):
        # TODO: ANYONE CALLING THIS ALMOST CERTAINLY HAS A BUG RELATED TO
        #       FLOATING POINT ERROR PERTURBING THEIR KEYS
        return hash(self.inBaseUnits().value)

    property unit:
        def __get__(self):
            return __unit(self)

    def __array__(self):
        if self.base_units.unit_count != 0:
            raise UnitMismatchError(
                "Only dimensionless values can be stripped into an array.")
        return np.array(self._scale_to_double() * self.value)

    __array_priority__ = 15

__unit = None
__unit_val_from_str = None
def init_base_unit_functions(unit, unit_val_from_str):
    global __unit
    global __unit_val_from_str
    __unit = unit
    __unit_val_from_str = unit_val_from_str


class Value(WithUnit):
    _numType = float


class Complex(WithUnit):
    _numType = complex


class ValueArray(WithUnit):
    _numType = np.array # Regular ndarray constructor doesn't work

    def __setitem__(WithUnit self, key, val):
        cdef WithUnit right = WithUnit.wrap(val)
        if self.base_units != right.base_units:
            raise UnitMismatchError("Item's units don't match array's units.")
        cdef double f = self._scale_to_double() / right._scale_to_double()
        self.value[key] = right.value * f

    def __copy__(WithUnit self):
        return self.__with_value(copy.copy(self.value))

    def __deepcopy__(WithUnit self, memo):
        return self.__with_value(copy.deepcopy(self.value))

    def __iter__(WithUnit self):
        for e in self.value:
            yield self.__with_value(e)

    def __len__(self):
        return len(self._value)

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def shape(self):
        return self.value.shape

    def allclose(self, other, *args, **kw):
        return np.allclose(self.value, other[self.unit], *args, **kw)

class UnitMismatchError(TypeError):
    pass


def __unpickle_UnitArray(x):
    return UnitArray.raw(x)


def __unpickle_WithUnit(*x):
    return WithUnit.raw(*x)


copy_reg.pickle(
    UnitArray,
    lambda e: (__unpickle_UnitArray, (list(e),)))

copy_reg.pickle(
    WithUnit,
    lambda e: (__unpickle_WithUnit, (
        e.value,
        e.numer,
        e.denom,
        e.exp10,
        e.base_units,
        e.display_units)))
