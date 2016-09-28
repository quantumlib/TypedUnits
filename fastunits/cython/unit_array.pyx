from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.object cimport Py_EQ, Py_NE, Py_LE, Py_GE, Py_LT, Py_GT
import copy
import copy_reg


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


def __unpickle_UnitArray(x):
    return UnitArray.raw(x)

copy_reg.pickle(
    UnitArray,
    lambda e: (__unpickle_UnitArray, (list(e),)))
