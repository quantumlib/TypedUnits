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

from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.object cimport Py_EQ, Py_NE, Py_LE, Py_GE, Py_LT, Py_GT
import copy


# A symbol raised to a power.
cdef struct UnitTerm:
    PyObject *name
    frac power


cpdef raw_UnitArray(name_numer_denom_tuples):
    """
    A factory method the creates and directly sets the items of a UnitArray.
    (__init__ couldn't play this role for backwards-compatibility reasons.)

    :param list((name, power.numer, power.denom)) name_numer_denom_tuples:
        The list of properties that units in the resulting list should have.
    :return UnitArray:
    """
    cdef int n = len(name_numer_denom_tuples)
    if n == 0:
        return _EmptyUnit
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
        return 'raw_UnitArray(%s)' % repr(list(self))

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

    def __truediv__(UnitArray a, UnitArray b):
        return a.__times_div(b, -1)

    def __times_div(UnitArray left, UnitArray right, int sign_r):
        if right.unit_count == 0:
            return left
        if left.unit_count == 0 and sign_r == 1:
            return right

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
            return _EmptyUnit
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

    def __getstate__(self):
        return {
            'unit_count': self.unit_count,
            'units': [*self],
        }

    def __setstate__(self, pickle_info: dict[str, Any]):
        self.unit_count = pickle_info['unit_count']
        self.units = <UnitTerm *>PyMem_Malloc(self.unit_count*sizeof(UnitTerm))
        for i, (name, numer, denom) in enumerate(pickle_info['units']):
            Py_INCREF(name)
            self.units[i].name = <PyObject *>name
            self.units[i].power.numer = numer
            self.units[i].power.denom = denom

_EmptyUnit = UnitArray()
