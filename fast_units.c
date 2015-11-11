#include <Python.h>
#include <structmember.h>
#include <stdio.h>

/*
 *  Fast_units
 *
 * This module implements low-overhead unit based arithmetic types.  The	*
 * purpose is to support a python implmentation of units while keeping basic    *
 * operations fast and correct.  The actual unit dictionary and conversions     *
 * will be handled in python.
 *
 * There are C types for Value, Complex, DimensionlessValue, and		*
 * DimensionlessComplex.  The C value type is defined as:			*
 *
 * typedef struct {
 *	pyObject_HEAD  // Macro for standard python object members		*
 *	double value;  // Base numeric value					*
 *	int exponent;  // power-of-10.  This is used for SI prefixes		*
 *	int unit_powers[SI_BASE_NR];
 * }
 *
 * unit_powers stores the power of each SI base unit.  Addition requires	*
 * matching units, while multiplication of values causes addition of the units  *
 * 
 * The exponent field is used to support exact representation of SI prefixes.   *
 * For instance, "10 cm" will be stored with a value=10, and an exponent=-2.    *
 * If we stored this as 0.1 m, that would not be exact as 0.1 is not		*
 * representable as a base-2 floating point.
 *
 * Complex is defined the same way, but contains real and imaginary parts,	*
 * 
 * DimensionlessValue/Complex are intended to implement the Value API, but be   *
 * normal float/complex as well.  The reason is so that an expression that      *
 * yields a dimensionless value can be directly used in non-unit aware		*
 * functions.  There are at least three ways to implement that, I don't know	*
 * which actually works best:							*
 *										*
 * 1) Subclass float and complex.  This is nice because we get the default      *
 *    implementation automatically, and everything in python that expects       *
 *    a number can handle float and complex.  It also means we automatically    *
 *    win operator precidence -- 1.0 * DimensionlessFloat(2.0) will call our    *
 *    __add__ function first.  Downsides: dependent on the internal C float     *
 *    API.  Hard to see how to track exponent.  Also, because of operator       *
 *    precidence, we might "infect" lots of non-unit aware code, causing        *
 *    undesirable overhead unless we eventually revert to plain float(). Finally *
 *    it is not possible to use multiple inheritance with the C api.  This      *
 *    means we wouldn't be able to subclass Float and Value/WithUnit.  We could *
 *    make WithUnit an ABC and register that in python.	      			*
 *										*
 * 2) Implement a __float__ method.  This allows explicit conversion to	native  *
 *    float.  Many operations are expected to use float() to coerce input to    *
 *    a floating point number, but maybe not everything.  We reduce the		*
 *    infection problem, but may have worse compatibility.			*
 * 
 * 3) Use the ABC module to register as a real/complex data type.  This will    *
 *    allow DimensionlessValue to pass "isinstance()" checks without having to  *
 *    rely on the PyFloat C implementation.  Disadvantages: ABC is new in 2.7   *
 *    so it may not work for everything, particularly C libraries, and it looks *
 *    like it is only accessible from python.  FWIW, decimal.Decimal takes this * 
 *    approach and subclasses numbers.Number, but not numbers.Real (yet still   *
 *    implements a __float__ method.  This causes Decimal(3.2)+1.5 to raise a   *
 *    TypeError, but still allows explicit float() casting.
 *
 * It seems like doing both options 2 and 3 is best. 
 *
 * Pint uses option 2.  In addition, it uses the same type for
 * dimensionless and non-dimensionless values.  Instead, __float__() raises an *
 * exception when the instance is not dimensionless.  This allows preserving   *
 * "dimensionless" quantities like 5 kHz/Hz, but converting to 5000.0 when     *
 * needed.
 * 
 * Pint does not use ABCs, so maybe we should just stick with option 2.
 */

/*
 * WithUnit API:
 *
 * unit:			Unit object
 * units			Unit string
 * inBaseUnits:			Convert to SI units
 * isCompatible			Check units
 * isDimensionless		Check units
 * __getitem__			Unit conversion
 * __reduce__			pickle
 * __copy__
 * __deepcopy__
 * __str__
 * __repr__
 * +, -, *, /, //, **, +x, -x	
 * <, >, <=, >=, ==, !=, !=0
 * sqrt()
 * complex() / float() / array()
 */
#define SI_BASE_NR 9

typedef struct {
    PyObject_HEAD
    double value;
    int exponent;
    int unit_powers[SI_BASE_NR];
} ValueObject;

typedef struct {
    PyObject_HEAD
    Py_complex value;
    int exponent;
    int unit_powers[SI_BASE_NR];
} ComplexObject;


typedef struct {
    PyObject_HEAD
    PyArrayObject *obj;
    int exponent;
    int unit_powers[SI_BASE_NR];
} ValueArray;

static PyTypeObject ComplexType;
static PyTypeObject ValueType;

int DimensionlessUnits[SI_BASE_NR] = {0,0,0,0,0,0,0,0,0};

static void
mul_unit_names(UnitName *dest, UnitName *a, UnitName *b)
{
    const char *left, *right;
    int x;

    left = PyString_AsString(a);
    right = PyString_AsString(b);
    for(;;) {
	x = strcmp(left, right);
	if (x < 0) {
	    dest->name = a->name;
	    PyIncRef(dest->name);
	    dest->power = a->power;
	    a++;
	    left = PyString_AsString(a);
	} else if (x > 0) {
	    dest->name = right->name;
	    PyIncRef(dest->name);
	    dest->power = right->power;
	    b++;
	    right = PyString_AsString(b);
	} else {
	    dest->name - left->name;
	    PyIncRef(dest->name);
	    dest->power = left->power + right->power;
	    a++;
	    b++;
	    left = PyString_AsString(a);
	    right = PyString_AsString(b);
	}
    }
}

static void
copy_unit_powers(int *dest, int *src)
{
    memcpy(dest, src, sizeof(int[SI_BASE_NR]));
}

static void
add_unit_powers(int *dest, int *a, int *b)
{
    int i;
    for(i=0; i<SI_BASE_NR; i++)
	dest[i] = a[i] + b[i];
}

static void
sub_unit_powers(int *dest, int *a, int *b)
{
    int i;
    for(i=0; i<SI_BASE_NR; i++)
	dest[i] = a[i] - b[i];
}

static void
init_unit_powers(int *dest)
{
    memset(dest, 0, sizeof(int[SI_BASE_NR]);
}

static ComplexObject *
Complex_new(PyTypeObject *type, Py_complex value, int exponent, int *unit_powers) 
{
    ComplexObject *self;
    int i;
    self = (ComplexObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
	self->value = value;
	self->exponent = exponent;
	copy_unit_powers(self->unit_powers, unit_powers)
    }
    return self;
}
/*
 * Create a new Value object.  This version takes native parameters rathern than PyObjects
 * so it can be called directly in other methods such as __add__.
 */
static ValueObject *
Value_new(PyTypeObject *type, double value, int exponent, int *unit_powers)
{
    ValueObject *self;
    int i;
    
    self = (ValueObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
	self->value = value;
	self->exponent = exponent;
	copy_unit_powers(self->unit_powers, unit_powers)
    }
    return self;
}

static ValueArrayObject *
VArray_new(PyTypeObject *type, PyObject *value, int exponent, int *unit_powers)
{
    ValueArrayObject *self;
    int i;
    if(!PyObject_IsInstance(value, PyArray_Type)) {
	PyErr_SetString("Can't make value array with non-array");
	return 0;
    }
    self = (ValueArrayObject *)type->tp_alloc(type, 0);
    self->value = value;
    self->exponent = exponent;
    copy_unit_powers(self->unit_powers, unit_powers)
    return self;
}

static PyObject *
Complex_new_Object(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    ComplexObject *self;
    Py_complex value;
    int exponent;
    int unit_powers[SI_BASE_NR] = {0};
    PyObject *unit_list=0;
    int i, rv;

    static char *kwlist[] = {"value", "exponent", "units", 0};

    rv = PyArg_ParseTupleAndKeywords(args, kwds, "Di|O", kwlist, 
				     &value, &exponent, &unit_list);
    if (!rv) 
	return 0;
    if (unit_list != NULL) {
	rv = PyArg_ParseTuple(unit_list, "iiiiiiiii", unit_powers, unit_powers+1, 
			      unit_powers+2, unit_powers+3, unit_powers+4, 
			      unit_powers+5, unit_powers+6, unit_powers+7, 
			      unit_powers+8);
	if (!rv)
	    return 0;
    }
    self = Complex_new(type, value, exponent, unit_powers);
    return (PyObject *)self;
 
}

static PyObject *
Value_new_Object(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    ValueObject *self;
    double value;
    int exponent;
    int unit_powers[SI_BASE_NR] = {0};
    PyObject *unit_list=0;
    int i;
    int rv;

    static char *kwlist[] = {"value", "exponent", "units", 0};

    rv = PyArg_ParseTupleAndKeywords(args, kwds, "di|O", kwlist, 
				     &value, &exponent, &unit_list);
    if (!rv) 
	return 0;
    if (unit_list != NULL) {
	rv = PyArg_ParseTuple(unit_list, "iiiiiiiii", unit_powers, unit_powers+1, 
			      unit_powers+2, unit_powers+3, unit_powers+4, 
			      unit_powers+5, unit_powers+6, unit_powers+7, 
			      unit_powers+8);
	if (!rv)
	    return 0;
    }

    self = Value_new(type, value, exponent, unit_powers);
    return (PyObject *)self;
}

static PyObject *
VArray_new_Object(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *value;
    ValueObject *self;
    int exponent;
    int unit_powers[SI_BASE_NR] = {0};
    PyObject *unit_list=0;
    int i;
    int rv;

    static char *kwlist[] = {"value", "exponent", "units"};

    rv = PyArg_ParseTupleAndKeywords(args, kwds, "Oi|O", kwlist, 
				     &value, &exponent, &unit_list);
    if (!rv) 
	return 0;
    if (unit_list != NULL) {
	rv = PyArg_ParseTuple(unit_list, "iiiiiiiii", unit_powers, unit_powers+1, 
			      unit_powers+2, unit_powers+3, unit_powers+4, 
			      unit_powers+5, unit_powers+6, unit_powers+7, 
			      unit_powers+8);
	if (!rv)
	    return 0;
    } else
    self = VArray_new(type, value, exponent, unit_powers);
}

static PyMemberDef Value_members[] = {
    {"value", T_DOUBLE, offsetof(ValueObject, value), 0,
     "Floating point value"},
    {"exponent", T_INT, offsetof(ValueObject, exponent), 0,
     "Power of 10 exponent"},
    {NULL}
};

static PyMemberDef Complex_members[] = {
    {"real", T_DOUBLE, offsetof(ComplexObject, value.real), 0, "Real part"},
    {"imag", T_DOUBLE, offsetof(ComplexObject, value.imag), 0, "Imaginary part"},
    {"exponent", T_INT, offsetof(ComplexObject, exponent), 0, "Power of 10"}};

static int
Value_init(ValueObject *self, PyObject *args, PyObject *kwds)
{
    return 0;
}

static int
Complex_init(ComplexObject *self, PyObject *args, PyObject *kwds)
{
    return 0;
}

static PyObject *
value_add(PyObject *a, PyObject *b);
static PyObject *
value_mul(PyObject *a, PyObject *b);
static PyObject *
value_rich_compare(PyObject *a, PyObject *b, int op);
static PyObject *
value_sub(PyObject *a, PyObject *b);
static PyObject *
value_div(PyObject *a, PyObject *b);
static ValueObject *
value_neg(ValueObject *a);
static PyObject *
value_pos(PyObject *a);
static PyObject *
value_abs(PyObject *a);
int
value_nz(ValueObject *a);

static PyObject *
complex_add(PyObject *a, PyObject *b);
static PyObject *
complex_mul(PyObject *a, PyObject *b);
static PyObject *
complex_rich_compare(PyObject *a, PyObject *b, int op);
static PyObject *
complex_sub(PyObject *a, PyObject *b);
static PyObject *
complex_div(PyObject *a, PyObject *b);
static ComplexObject *
complex_neg(ComplexObject *a);
static PyObject *
complex_pos(PyObject *a);
static ValueObject *
complex_abs(ComplexObject *a);
int
complex_nz(ComplexObject *a);

static ValueArrayObject *
varray_add(PyObject *a, PyObject *b);

static PyNumberMethods ValueArrayNumberMethods = {
    varray_add,
    varray_sub,
    varray_mul,
    varray_div,
    0,
    0,
    0,
    (unaryfunc)varray_neg,
    (unaryfunc)varray_pos,
    (unaryfunc)varray_abs,
    (inquiry)varray_nz,
    0
};

static PyNumberMethods ValueNumberMethods = {
    value_add,			/* nb_add */
    value_sub,			/* nb_subtract */
    value_mul,			/* nb_multiply */
    value_div,			/* nb_divide */
    0,				/* nb_remainder */
    0,				/* nb_divmod */
    0,				/* nb_power */
    (unaryfunc)value_neg,     	/* nb_negative */
    (unaryfunc)value_pos,      	/* nb_positive */
    (unaryfunc)value_abs,      	/* nb_absolute */
    (inquiry)value_nz,	       	/* nb_nonzero (Used by PyObject_IsTrue */
    0				/* nb_invert */
};
static PyNumberMethods ComplexNumberMethods = {
    complex_add,		/* nb_add */
    complex_sub,		/* nb_subtract */
    complex_mul,		/* nb_multiply */
    complex_div,       		/* nb_divide */
    0,				/* nb_remainder */
    0,				/* nb_divmod */
    0,				/* nb_power */
    (unaryfunc)complex_neg,				/* nb_negative */
    complex_pos,				/* nb_positive */
    (unaryfunc)complex_abs,				/* nb_absolute */
    (inquiry)complex_nz,				/* nb_nonzero (Used by PyObject_IsTrue */
    0				/* nb_invert */
};

static PyTypeObject ValueArrayType = {
    PyObject_HEAD_INIT(NULL)
    0,
    "ValueArray",
    sizeof(ValueArrayObject),
    0,                         /*tp_itemsize*/
    0,                         /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,			       /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_CHECKTYPES,        /*tp_flags*/
    "Value Array objects",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    varray_rich_compare,	       /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    0,		               /* tp_methods */
    VArray_members,	       /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)VArray_init,      /* tp_init */
    0,                         /* tp_alloc */
    VArray_new_Object,          /* tp_new */
};


static PyTypeObject ValueType = {
    PyObject_HEAD_INIT(NULL) 
    0,			       /* ob_size */
    "Value",		       /* tp_name */
    sizeof(ValueObject),       /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    0,                         /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,			       /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_CHECKTYPES,        /*tp_flags*/
    "Value objects",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    value_rich_compare,	       /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    0,		               /* tp_methods */
    Value_members,	       /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Value_init,      /* tp_init */
    0,                         /* tp_alloc */
    Value_new_Object,          /* tp_new */
};

static PyTypeObject ComplexType = {
    PyObject_HEAD_INIT(NULL) 
    0,			       /* ob_size */
    "Complex",		       /* tp_name */
    sizeof(ComplexObject),     /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    0,                         /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,			       /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_CHECKTYPES,        /*tp_flags*/
    "Complex objects",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    complex_rich_compare,      /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    0,		               /* tp_methods */
    Complex_members,           /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Complex_init,    /* tp_init */
    0,                         /* tp_alloc */
    Complex_new_Object,        /* tp_new */
};

static int check_units_match(int *left_units, int *right_units)
{
    int i;
    for(i=0; i<SI_BASE_NR; i++) {
	if (left_units[i] != right_units[i])
	    return 0;
    }
    return 1;
}

/*
 * This is an interesting helper function for type promotion.  It is
 * needed to handle cases like "Value() + 1j.  Both arguments need to
 * be promoted, and we need to call from the Value __add__ method to
 * the Complex version.  This method figures out if a given argument
 * can only be handled as a complex number, and therefore we should
 * call to the complex method to try.
 */
static int only_complex(PyObject *x)
{
    if (PyObject_IsInstance(x, (PyObject *)&ComplexType))
	return 1;
    if (PyComplex_Check(x) && !PyFloat_Check(x))
	return 1;
    return 0;
}

static ValueObject *promote_to_value(PyObject *x)
{
    double d;
    ValueObject *rv;

    if(PyObject_IsInstance(x, (PyObject *)&ValueType)) {
	Py_INCREF(x);
	return (ValueObject *) x;
    }
    d = PyFloat_AsDouble(x);
    if (d==-1.0 && PyErr_Occurred()) {
	PyErr_Clear();
	return 0;
    }
    rv = Value_new(&ValueType, d, 0, DimensionlessUnits);
    return rv;
}

static ComplexObject *promote_to_complex(PyObject *x)
{
    Py_complex val;
    ComplexObject *rv;
    
    if(PyObject_IsInstance(x, (PyObject *)&ComplexType)) {
	Py_INCREF(x);
	return (ComplexObject *) x;
    }
    if (PyObject_IsInstance(x, (PyObject *)&ValueType)) {
	ValueObject *unit_x = (ValueObject *)x;
	val.real = unit_x->value;
	val.imag = 0.0;
	rv = Complex_new(&ComplexType, val, unit_x->exponent, unit_x->unit_powers);
	return rv;
    } else {
	val = PyComplex_AsCComplex(x);
	if(val.real==-1.0 && val.imag==0 && PyErr_Occurred()) {
	    PyErr_Clear();
	    return 0;
	}
    }
    rv = Complex_new(&ComplexType, val, 0, DimensionlessUnits);
    return rv;
}

/*
 * This converts the input object to be something with units.  If the
 * input is a float or can be coerced to float (for int, etc.), we
 * create a dimensionless Value.  If the object cannot be coerced to a
 * float but is Complex or can be coerced to complex, we return
 * Complex.  If the object is Value, Complex, or ValueArray, we bump
 * the reference count and return it.
 *
 * The force_complex input causes scalar inputs to be converted to
 * Complex: either a Value object or a float scalar.
 */
static PyObject *
add_units(PyObject *x, int force_complex)
{
}
static int
coerce(PyObject *a, PyObject *b)
{
}
/*
 *  Promotion hierarchy
 * 
 *  We need separte promotion rules for addition and multiplication.  Addition  *
 *  can only happen between objects with the same units.  Therefore we only     *
 *  need to check for Value and Complex object types.  For multiplication we    *
 *  must support native float and complex types.
 *
 * Types that we need to handle:
 *
 * Value, Complex: we define these!
 * int, long, float, scalar numpy types:  all of these should have __float__ 
 *	method in PyNumberMethods, use that.
 * complex: PyComplex_Check: this handles complex objects and subclasses
 * complex protocol: __complex__ method -- for objects that act like complex 
 *	numbers but do not subclass complex
 */
static ValueArrayObject *
varray_add(PyObject *a, PyObject *b)
{
    /*
      We need to handle:
     */
}
static PyObject *
value_rich_compare(PyObject *a, PyObject *b, int op)
{
    ValueObject *left, *right;
    int rv;
    double left_val, right_val;

    if (only_complex(b) || only_complex(a))
	return complex_rich_compare(a, b, op);
    left = promote_to_value(a);
    right = promote_to_value(b);
    if (!left || !right) {
	Py_XDECREF(left);
	Py_XDECREF(right);
	Py_INCREF(Py_NotImplemented);
	return Py_NotImplemented;
    }

    if (!check_units_match(left->unit_powers, right->unit_powers)) {
	Py_XDECREF(left);
	Py_XDECREF(right);
	if (op == Py_EQ)
	    Py_RETURN_FALSE;
	else if(op == Py_NE)
	    Py_RETURN_TRUE;
	PyErr_SetString(PyExc_ValueError, "Units don't match");
	return 0;
    }

    if (left->exponent > right->exponent) {
	left_val = left->value * pow(10.0, left->exponent-right->exponent);
	right_val = right->value;
    } else {
	left_val = left->value;
	right_val = right->value * pow(10.0, right->exponent-left->exponent);
    }

    switch(op) {
    case Py_LT:
	rv = left_val < right_val;
	break;
    case Py_LE:
	rv = left_val <= right_val;
	break;
    case Py_EQ:
	rv = left_val == right_val;
	break;
    case Py_NE:
	rv = left_val != right_val;
	break;
    case Py_GE:
	rv = left_val >= right_val;
	break;
    case Py_GT:
	rv = left_val > right_val;
	break;
    default:
	Py_XDECREF(left);
	Py_XDECREF(right);
	PyErr_SetString(PyExc_RuntimeError, "Rich compare called with invalid operator");
	return 0;
    }
    Py_XDECREF(left);
    Py_XDECREF(right);
    if(rv)
	Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
complex_rich_compare(PyObject *a, PyObject *b, int op)
{
    ComplexObject *left, *right;
    Py_complex left_val, right_val;
    int rv;
    left = promote_to_complex(a);
    right = promote_to_complex(b);
        if (!left || !right) {
	Py_XDECREF(left);
	Py_XDECREF(right);
	Py_INCREF(Py_NotImplemented);
	return Py_NotImplemented;
    }
    if (!check_units_match(left->unit_powers, right->unit_powers)) {
	Py_XDECREF(left);
	Py_XDECREF(right);
	if (op == Py_EQ)
	    Py_RETURN_FALSE;
	else if(op == Py_NE)
	    Py_RETURN_TRUE;
	PyErr_SetString(PyExc_ValueError, "Units don't match");
	return 0;
    }
    if (left->exponent > right->exponent) {
	left_val.real = left->value.real * pow(10.0, left->exponent-right->exponent);
	left_val.imag = left->value.imag * pow(10.0, left->exponent-right->exponent);
	right_val = right->value;
    } else {
	left_val = left->value;
	right_val.real = right->value.real * pow(10.0, right->exponent-left->exponent);
	right_val.imag = right->value.imag * pow(10.0, right->exponent-left->exponent);
    }

    switch(op) {
    case Py_EQ:
	rv = (left_val.real == right_val.real) && (left_val.imag == right_val.imag);
	break;
    case Py_NE:
	rv = (left_val.real != right_val.real) || (left_val.imag != right_val.imag);
	break;
    default:
	Py_XDECREF(left);
	Py_XDECREF(right);
	PyErr_SetString(PyExc_TypeError, "Complex numbers don't support comparison");
	return 0;
    }
    Py_XDECREF(left);
    Py_XDECREF(right);
    if(rv)
	Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}
	
static PyObject *
complex_mul(PyObject *a, PyObject *b)
{
    ComplexObject *left, *right;
    ComplexObject *result=0;
    Py_complex new_val;
    int new_exp;
    int i;
    int new_unit_pow[SI_BASE_NR];

    left = promote_to_complex(a);
    right = promote_to_complex(b);

    if (!left || !right) {
	Py_XDECREF(left);
	Py_XDECREF(right);
	Py_INCREF(Py_NotImplemented);
	return Py_NotImplemented;
    }
    new_val = _Py_c_prod(left->value, right->value);
    new_exp = left->exponent + right->exponent;
    add_unit_powers(new_unit_pow, left->unit_powers, right->unit_powers);
    result = Complex_new(&ComplexType, new_val, new_exp, new_unit_pow);
    return (PyObject *)result;
}

static PyObject *
complex_div(PyObject *a, PyObject *b)
{
    ComplexObject *rhs, *inv;
    PyObject *result;
    int unit_powers[SI_BASE_NR], i;
    Py_complex val;
    Py_complex one = {1.0, 0.0};

    rhs = promote_to_complex(b);
    if(!rhs) {
	Py_INCREF(Py_NotImplemented);
	return Py_NotImplemented;
    }
    if ((rhs->value.real == 0) && (rhs->value.imag == 0.0)) {
	Py_DECREF(rhs);
	PyErr_SetString(PyExc_ZeroDivisionError, "complex division by zero");
	return 0;
    }
    sub_unit_powers(unit_powers, DimensionlessUnits, right->unit_powers);
    val = _Py_c_quot(one, rhs->value);
    inv = Complex_new(rhs->ob_type, val, -rhs->exponent, unit_powers);
    Py_DECREF(rhs);
    result = complex_mul(a, (PyObject *)inv);
    Py_DECREF(inv);
    return  result;
}
static PyObject *
value_mul(PyObject *a, PyObject *b)
{
    ValueObject *left, *right;
    ValueObject *result;
    double new_val;
    int new_exp;
    int i;
    int new_unit_pow[SI_BASE_NR];

    if (only_complex(b) || only_complex(a))
	return complex_mul(a, b);
    left = promote_to_value(a);
    right = promote_to_value(b);
    
    if (!left || !right) {
	Py_XDECREF(left);
	Py_XDECREF(right);
	Py_INCREF(Py_NotImplemented);
	return Py_NotImplemented;
    }
    new_val = left->value * right->value;
    new_exp = left->exponent + right->exponent;
    add_unit_powers(new_unit_pow, left->unit_powers, right->unit_powers);
    result = Value_new(&ValueType, new_val, new_exp, new_unit_pow);
    return (PyObject *)result;
}

static PyObject *
complex_add(PyObject *left, PyObject *right)
{
    ComplexObject *a, *b, *result;
    Py_complex new_val;
    int new_exp;
    
    a = promote_to_complex(left);
    b = promote_to_complex(right);
    if (!a || !b) {
	Py_XDECREF(a);
	Py_XDECREF(b);
	Py_INCREF(Py_NotImplemented);
	return Py_NotImplemented;
    }
    if (!check_units_match(a->unit_powers, b->unit_powers)) {
	Py_XDECREF(a);
	Py_XDECREF(b);
	PyErr_SetString(PyExc_ValueError, "Units don't match");
	return 0;
    }
    if (a->exponent < b->exponent) {
	new_exp = a->exponent;
	new_val.real = a->value.real + b->value.real * pow(10.0, b->exponent-a->exponent);
	new_val.imag = a->value.imag + b->value.imag * pow(10.0, b->exponent-a->exponent);
    } else {
	new_exp = b-> exponent;
	new_val.real = b->value.real + a->value.real * pow(10.0, a->exponent-b->exponent);
	new_val.imag = b->value.imag + a->value.imag * pow(10.0, a->exponent-b->exponent);
    }
    result = Complex_new(&ComplexType, new_val, new_exp, a->unit_powers);
    Py_XDECREF(a);
    Py_XDECREF(b);
    return (PyObject *)result;


}
static PyObject *
complex_sub(PyObject *a, PyObject *b)
{
    return complex_add(a, PyNumber_Negative(b));
}

static ComplexObject *
complex_neg(ComplexObject *self)
{
    Py_complex tmp;
    tmp.real = -self->value.real;
    tmp.imag = -self->value.imag;
    return Complex_new(self->ob_type, tmp, self->exponent, self->unit_powers);
}

static PyObject *
complex_pos(PyObject *x)
{
    Py_INCREF(x);
    return x;
}

static ValueObject *
complex_abs(ComplexObject *x)
{
    double tmp;
    tmp = hypot(x->value.real, x->value.imag);
    return Value_new(&ValueType, tmp, x->exponent, x->unit_powers);
}

int
complex_nz(ComplexObject *a)
{
    if ((a->value.real == 0) && (a->value.imag == 0))
	return 0;
    return 1;
}

static PyObject *
value_add(PyObject *a, PyObject *b)
{
    ValueObject *left, *right;
    ValueObject *result;
    double new_val;
    int new_exp;

    if (only_complex(b) || only_complex(a))
	return complex_add(a, b);
    left = promote_to_value(a);
    right = promote_to_value(b);
    
    if (!left || !right) {
	Py_XDECREF(left);
	Py_XDECREF(right);
	Py_INCREF(Py_NotImplemented);
	return Py_NotImplemented;
    }
    if (!check_units_match(left->unit_powers, right->unit_powers)) {
	Py_XDECREF(left);
	Py_XDECREF(right);
	PyErr_SetString(PyExc_ValueError, "Units don't match");
	return 0;
    }
    if (left->exponent < right->exponent) {
	new_exp = left->exponent;
	new_val = left->value + right->value * pow(10.0, right->exponent-left->exponent);
    } else {
	new_exp = right-> exponent;
	new_val = right->value + left->value * pow(10.0, left->exponent-right->exponent);
    }
    result = Value_new(&ValueType, new_val, new_exp, left->unit_powers);
    Py_XDECREF(left);
    Py_XDECREF(right);
    return (PyObject *)result;
}

static PyObject *
value_sub(PyObject *a, PyObject *b)
{
    return value_add(a, PyNumber_Negative(b));
}

static PyObject *
value_div(PyObject *a, PyObject *b)
{
    ValueObject *right = promote_to_value(b);
    PyObject *inv, *result;
    int i;
    int unit_powers[SI_BASE_NR];

    if (!right) {
	Py_INCREF(Py_NotImplemented);
	return Py_NotImplemented;
    }
    if(right->value == 0) {
	Py_XDECREF(right);
	PyErr_SetString(PyExc_ZeroDivisionError, "value division by zero");
	return 0;
    }
    sub_unit_powers(unit_powers, DimensionlessUnits, right->unit_powers);
    inv = (PyObject *) Value_new(b->ob_type, 1.0 / right->value, -right->exponent, unit_powers);
    Py_XDECREF(right);
    result = value_mul(a, inv);
    Py_XDECREF(inv);
    return result;
}

static ValueObject *
value_neg(ValueObject *self)
{
    return Value_new(self->ob_type, -self->value, self->exponent, self->unit_powers);
}

static PyObject *
value_pos(PyObject *a)
{
    Py_INCREF(a);
    return a;
}

static PyObject *
value_abs(PyObject *a)
{
    ValueObject *self = (ValueObject *)a;
    if (self->value < 0)
	return (PyObject *)Value_new(self->ob_type, -self->value, self->exponent, self->unit_powers);
    Py_INCREF(a);
    return a;
}

int
value_nz(ValueObject *a)
{
    return (a->value != 0);
}

static PyObject *
get_value_unit_power(ValueObject *self, PyObject *args)
{
    int index;
    PyObject *rv;

    PyArg_ParseTuple(args, "i", &index);
    if (index < 0 || index > 8) {
	PyErr_SetString(PyExc_IndexError, "Unit index must be 0..8");
	return 0;
    }
    rv = Py_BuildValue("i", (int)(self->unit_powers[index]));
    return rv;
}

static PyMethodDef Value_methods[] = {
    {"get_unit_power", (PyCFunction)get_value_unit_power, METH_VARARGS,
     "Return the power of the unit with the specified index"},
    {NULL}  /* Sentinel */
};

static PyObject *
get_complex_unit_power(ComplexObject *self, PyObject *args)
{
    int index;
    PyObject *rv;

    PyArg_ParseTuple(args, "i", &index);
    if (index < 0 || index > 8) {
	PyErr_SetString(PyExc_IndexError, "Unit index must be 0..8");
	return 0;
    }
    rv = Py_BuildValue("i", (int)(self->unit_powers[index]));
    return rv;
}

static PyMethodDef Complex_methods[] = {
    {"get_unit_power", (PyCFunction)get_complex_unit_power, METH_VARARGS,
     "Return the power of the unit with the specified index"},
    {NULL}  /* Sentinel */
};

static PyMethodDef value_module_methods[] = {
    {NULL} /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initfast_units(void) 
{
    PyObject* m;

    ValueType.tp_methods = Value_methods;
    ValueType.tp_as_number = &ValueNumberMethods;
    ComplexType.tp_methods = Complex_methods;
    ComplexType.tp_as_number = &ComplexNumberMethods;
    if (PyType_Ready(&ValueType) < 0)
        return;
    if (PyType_Ready(&ComplexType) < 0)
	return;
    m = Py_InitModule3("fast_units", value_module_methods,
                       "Example module that creates an extension type.");

    Py_INCREF(&ValueType);
    Py_INCREF(&ComplexType);
    PyModule_AddObject(m, "Value", (PyObject *)&ValueType);
    PyModule_AddObject(m, "Complex", (PyObject *)&ComplexType);
}
