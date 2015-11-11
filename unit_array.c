#include <Python.h>
#include <structmember.h>
#include <stdio.h>
#include <string.h>

/*
 * Unit dictionary type.  This is the guts of the unit implementaiton,
 * and fills the same purpose as the NumberDict class used in the
 * python implementation.  This implementation is based on arrays, not
 * dictionaries.
 */


typedef struct {
    PyObject *name;
    int numer, denom;
} UnitName;

typedef struct {
    PyObject_VAR_HEAD
    UnitName data[0];
} UnitArray;

UnitArray *dimensionless=0;

typedef struct {
    PyObject_HEAD
    double value;
    long long numer, denom;
    int exp_10;
    UnitArray *base_units;
    UnitArray *display_units;
} ValueObject;


static PyObject *
unit_array_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static PyObject *
unit_array_repr(UnitArray *obj);
static PyObject *
unit_array_op(UnitArray *left, UnitArray *right, int sign_r);
static PyObject *
unit_array_mul(PyObject *a, PyObject *b);
static PyObject *
unit_array_div(PyObject *a, PyObject *b);
static PyTypeObject UnitArrayType;
static PyTypeObject ValueType;

static void
least_terms(int *a, int *b)
{
    int x=*a, y=*b, z;
    if (x==0) {
	*b = 1;
	return;
    }
    for(;;) {
	z = x%y;
	if (z==1)
	    return;
	if (z==0) {
	    *a /= y;
	    *b /= y;
	    return;
	}
	x = y;
	y = z;
    }
}

static long long gcd(long long a, long long b);

static PyObject *
unit_array_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    UnitArray *self;
    PyObject *name;
    static char *kwlist[] = {"name", "numer", "denom", 0};
    int rv;
    int numer=1, denom=1;

    rv = PyArg_ParseTupleAndKeywords(args, kwds, "S|ii", kwlist, 
				     &name, &numer, &denom);
    if (!rv)
	return 0;
    if (denom==0) {
	PyErr_SetString(PyExc_ValueError, "Denominator cannot be zero");
	return 0;
    }
    if (denom < 0) {
	denom = -denom;
	numer = -numer;
    }
    least_terms(&numer, &denom);
    self = (UnitArray *)type->tp_alloc(type, 1);
    if (self != NULL) {
	self->ob_size = 1;
	self->data[0].name = name;
	self->data[0].numer = numer;
	self->data[0].denom = denom;
	Py_INCREF(self->data[0].name);
    }
    return (PyObject *)self;
}

static PyObject *
unit_array_repr(UnitArray *obj)
{
    // Repr length: 2 braces + 1 null + N*(name + 6 + len(number)) - 2
    // repr format:
    // {'s': 1, 'km': 2}
    int i;
    PyObject *result;
    PyObject *tmp;

    if(obj->ob_size == 0) {
	result = PyString_FromString("{}");
	return result;
    }
    result = PyString_FromString("{");
    for(i=0; i<obj->ob_size-1; i++) {
	tmp = PyString_FromFormat("\"%s\": %d/%d, ", PyString_AsString(obj->data[i].name), obj->data[i].numer, obj->data[i].denom);
	if(!tmp) {
	    Py_DECREF(result);
	    return 0;
	}
	PyString_ConcatAndDel(&result, tmp);
	if(!result)
	    return 0;
    }
    tmp = PyString_FromFormat("\"%s\": %d/%d}", PyString_AsString(obj->data[i].name), obj->data[i].numer, obj->data[i].denom);
    if(!tmp) {
	Py_DECREF(result);
	return 0;
    }
    PyString_ConcatAndDel(&result, tmp);
    return result;
}

static PyObject *
unit_array_richcompare(PyObject *a, PyObject *b, int op)
{
    UnitArray *left, *right;
    int result=0;
    int i=0;

    if(!PyObject_IsInstance(a, (PyObject *)&UnitArrayType) || !PyObject_IsInstance(b, (PyObject *)&UnitArrayType)) {
	Py_INCREF(Py_NotImplemented);
	return Py_NotImplemented;
    }
    if (op != Py_EQ && op != Py_NE) {
	PyErr_SetString(PyExc_TypeError, "Can't compare value arrays except for equality");
	return 0;
    }
    left = (UnitArray *)a;
    right = (UnitArray *)b;
    if (left->ob_size == right->ob_size) {
	for (i=0; i<left->ob_size; i++) {
	    if (PyObject_RichCompareBool(left->data[i].name, right->data[i].name, Py_NE))
		break;
	    if (left->data[i].numer != right->data[i].numer)
		break;
	    if (left->data[i].denom != right->data[i].denom)
		break;
	}
	if (i == left->ob_size)
	    result = 1;
    }
    if ((op == Py_EQ && result) || (op == Py_NE && !result))
	Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
unit_array_div(PyObject *a, PyObject *b)
{
    if(!PyObject_IsInstance(a, (PyObject *)&UnitArrayType) || ! PyObject_IsInstance(b, (PyObject *)&UnitArrayType)) {
	Py_INCREF(Py_NotImplemented);
	return Py_NotImplemented;
    }
    return unit_array_op((UnitArray *) a, (UnitArray *)b, -1);
}

static PyObject *
unit_array_mul(PyObject *a, PyObject *b)
{
    if(!PyObject_IsInstance(a, (PyObject *)&UnitArrayType) || ! PyObject_IsInstance(b, (PyObject *)&UnitArrayType)) {
	Py_INCREF(Py_NotImplemented);
	return Py_NotImplemented;
    }
    return unit_array_op((UnitArray *) a, (UnitArray *)b, 1);
}

static PyObject *
unit_array_op(UnitArray *left, UnitArray *right, int sign_r)
{
    int idx_l=0, idx_r=0, idx_out=0, new_numer, new_denom;
    int cmp_val;

    UnitArray *out;
    // Compute the needed array size
    while(idx_l < left->ob_size || idx_r < right->ob_size) {
	if (idx_l == left->ob_size)
	    cmp_val = 1;
	else if (idx_r == right->ob_size)
	    cmp_val = -1;
	else
	    cmp_val = strcmp(PyString_AsString(left->data[idx_l].name), PyString_AsString(right->data[idx_r].name));
	if (cmp_val < 0)
	    idx_l += 1;
	if (cmp_val > 0)
	    idx_r += 1;
	if (cmp_val == 0) {
	    new_numer = left->data[idx_l].numer * right->data[idx_r].denom + sign_r * right->data[idx_r].numer * left->data[idx_l].denom;
	    if(!new_numer)
		idx_out -= 1;
	    idx_l++;
	    idx_r++;
	}
	idx_out += 1;
    }
    out =  (UnitArray *)(&UnitArrayType)->tp_alloc(&UnitArrayType, idx_out);

    if (!out)
	return 0;
    idx_l = idx_r = idx_out = 0;
    while(idx_l < left->ob_size || idx_r < right->ob_size) {
	if (idx_l == left->ob_size)
	    cmp_val = 1;
	else if (idx_r == right->ob_size)
	    cmp_val = -1;
	else {
	    cmp_val = strcmp(PyString_AsString(left->data[idx_l].name), PyString_AsString(right->data[idx_r].name));
	}
	if (cmp_val < 0) {
	    out->data[idx_out].name = left->data[idx_l].name;
	    out->data[idx_out].numer = left->data[idx_l].numer;
	    out->data[idx_out].denom = left->data[idx_l].denom;
	    Py_INCREF(out->data[idx_out].name);
	    idx_out++;
	    idx_l++;
	}
	if (cmp_val > 0) {
	    out->data[idx_out].name = right->data[idx_r].name;
	    out->data[idx_out].numer = sign_r * right->data[idx_r].numer;
	    out->data[idx_out].denom = right->data[idx_r].denom;
	    Py_INCREF(out->data[idx_out].name);
	    idx_out++;
	    idx_r++;
	}
	if (cmp_val == 0) {
	    new_numer = left->data[idx_l].numer * right->data[idx_r].denom + sign_r * right->data[idx_r].numer * left->data[idx_l].denom;
	    if (new_numer) {
		out->data[idx_out].name = left->data[idx_l].name;
		new_denom = left->data[idx_l].denom * right->data[idx_r].denom;
		least_terms(&new_numer, &new_denom);
		out->data[idx_out].numer = new_numer;
		out->data[idx_out].denom = new_denom;
		Py_INCREF(out->data[idx_out].name);
		idx_out++;
	    }
	    idx_l++;
	    idx_r++;
	}
    }
    out->ob_size = idx_out;
    return (PyObject *)out;
}

static int
rational_power(PyObject *a, long *numer, long *denom)
{
    double d;
    int x;

    if (PyObject_IsInstance(a,(PyObject *) &PyInt_Type)) {
	*numer = PyInt_AsLong(a);
	*denom = 1;
	return 1;
    }
    d = PyFloat_AsDouble(a);
    printf("rational power from %f\n", d);
    if (d==-1.0 && PyErr_Occurred()) 
	return 0;
    x = round(12*d);
    if ( fabs(12*d - x) < 1e-5 ) {
	long tmp = gcd(x, 12);
	*numer = x/tmp;
	*denom = 12/tmp;
	
	return 1;
    }
    PyErr_SetString(PyExc_ValueError, "power can't be converted to simple fraction (N/12)");
    return 0;
}

static
UnitArray *
unit_array_pow_frac(UnitArray *obj, int numer, int denom)
{
    UnitArray *result;
    int idx;

    printf("unit array fractional power: %d / %d\n", numer, denom);
    result =  (UnitArray *)(&UnitArrayType)->tp_alloc(&UnitArrayType, obj->ob_size);
    result->ob_size = obj->ob_size;
    for (idx=0; idx<obj->ob_size; idx++) {
	result->data[idx].name = obj->data[idx].name;
	Py_INCREF(result->data[idx].name);
	result->data[idx].numer = obj->data[idx].numer * numer;
	result->data[idx].denom = obj->data[idx].denom * denom;
	least_terms(&result->data[idx].numer, &result->data[idx].denom);
    }
    return result;
}

static
PyObject *
unit_array_pow(PyObject *a, PyObject *b, PyObject *c)
{
    long numer, denom;

    if (c != Py_None) {
	PyErr_SetString(PyExc_ValueError, "Value power does not support third argument");
	return 0;
    }

    if(!PyObject_IsInstance(a, (PyObject *)&UnitArrayType)) {
	PyErr_SetString(PyExc_TypeError, "UnitArray __pow__() requires left argument to be UnitArray");
	return 0;
    }
    if (!rational_power(b, &numer, &denom))
	return 0;
    return (PyObject *)unit_array_pow_frac((UnitArray *)a, numer, denom);
}

static void
unit_array_dealloc(UnitArray *self)
{
    int idx;
    for(idx=0; idx<self->ob_size; idx++) {
	Py_XDECREF(self->data[idx].name);
    }
    self->ob_type->tp_free((PyObject *)self);
}

static PyNumberMethods UnitArrayNumberMethods = {
    0,			/* nb_add */
    0,			/* nb_subtract */
    unit_array_mul,		/* nb_multiply */
    unit_array_div,		/* nb_divide */
    0,				/* nb_remainder */
    0,				/* nb_divmod */
    unit_array_pow,	       	/* nb_power */
    0,     	/* nb_negative */
    0,      	/* nb_positive */
    0,      	/* nb_absolute */
    0,	       	/* nb_nonzero (Used by PyObject_IsTrue */
    0				/* nb_invert */
};

static PyTypeObject UnitArrayType = {
    PyObject_HEAD_INIT(NULL) 
    0,			       /* ob_size */
    "UnitArray",	       /* tp_name */
    sizeof(UnitArray),         /*tp_basicsize*/
    sizeof(UnitName),          /*tp_itemsize*/
    (destructor)unit_array_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)unit_array_repr, /*tp_repr*/
    &UnitArrayNumberMethods,    /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_CHECKTYPES,        /*tp_flags*/
    "Unit Array object",        /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    unit_array_richcompare,    /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    0,		               /* tp_methods */
    0,	                       /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    unit_array_new             /* tp_new */
};

/*
 * Used by binary operators for coercion.  If the given object is a
 * Value, call Py_INCREF and return it.  Otherwise, convert to float,
 * and construct a temporary object with refcount 1 and dimensionless
 * units and return that.  On failure, returns zero, so make sure to
 * use Py_XDECREF on it.
 */
static ValueObject *
value_wrap(PyObject *obj)
{
    double val;
    ValueObject *result;

    if (PyObject_IsInstance(obj, (PyObject *)&ValueType)) {
	Py_INCREF(obj);
	return (ValueObject *)obj;
    }
    val = PyFloat_AsDouble(obj);
    if(PyErr_Occurred())
	return 0;
    result = (ValueObject *)ValueType.tp_alloc(&ValueType, 0);
    if (!result)
	return 0;
    result->value = val;
    result->numer = 1;
    result->denom = 1;
    result->exp_10 = 0;
    result->base_units = dimensionless;
    Py_INCREF(result->base_units);
    result->display_units = dimensionless;
    Py_INCREF(result->display_units);
    return result;
}

static PyObject *
value_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    double value=0.0;
    //double factor_base=1.0;
    //int factor_exp10=0;
    long long numer, denom;
    int exp_10;
    int rv;
    UnitArray *base_units=dimensionless, *display_units=dimensionless;
    ValueObject *obj;
    //char *kwlist[] = {"value", "factor", "exp10", "base_units", "display_units", 0};
    char *kwlist[] = {"value", "numer", "denom", "exp10", "base_units", "display_units", 0};

    rv = PyArg_ParseTupleAndKeywords(args, kwds, "d|LLiOO", kwlist, 
				     &value, &numer, &denom, &exp_10, &base_units, &display_units);
    if (!rv) 
	return 0;
    if(!PyObject_IsInstance((PyObject *)base_units, (PyObject *)&UnitArrayType) || !PyObject_IsInstance((PyObject *)display_units, (PyObject *)&UnitArrayType) ) {
	PyErr_SetString(PyExc_TypeError, "Base names and display names must be UnitArray objects");
	return 0;
    }
    obj = (ValueObject *)type->tp_alloc(type, 0);
    if (!obj)
	return 0;
    obj->value = value;
    obj->numer = numer;
    obj->denom = denom;
    obj->exp_10 = exp_10;
    obj->base_units = base_units;
    Py_INCREF(obj->base_units);
    obj->display_units = display_units;
    Py_INCREF(obj->display_units);
    return (PyObject *)obj;
}

static void
value_dealloc(ValueObject *obj)
{
    Py_XDECREF(obj->base_units);
    Py_XDECREF(obj->display_units);
}

/* This returns a clone of an old object with refcount 1, which can be modified
without compromising the immutable nature of Values. */
static
ValueObject *
value_clone(ValueObject *obj)
{
    ValueObject *result;

    result = (ValueObject *)obj->ob_type->tp_alloc(obj->ob_type, 0);  
    if (!result)
	return 0;
    result->value = obj->value;
    result->numer = obj->numer;
    result->denom = obj->denom;
    result->exp_10 = obj->exp_10;
    result->base_units = obj->base_units;
    Py_INCREF(result->base_units);
    result->display_units = obj->display_units;
    Py_INCREF(result->display_units);
    return result;
}

static
ValueObject *
value_neg(ValueObject *obj)
{
    ValueObject *result;
    result = value_clone(obj);
    if (!result)
	return 0;
    result->value = -result->value;
    return result;
}

static
PyObject *
value_pos(PyObject *obj)
{
    Py_INCREF(obj);
    return obj;
}

static
PyObject *
value_abs(ValueObject *obj)
{
    if (obj->value < 0)
	return (PyObject *)value_neg(obj);
    return value_pos((PyObject *)obj);
}

static
int
value_nz(ValueObject *obj)
{
    return obj->value != 0;
}

static
long long
gcd(long long a, long long b)
{
    if (a==0) return b;
    return gcd(b%a, a);
}

static
PyObject *
value_add(PyObject *a, PyObject *b)
{
    ValueObject *left, *right, *result, *example;
    double factor_l=1.0, factor_r=1.0;

    left = value_wrap(a);
    right = value_wrap(b);
    if(!left || !right) {
	goto fail;
    }
    if(!PyObject_RichCompareBool((PyObject *)left->base_units, (PyObject *)right->base_units, Py_EQ)) {
	PyErr_SetString(PyExc_ValueError, "UnitArray __add__ requires equivalent units");
	goto fail;
    }

    factor_l = left->numer * 1.0 / left->denom * exp10(left->exp_10);
    factor_r = right->numer * 1.0 / right->denom * exp10(right->exp_10);
    if (factor_l < factor_r) {
	example = left;
	factor_l = 1.0;
	factor_r = (1.0 * right->numer * left->denom) / (1.0 * left->numer * right->denom) * exp10(right->exp_10-left->exp_10);
    } 
    else {
	example = right;
	factor_l = (1.0 * left->numer * right->denom) / (1.0 * right->numer * left->denom) * exp10(left->exp_10-right->exp_10);
	factor_r = 1.0;
    }

    result = (ValueObject *)ValueType.tp_alloc(&ValueType, 0);
    if (!result)
	goto fail;
    
    result->value = left->value*factor_l + right->value*factor_r;
    result->numer = example->numer;
    result->denom = example->denom;
    result->exp_10 = example->exp_10;
    result->base_units = example->base_units;
    Py_INCREF(result->base_units);
    result->display_units = example->display_units;
    Py_INCREF(result->display_units);

    Py_XDECREF(left);
    Py_XDECREF(right);

    return (PyObject *)result;
 fail:
    if (!PyErr_Occurred())
	PyErr_SetString(PyExc_MemoryError, "value_add failed");
    Py_XDECREF(left);
    Py_XDECREF(right);
    return 0;
}

static 
PyObject *
value_sub(PyObject *a, PyObject *b)
{
    PyObject *right, *result;
    right = PyNumber_Negative(b);
    if (!right) {
	Py_INCREF(Py_NotImplemented);
	return Py_NotImplemented;
    }
    result = PyNumber_Add(a, right);
    return result;
}

static 
PyObject *
value_mul(PyObject *a, PyObject *b)
{
    ValueObject *left, *right, *result;
    long long gcd1, gcd2;

    left = value_wrap(a);
    right = value_wrap(b);
    if (!left || !right) {
	Py_XDECREF(left);
	Py_XDECREF(right);
    }
    result = (ValueObject *)ValueType.tp_alloc(&ValueType, 0);
    if (!result)
	return 0;
    result->value = left->value*right->value;
    gcd1 = gcd(left->numer, right->denom);
    gcd2 = gcd(right->numer, left->denom);
    result->numer = (left->numer/gcd1) * (right->numer/gcd2);
    result->denom = (left->denom/gcd2) * (right->denom/gcd1);
    result->exp_10 = left->exp_10 + right->exp_10;
    result->base_units = (UnitArray *)PyNumber_Multiply((PyObject *)left->base_units, (PyObject *)right->base_units);
    result->display_units = (UnitArray *)PyNumber_Multiply((PyObject *)left->display_units, (PyObject *)right->display_units);
    Py_XDECREF(left);
    Py_XDECREF(right);
    if (!result->numer || !result->denom || !result->base_units || !result->display_units) {
	PyErr_SetString(PyExc_MemoryError, "value __mul__ unable to create result");
	Py_XDECREF(result->base_units);
	Py_XDECREF(result->display_units);
	Py_XDECREF(result);
	return 0;
    }
    return (PyObject *) result;
}

static 
PyObject *
value_div(PyObject *a, PyObject *b)
{
    ValueObject *left, *right, *result;
    long long gcd1, gcd2;

    left = value_wrap(a);
    right = value_wrap(b);
    if (!left || !right) {
	Py_XDECREF(left);
	Py_XDECREF(right);
    }
    result = (ValueObject *)ValueType.tp_alloc(&ValueType, 0);
    if (!result) {
	Py_XDECREF(left);
	Py_XDECREF(right);
	return 0;
    }
    gcd1 = gcd(left->numer, right->numer);
    gcd2 = gcd(right->denom, left->denom);
    result->value = left->value/right->value;
    result->numer = (left->numer/gcd1) * (right->denom/gcd2);
    result->denom = (left->denom/gcd2) * (right->numer/gcd1);
    
    result->exp_10 = left->exp_10 - right->exp_10;
    result->base_units = (UnitArray *)PyNumber_Divide((PyObject *)left->base_units, (PyObject *)right->base_units);
    result->display_units = (UnitArray *)PyNumber_Divide((PyObject *)left->display_units, (PyObject *)right->display_units);
    Py_XDECREF(left);
    Py_XDECREF(right);
    if (!result->base_units || !result->display_units) {
	PyErr_SetString(PyExc_MemoryError, "value __div__ unable to create result");
	Py_XDECREF(result->base_units);
	Py_XDECREF(result->display_units);
	Py_XDECREF(result);
	return 0;
    }
    return (PyObject *) result;
}

static
long long ipow(long long x, int y)
{
    double tmp;

    tmp = pow(x, y);
    printf("%d power of %lld: %lld\n", y, x, (long long) tmp);
    return (long long) tmp;
}

static
long long iroot(long long x, int y)
{
    long long tmp;
    tmp = pow(x, 1.0/y);
    if (pow(tmp, y) != x) {
	printf("%d root of %lld not an integer\n", y, x);
	return 0;
    }
    printf("%d root of %lld: %lld\n", y, x, tmp);
    return tmp;
}

static
PyObject *
value_pow(PyObject *a, PyObject *b, PyObject *c)
{
    ValueObject *left, *result;
    long pow_numer, pow_denom;
    int pow_sign=1;

    if (c != Py_None) {
	PyErr_SetString(PyExc_ValueError, "Value power does not support third argument");
	return 0;
    }
    if (!PyObject_IsInstance(a, (PyObject *)&ValueType)) {
	PyErr_SetString(PyExc_TypeError, "Can only raise value type to integer power");
	return 0;
    }
    left = (ValueObject *)a;
    if (!rational_power(b, &pow_numer, &pow_denom))
	return 0;

    if (pow_numer < 0) {
	pow_numer = -pow_numer;
	pow_sign = -1;
    }
    printf("Doing rational power: %d * %ld/%ld\n", pow_sign, pow_numer, pow_denom);

    if ((left->exp_10 * pow_numer) % pow_denom) {
	PyErr_SetString(PyExc_RuntimeError, "Unable to take root of specifified unit\n");
	return 0;
    }

    result = (ValueObject *)ValueType.tp_alloc(&ValueType, 0);
    if(!result)
	return 0;

    result->value = pow(left->value, pow_sign * pow_numer*1.0/pow_denom);
    result->numer = iroot(left->numer, pow_denom);
    printf("numer after root: %lld\n", result->numer);
    result->numer = ipow(result->numer, pow_numer);
    printf("numer after raising: %lld\n", result->numer);
    result->denom = iroot(left->denom, pow_denom);
    result->denom = ipow(result->denom, pow_numer);
    result->exp_10 = left->exp_10 * pow_sign * pow_numer / pow_denom;
    if (pow_sign == -1) {
	long long tmp = result->numer;
	result->numer = result->denom;
	result->denom = tmp;
    }
    result->base_units = unit_array_pow_frac(left->base_units, pow_sign * pow_numer, pow_denom);
    result->display_units = unit_array_pow_frac(left->display_units, pow_sign * pow_numer, pow_denom);
    if (!result->base_units || !result->display_units || !result->numer || !result->denom) {
	PyErr_SetString(PyExc_RuntimeError, "Unable to take power");
	Py_XDECREF(result);
	return 0;
    }
    return (PyObject *)result;
}


static PyNumberMethods ValueNumberMethods = {
    value_add,			/* nb_add */
    value_sub,			/* nb_subtract */
    value_mul,		/* nb_multiply */
    value_div,			/* nb_divide */
    0,				/* nb_remainder */
    0,				/* nb_divmod */
    value_pow,	       	/* nb_power */
    (unaryfunc)value_neg,     	/* nb_negative */
    value_pos,      	/* nb_positive */
    (unaryfunc)value_abs,      	/* nb_absolute */
    (inquiry)value_nz,	       	/* nb_nonzero (Used by PyObject_IsTrue */
    0,				/* nb_invert */
};

static
PyObject *
value_richcompare(PyObject *a, PyObject *b, int op)
{
    ValueObject *left, *right, *diff;
    int rv;

    left = value_wrap(a);
    right = value_wrap(b);

     if (!left || !right) {
	Py_XDECREF(left);
	Py_XDECREF(right);
	Py_INCREF(Py_NotImplemented);
	return Py_NotImplemented;
     }
     if(!PyObject_RichCompareBool((PyObject *)left->base_units, (PyObject *)right->base_units, Py_EQ)) {
	 Py_XDECREF(left);
	 Py_XDECREF(right);
	 if (op == Py_EQ)
	     Py_RETURN_FALSE;
	 if (op == Py_NE)
	     Py_RETURN_TRUE;
	 PyErr_SetString(PyExc_ValueError, "UnitArray comparison requires equivalent units");
	 return 0;
     }
     diff = (ValueObject *)value_sub((PyObject *)left, (PyObject *)right);
     switch (op) {
     case Py_LT:
	 rv = diff->value < 0;
	 break;
     case Py_LE:
	 rv = diff->value <= 0;
	 break;
     case Py_EQ:
	 rv = diff->value == 0;
	 break;
     case Py_NE:
	 rv = diff->value != 0;
	 break;
     case Py_GE:
	 rv = diff->value >= 0;
	 break;
     case Py_GT:
	 rv = diff->value > 0;
	 break;
     default:
	Py_XDECREF(diff);
	PyErr_SetString(PyExc_RuntimeError, "Rich compare called with invalid operator");
	return 0;
     }
     Py_XDECREF(diff);
     if (rv)
	 Py_RETURN_TRUE;
     Py_RETURN_FALSE;
}

static
PyObject *
value_repr(ValueObject *obj)
{
    PyObject *result;
    PyObject *unit_repr_s;
    char value_repr_s[20];

    unit_repr_s=unit_array_repr(obj->display_units);
    if(!unit_repr_s)
	return 0;
    snprintf(value_repr_s, 20, "%g", obj->value);
    result = PyString_FromFormat("%s %s", value_repr_s, PyString_AsString(unit_repr_s));
    Py_XDECREF(unit_repr_s);
    return result;
}

static PyObject *
value_test_long_mul(PyObject *self, PyObject *args)
{
    PyObject *a, *b, *out;
    int rv, i;
    rv = PyArg_ParseTuple(args, "OO", &a, &b);
    if (!rv)
	return 0;
    for(i=0; i<1000; i++) {
	out = PyNumber_Add(a, b);
	Py_XDECREF(out);
    }
    Py_RETURN_NONE;
}

static PyObject *
value_test_int_mul(PyObject *self, PyObject *args)
{
    PyObject *a, *b, *result;
    int rv, i;
    long x1, x2;

    rv = PyArg_ParseTuple(args, "OO", &a, &b);
    if(!rv)
	return 0;
    x1 = PyInt_AsLong(a);
    x2 = x1;
    for(i=0; i<1000; i++) {
	x2 = x1 * x2;
    }
    result = PyInt_FromLong(x2);
    return result;
}

static PyMethodDef Value_methods[] = {
    {"test_long_mul", value_test_long_mul, METH_VARARGS, "test long multiplication"},
    {"test_int_mul", value_test_int_mul, METH_VARARGS, "test int multiplication"},
    {0}
};

static PyMemberDef Value_members[] = {
    {"value", T_DOUBLE, offsetof(ValueObject, value), READONLY, "Floating point value"},
    {"base_units", T_OBJECT_EX, offsetof(ValueObject, base_units), READONLY, "Units in base units"},
    {"display_units", T_OBJECT_EX, offsetof(ValueObject, display_units), READONLY, "Units for display"},
    {"numer", T_LONGLONG, offsetof(ValueObject, numer), READONLY, "Fractional part of ratio between base and display units"},
    {"denom", T_LONGLONG, offsetof(ValueObject, denom), READONLY, "Fractional part of ratio between base and display units"},
    {"exp10", T_INT, offsetof(ValueObject, exp_10), READONLY, "Power of 10 ratio between base and display units"},
    {NULL}};

static PyTypeObject ValueType = {
    PyObject_HEAD_INIT(NULL) 
    0,			       /* ob_size */
    "Value",		       /* tp_name */
    sizeof(ValueObject),         /*tp_basicsize*/
    0,          /*tp_itemsize*/
    (destructor)value_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)value_repr, /*tp_repr*/
    &ValueNumberMethods,    /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_CHECKTYPES,        /*tp_flags*/
    "Unit Array object",        /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    value_richcompare,	                       /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    Value_methods,		               /* tp_methods */
    Value_members,	                       /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    value_new             /* tp_new */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initunit_array(void) 
{
    PyObject* m;

    //    ValueType.tp_methods = Value_methods;
    //    ValueType.tp_as_number = &ValueNumberMethods;

    if (PyType_Ready(&UnitArrayType) < 0)
	return;
    if (PyType_Ready(&ValueType) < 0)
	return;
    m = Py_InitModule3("unit_array", 0, "Module that creates unit arrays");
    Py_INCREF(&UnitArrayType);
    Py_INCREF(&ValueType);
    dimensionless = (UnitArray *)UnitArrayType.tp_alloc(&UnitArrayType, 0);
    dimensionless->ob_size = 0;
    PyModule_AddObject(m, "UnitArray", (PyObject *)&UnitArrayType);
    PyModule_AddObject(m, "Value", (PyObject *)&ValueType);
    PyModule_AddObject(m, "DimensionlessUnit", (PyObject *)dimensionless);
}
