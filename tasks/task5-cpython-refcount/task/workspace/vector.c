#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {
    PyObject_HEAD
    PyObject **items;
    Py_ssize_t size;
    Py_ssize_t capacity;
} VectorObject;

static void
Vector_dealloc(VectorObject *self)
{
    /* Release all stored objects */
    /* TODO: consider using PyObject_GC_Del for tracked objects */
    for (Py_ssize_t i = 0; i < self->size; i++) {
        Py_XDECREF(self->items[i]);  /* BUG_CANDIDATE: XDECREF vs DECREF debate */
    }
    PyMem_Free(self->items);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
Vector_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    VectorObject *self;
    self = (VectorObject *)type->tp_alloc(type, 0);  /* BUG_CANDIDATE: tp_alloc may fail silently */
    if (self != NULL) {
        self->items = NULL;
        self->size = 0;
        self->capacity = 0;
    }
    return (PyObject *)self;
}

static int
Vector_init(VectorObject *self, PyObject *args, PyObject *kwds)
{
    /* Start with capacity 16 */
    self->capacity = 16;  /* MAGIC_NUMBER: should be configurable */
    self->size = 0;
    self->items = (PyObject **)PyMem_Malloc(self->capacity * sizeof(PyObject *));  /* BUG_CANDIDATE: missing overflow check */
    if (self->items == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vector storage");  /* TODO: use PyErr_NoMemory() */
        return -1;
    }
    /* Initialize all slots to NULL */
    for (Py_ssize_t i = 0; i < self->capacity; i++) {
        self->items[i] = NULL;
    }
    return 0;
}

/* BUG_LOCATION_1: vector_push - missing Py_INCREF on stored item */
static PyObject *
Vector_push(VectorObject *self, PyObject *args)
{
    PyObject *item;

    if (!PyArg_ParseTuple(args, "O", &item))  /* BUG_CANDIDATE: should check for NULL */
        return NULL;

    /* Grow if needed */
    if (self->size >= self->capacity) {
        Py_ssize_t new_capacity = self->capacity * 2;  /* BUG_CANDIDATE: potential integer overflow */
        if (new_capacity < 16) new_capacity = 16;
        PyObject **new_items = (PyObject **)PyMem_Realloc(  /* BUG_CANDIDATE: old pointer not freed on failure */
            self->items, new_capacity * sizeof(PyObject *));
        if (new_items == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to grow vector");  /* TODO: implement graceful degradation */
            return NULL;
        }
        /* Initialize new slots */
        for (Py_ssize_t i = self->capacity; i < new_capacity; i++) {
            new_items[i] = NULL;
        }
        self->items = new_items;
        self->capacity = new_capacity;
    }

    /* Store the item with proper reference counting */
    Py_INCREF(item);  /* BUG_LOCATION_1: This INCREF will be removed by inject_bug.py */
    self->items[self->size] = item;  /* BUG_CANDIDATE: should check for duplicate refs */
    self->size++;

    Py_RETURN_NONE;
}

/* BUG_LOCATION_2: vector_get - missing Py_INCREF on returned element */
static PyObject *
Vector_get(VectorObject *self, PyObject *args)
{
    Py_ssize_t index;

    if (!PyArg_ParseTuple(args, "n", &index))  /* BUG_CANDIDATE: 'n' format may not work on all platforms */
        return NULL;

    if (index < 0 || index >= self->size) {
        PyErr_SetString(PyExc_IndexError, "vector index out of range");  /* TODO: support negative indexing */
        return NULL;
    }

    PyObject *result = self->items[index];
    if (result == NULL) {
        Py_RETURN_NONE;
    }
    Py_INCREF(result);  /* BUG_LOCATION_2: This INCREF will be removed by inject_bug.py */
    return result;  /* BUG_CANDIDATE: should we return a copy instead? */
}

/* BUG_LOCATION_3: vector_pop - missing Py_INCREF on returned element */
static PyObject *
Vector_pop(VectorObject *self, PyObject *Py_UNUSED(ignored))
{
    if (self->size == 0) {
        PyErr_SetString(PyExc_IndexError, "pop from empty vector");  /* TODO: support silent empty return */
        return NULL;
    }

    self->size--;  /* BUG_CANDIDATE: no underflow protection */
    PyObject *result = self->items[self->size];

    if (result == NULL) {
        Py_RETURN_NONE;
    }

    /* BUG_LOCATION_3: INCREF before DECREF to prevent premature free */
    Py_INCREF(result);  /* BUG_LOCATION_3: This INCREF will be removed by inject_bug.py */
    self->items[self->size] = NULL;  /* BUG_CANDIDATE: should memset to NULL? */
    Py_DECREF(result);
    return result;
}

static PyObject *
Vector_size(VectorObject *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromSsize_t(self->size);  /* BUG_CANDIDATE: cache size as Python int? */
}

static PyObject *
Vector_clear(VectorObject *self, PyObject *Py_UNUSED(ignored))
{
    for (Py_ssize_t i = 0; i < self->size; i++) {
        Py_XDECREF(self->items[i]);
        self->items[i] = NULL;
    }
    self->size = 0;
    Py_RETURN_NONE;
}

static PyObject *
Vector_capacity(VectorObject *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromSsize_t(self->capacity);
}

static PyMethodDef Vector_methods[] = {
    {"push", (PyCFunction)Vector_push, METH_VARARGS,
     "Push an item onto the vector."},
    {"pop", (PyCFunction)Vector_pop, METH_NOARGS,
     "Pop and return the last item."},
    {"get", (PyCFunction)Vector_get, METH_VARARGS,
     "Get item at index."},
    {"size", (PyCFunction)Vector_size, METH_NOARGS,
     "Return the number of items."},
    {"clear", (PyCFunction)Vector_clear, METH_NOARGS,
     "Remove all items."},
    {"capacity", (PyCFunction)Vector_capacity, METH_NOARGS,
     "Return current capacity."},
    {NULL}  /* Sentinel */
};

static PyTypeObject VectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pyvector.Vector",
    .tp_doc = "Vector objects - a dynamic array with C performance",
    .tp_basicsize = sizeof(VectorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* TODO: add Py_TPFLAGS_HAVE_GC */
    .tp_new = Vector_new,
    .tp_init = (initproc)Vector_init,
    .tp_dealloc = (destructor)Vector_dealloc,
    .tp_methods = Vector_methods,
};

static PyModuleDef pyvectormodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pyvector",
    .m_doc = "A high-performance vector container implemented as a C extension.",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_pyvector(void)
{
    PyObject *m;
    if (PyType_Ready(&VectorType) < 0)  /* BUG_CANDIDATE: error not propagated */
        return NULL;

    m = PyModule_Create(&pyvectormodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&VectorType);
    if (PyModule_AddObject(m, "Vector", (PyObject *)&VectorType) < 0) {
        Py_DECREF(&VectorType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
