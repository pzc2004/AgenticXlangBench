/*
 * Python C extension with GIL race condition.
 *
 * This module provides a shared counter and computation functions.
 * The GIL is released too early (before modifying shared state),
 * causing data races in multi-threaded usage.
 *
 * Contains 3 intentional bugs:
 *   Bug 1: Py_BEGIN_ALLOW_THREADS before modifying shared counter
 *   Bug 2: Missing GIL acquisition when reading shared state
 *   Bug 3: Py_END_ALLOW_THREADS in wrong position (before final update)
 *
 * Single-threaded: works perfectly
 * Multi-threaded: probabilistic incorrect results
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <math.h>

/* Shared module state */
static long global_counter = 0;
static double global_accumulator = 0.0;
static long global_call_count = 0;

/*
 * compute_sum: Sum an array of doubles.
 *
 * BUG 1: Releases GIL before updating global_counter
 * BUG 2: Reads global_accumulator without GIL
 */
static PyObject* compute_sum(PyObject* self, PyObject* args) {
    Py_buffer view;

    if (!PyArg_ParseTuple(args, "y*", &view)) {
        return NULL;
    }

    double* data = (double*)view.buf;
    Py_ssize_t n = view.len / sizeof(double);

    /*
     * BUG 1: GIL released too early
     * Correct: update shared state FIRST, then release GIL
     * Buggy:   release GIL, then update shared state (race condition!)
     *
     * The correct code would be:
     *   global_counter += n;
     *   global_call_count++;
     *   Py_BEGIN_ALLOW_THREADS
     *
     * The buggy code releases GIL first:
     */
    Py_BEGIN_ALLOW_THREADS  /* BUG: GIL released before updating shared state */

    /* These updates are now unprotected! */
    global_counter += n;     /* RACE: another thread may be updating too */
    global_call_count++;     /* RACE: increment is not atomic */

    /* Compute sum (this part is safe to run without GIL) */
    double sum = 0.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        sum += data[i];
    }

    /*
     * BUG 3: Shared state update after GIL re-acquisition is in wrong order
     * Correct: acquire GIL, then update accumulator
     * Buggy:   update accumulator while still in NO-GIL zone
     */
    global_accumulator += sum;  /* BUG: update without GIL */

    Py_END_ALLOW_THREADS

    PyBuffer_Release(&view);

    /* Read global_accumulator - may see partially updated value */
    /*
     * BUG 2: Reading shared state that was updated without GIL
     * The value of global_accumulator may be inconsistent
     */
    double result = sum + (global_accumulator * 0.0001);  /* Use shared state */

    return PyFloat_FromDouble(result);
}

/*
 * compute_product: Compute product of array elements.
 * Same GIL bugs as compute_sum.
 */
static PyObject* compute_product(PyObject* self, PyObject* args) {
    Py_buffer view;

    if (!PyArg_ParseTuple(args, "y*", &view)) {
        return NULL;
    }

    double* data = (double*)view.buf;
    Py_ssize_t n = view.len / sizeof(double);

    /* BUG 1: Same premature GIL release */
    Py_BEGIN_ALLOW_THREADS  /* BUG: released before shared state update */

    global_counter += n;     /* RACE */
    global_call_count++;     /* RACE */

    /* Compute product */
    double product = 1.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        product *= data[i];
    }

    /* BUG: update shared state without GIL */
    global_accumulator += log(fabs(product) + 1e-10);  /* BUG: no GIL */

    Py_END_ALLOW_THREADS

    PyBuffer_Release(&view);

    return PyFloat_FromDouble(product);
}

/*
 * get_stats: Return current shared state.
 * Should be called with GIL held (it is, since it returns Python objects).
 */
static PyObject* get_stats(PyObject* self, PyObject* args) {
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }

    return Py_BuildValue("{s:l, s:d, s:l}",
        "counter", global_counter,
        "accumulator", global_accumulator,
        "call_count", global_call_count);
}

/*
 * reset_stats: Reset shared state.
 */
static PyObject* reset_stats(PyObject* self, PyObject* args) {
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }

    global_counter = 0;
    global_accumulator = 0.0;
    global_call_count = 0;

    Py_RETURN_NONE;
}

/* Method table */
static PyMethodDef ComputeMethods[] = {
    {"compute_sum", compute_sum, METH_VARARGS,
     "Compute sum of array of doubles. Thread-safe (supposedly)."},
    {"compute_product", compute_product, METH_VARARGS,
     "Compute product of array of doubles. Thread-safe (supposedly)."},
    {"get_stats", get_stats, METH_VARARGS,
     "Get module statistics."},
    {"reset_stats", reset_stats, METH_VARARGS,
     "Reset module statistics."},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef computemodule = {
    PyModuleDef_HEAD_INIT,
    "compute",
    "Module with GIL race condition bug",
    -1,
    ComputeMethods
};

PyMODINIT_FUNC PyInit_compute(void) {
    return PyModule_Create(&computemodule);
}
