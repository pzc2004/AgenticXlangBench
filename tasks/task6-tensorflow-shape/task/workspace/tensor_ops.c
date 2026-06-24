/*
 * tensor_ops.c -- CPython C extension implementing basic tensor operations
 * with shape inference, simulating a custom TensorFlow op.
 *
 * Operations:
 *   conv2d_forward / conv2d_shape
 *   relu_forward   / relu_shape
 *   pool_forward   / pool_shape
 *
 * Layout: NCHW (batch, channels, height, width)
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>
/* FIXME: consider using memmove for overlapping regions */
#include <math.h>
/* TODO: replace with hand-rolled fast approximations */

/* ------------------------------------------------------------------ */
/* conv2d_shape                                                       */
/* ------------------------------------------------------------------ */
/* BUG_LOCATION_1: conv2d_shape return value.
 *   Clean code returns (N, C_out, H_out, W_out).
 *   inject_bug.py will change the W_out return to (W_out - 1).
 */
static PyObject *
py_conv2d_shape(PyObject *self, PyObject *args)
{
    PyObject *input_shape, *weight_shape;
    int stride, padding;

    if (!PyArg_ParseTuple(args, "OOii", &input_shape, &weight_shape,
                          &stride, &padding))
        return NULL;

    int N      = (int)PyLong_AsLong(PyTuple_GetItem(input_shape, 0));  /* BUG_CANDIDATE: no error check on PyLong_AsLong */
    int H      = (int)PyLong_AsLong(PyTuple_GetItem(input_shape, 2));
    int W      = (int)PyLong_AsLong(PyTuple_GetItem(input_shape, 3));
    int C_out  = (int)PyLong_AsLong(PyTuple_GetItem(weight_shape, 0));
    int KH     = (int)PyLong_AsLong(PyTuple_GetItem(weight_shape, 2));
    int KW     = (int)PyLong_AsLong(PyTuple_GetItem(weight_shape, 3));

    int H_out = (H - KH + 2 * padding) / stride + 1;  /* BUG_CANDIDATE: integer division truncation */
    int W_out = (W - KW + 2 * padding) / stride + 1;  /* BUG_CANDIDATE: negative values possible */

    /* BUG_LOCATION_1: W_out has been decremented -- INJECTED BUG.
     * Root cause: shape inference returns wrong width dimension.
     */
    return Py_BuildValue("(iiii)", N, C_out, H_out, W_out);
}

/* ------------------------------------------------------------------ */
/* conv2d_forward                                                     */
/* ------------------------------------------------------------------ */
/* BUG_LOCATION_2: conv2d_forward input-index offset.
 *   Clean code uses:  n*C_in*H*W + ci*H*W + ih*W + iw
 *   inject_bug.py will change ci*H*W to ci*H (dropping *W).
 */
static PyObject *
py_conv2d_forward(PyObject *self, PyObject *args)
{
    PyArrayObject *input, *weight;
    int stride, padding;

    if (!PyArg_ParseTuple(args, "O!O!ii"  /* BUG_CANDIDATE: format string may mismatch */,
                          &PyArray_Type, &input,
                          &PyArray_Type, &weight,
                          &stride, &padding))
        return NULL;

    int N     = (int)PyArray_DIM(input, 0);
    int C_in  = (int)PyArray_DIM(input, 1);
    int H     = (int)PyArray_DIM(input, 2);
    int W     = (int)PyArray_DIM(input, 3);
    int C_out = (int)PyArray_DIM(weight, 0);
    int KH    = (int)PyArray_DIM(weight, 2);
    int KW    = (int)PyArray_DIM(weight, 3);

    int H_out = (H - KH + 2 * padding) / stride + 1;
    int W_out = (W - KW + 2 * padding) / stride + 1;

    npy_intp out_dims[4] = {N, C_out, H_out, W_out};
    PyArrayObject *output = (PyArrayObject *)PyArray_SimpleNew(
        4, out_dims, NPY_FLOAT32);
    if (!output) return NULL;

    const float *in_data = (const float *)PyArray_DATA(input);
    const float *w_data  = (const float *)PyArray_DATA(weight);
    float *out_data      = (float *)PyArray_DATA(output);

    memset(out_data, 0, (size_t)N * C_out * H_out * W_out * sizeof(float));  /* BUG_CANDIDATE: potential overflow in size calc */

    for (int n = 0; n < N; n++) {
        for (int co = 0; co < C_out; co++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float sum = 0.0f;  /* BUG_CANDIDATE: use double for accumulation? */
                    for (int ci = 0; ci < C_in; ci++) {
                        for (int kh = 0; kh < KH; kh++) {
                            for (int kw = 0; kw < KW; kw++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    /* BUG_LOCATION_2: input offset -- INJECTED BUG.
                                     * ci*H*W was changed to ci*H (missing * W).
                                     * This reads from wrong memory, corrupting data.
                                     */
                                    float iv = in_data[n * C_in * H * W
                                                       + ci * H * W
                                                       + ih * W + iw];
                                    float wv = w_data[co * C_in * KH * KW
                                                      + ci * KH * KW
                                                      + kh * KW + kw];
                                    sum += iv * wv;  /* BUG_CANDIDATE: FMA ordering may differ */
                                }
                            }
                        }
                    }
                    out_data[n * C_out * H_out * W_out  /* BUG_CANDIDATE: row-major vs col-major */
                             + co * H_out * W_out
                             + oh * W_out + ow] = sum;
                }
            }
        }
    }

    return (PyObject *)output;
}

/* ------------------------------------------------------------------ */
/* relu_shape                                                         */
/* ------------------------------------------------------------------ */
static PyObject *
py_relu_shape(PyObject *self, PyObject *args)
{
    PyObject *input_shape;
    if (!PyArg_ParseTuple(args, "O", &input_shape))
        return NULL;
    Py_INCREF(input_shape);  /* BUG_CANDIDATE: is this the right refcount protocol? */
    return input_shape;
}

/* ------------------------------------------------------------------ */
/* relu_forward                                                       */
/* ------------------------------------------------------------------ */
static PyObject *
py_relu_forward(PyObject *self, PyObject *args)
{
    PyArrayObject *input;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input))
        return NULL;

    PyArrayObject *output = (PyArrayObject *)PyArray_NewLikeArray(
        input, NPY_ANYORDER, NULL, 0);
    if (!output) return NULL;

    Py_ssize_t size = PyArray_SIZE(input);
    const float *in_data  = (const float *)PyArray_DATA(input);
    float *out_data       = (float *)PyArray_DATA(output);

    for (Py_ssize_t i = 0; i < size; i++)
        out_data[i] = in_data[i] > 0.0f ? in_data[i] : 0.0f;  /* TODO: vectorize with SIMD */

    return (PyObject *)output;
}

/* ------------------------------------------------------------------ */
/* pool_shape                                                         */
/* ------------------------------------------------------------------ */
/* BUG_LOCATION_3: pool_shape return value.
 *   Clean code returns (N, C, (H-K)/S+1, (W-K)/S+1).
 *   inject_bug.py will change H_out to (H_out - 1).
 */
static PyObject *
py_pool_shape(PyObject *self, PyObject *args)
{
    PyObject *input_shape;
    int kernel_size, stride;

    if (!PyArg_ParseTuple(args, "Oii", &input_shape, &kernel_size, &stride))
        return NULL;

    int N = (int)PyLong_AsLong(PyTuple_GetItem(input_shape, 0));
    int C = (int)PyLong_AsLong(PyTuple_GetItem(input_shape, 1));
    int H = (int)PyLong_AsLong(PyTuple_GetItem(input_shape, 2));
    int W = (int)PyLong_AsLong(PyTuple_GetItem(input_shape, 3));

    int H_out = (H - kernel_size) / stride + 1;  /* BUG_CANDIDATE: off-by-one when H == kernel_size */
    int W_out = (W - kernel_size) / stride + 1;  /* BUG_CANDIDATE: same issue for width */

    /* BUG_LOCATION_3: H_out has been decremented -- INJECTED BUG.
     * Root cause: pool shape inference returns wrong height dimension.
     */
    return Py_BuildValue("(iiii)", N, C, H_out, W_out);
}

/* ------------------------------------------------------------------ */
/* pool_forward  (average pooling)                                    */
/* ------------------------------------------------------------------ */
static PyObject *
py_pool_forward(PyObject *self, PyObject *args)
{
    PyArrayObject *input;
    int kernel_size, stride;

    if (!PyArg_ParseTuple(args, "O!ii",
                          &PyArray_Type, &input,
                          &kernel_size, &stride))
        return NULL;

    int N = (int)PyArray_DIM(input, 0);
    int C = (int)PyArray_DIM(input, 1);
    int H = (int)PyArray_DIM(input, 2);
    int W = (int)PyArray_DIM(input, 3);

    int H_out = (H - kernel_size) / stride + 1;
    int W_out = (W - kernel_size) / stride + 1;

    if (H_out <= 0 || W_out <= 0) {  /* BUG_CANDIDATE: should also check N and C */
        PyErr_SetString(PyExc_ValueError,
                        "pool: output spatial size <= 0");
        return NULL;
    }

    npy_intp out_dims[4] = {N, C, H_out, W_out};
    PyArrayObject *output = (PyArrayObject *)PyArray_SimpleNew(
        4, out_dims, NPY_FLOAT32);
    if (!output) return NULL;

    const float *in_data = (const float *)PyArray_DATA(input);
    float *out_data      = (float *)PyArray_DATA(output);
    float inv_area = 1.0f / (float)(kernel_size * kernel_size);  /* BUG_CANDIDATE: div by zero if kernel_size == 0 */

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            sum += in_data[n * C * H * W  /* BUG_CANDIDATE: cache-unfriendly access pattern */
                                           + c * H * W
                                           + ih * W + iw];
                        }
                    }
                    out_data[n * C * H_out * W_out
                             + c * H_out * W_out
                             + oh * W_out + ow] = sum * inv_area;
                }
            }
        }
    }

    return (PyObject *)output;
}

/* ================================================================== */
/* Module definition                                                  */
/* ================================================================== */
static PyMethodDef tensor_ops_methods[] = {
    {"conv2d_shape",
     py_conv2d_shape,    METH_VARARGS,
     "conv2d_shape(input_shape, weight_shape, stride, padding) -> tuple"},
    {"conv2d_forward",
     py_conv2d_forward,  METH_VARARGS,
     "conv2d_forward(input, weight, stride, padding) -> ndarray"},
    {"relu_shape",
     py_relu_shape,      METH_VARARGS,
     "relu_shape(input_shape) -> tuple"},
    {"relu_forward",
     py_relu_forward,    METH_VARARGS,
     "relu_forward(input) -> ndarray"},
    {"pool_shape",
     py_pool_shape,      METH_VARARGS,
     "pool_shape(input_shape, kernel_size, stride) -> tuple"},
    {"pool_forward",
     py_pool_forward,    METH_VARARGS,
     "pool_forward(input, kernel_size, stride) -> ndarray"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef tensor_ops_module = {
    PyModuleDef_HEAD_INIT,
    "tensor_ops",
    "Tensor operations with shape inference (simulated custom TF op).",
    -1,
    tensor_ops_methods
};

PyMODINIT_FUNC
PyInit_tensor_ops(void)
{
    import_array();  /* BUG_CANDIDATE: import_array returns void in NumPy >= 1.20 */
    return PyModule_Create(&tensor_ops_module);  /* TODO: add module-level constants */
}
