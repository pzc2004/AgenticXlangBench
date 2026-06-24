/*
 * conv_ops.c — Python C extension wrapper for custom CUDA conv2d kernels
 *
 * Provides:
 *   conv2d_forward(input, weight, bias, stride, padding) -> output
 *   conv2d_backward(grad_output, input, weight, stride, padding) -> (grad_input, grad_weight, grad_bias)
 *
 * All tensors are PyTorch CUDA float32 tensors.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>

/* Forward declaration from conv_kernel.cu */
extern int conv2d_forward_cuda(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride, int padding);

extern int conv2d_backward_cuda(
    const float* grad_output,
    const float* input, const float* weight,
    float* grad_input, float* grad_weight, float* grad_bias,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride, int padding);

/* We use the PyTorch C tensor API via torch/library includes */
/* Since we're a standalone extension, we access tensors through DLPack or raw data.
 * For simplicity, we use numpy + cuda array interface, or we directly use
 * PyTorch's THPC API.
 *
 * Actually, the simplest approach: use Python to pass raw pointers.
 * We'll call torch tensors' data_ptr() from Python and pass as integers.
 */

static PyObject* py_conv2d_forward(PyObject* self, PyObject* args) {
    /* Parse: input_ptr, weight_ptr, bias_ptr, output_ptr,
             N, C_in, H_in, W_in, C_out, kH, kW, stride, padding */
    long long input_ptr, weight_ptr, bias_ptr, output_ptr;
    int N, C_in, H_in, W_in, C_out, kH, kW, stride, padding;

    if (!PyArg_ParseTuple(args, "LLLLiiiiiiiii",
            &input_ptr, &weight_ptr, &bias_ptr, &output_ptr,
            &N, &C_in, &H_in, &W_in, &C_out, &kH, &kW, &stride, &padding)) {
        return NULL;
    }

    int ret = conv2d_forward_cuda(
        (const float*)input_ptr, (const float*)weight_ptr, (const float*)bias_ptr,
        (float*)output_ptr,
        N, C_in, H_in, W_in, C_out, kH, kW, stride, padding);

    if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError, "conv2d_forward_cuda failed");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* py_conv2d_backward(PyObject* self, PyObject* args) {
    /* Parse: grad_output_ptr, input_ptr, weight_ptr,
             grad_input_ptr, grad_weight_ptr, grad_bias_ptr,
             N, C_in, H_in, W_in, C_out, kH, kW, stride, padding */
    long long grad_output_ptr, input_ptr, weight_ptr;
    long long grad_input_ptr, grad_weight_ptr, grad_bias_ptr;
    int N, C_in, H_in, W_in, C_out, kH, kW, stride, padding;

    if (!PyArg_ParseTuple(args, "LLLLLLiiiiiiiii",
            &grad_output_ptr, &input_ptr, &weight_ptr,
            &grad_input_ptr, &grad_weight_ptr, &grad_bias_ptr,
            &N, &C_in, &H_in, &W_in, &C_out, &kH, &kW, &stride, &padding)) {
        return NULL;
    }

    int ret = conv2d_backward_cuda(
        (const float*)grad_output_ptr,
        (const float*)input_ptr, (const float*)weight_ptr,
        (float*)grad_input_ptr, (float*)grad_weight_ptr, (float*)grad_bias_ptr,
        N, C_in, H_in, W_in, C_out, kH, kW, stride, padding);

    if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError, "conv2d_backward_cuda failed");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyMethodDef ConvMethods[] = {
    {"conv2d_forward", py_conv2d_forward, METH_VARARGS,
     "conv2d_forward(input_ptr, weight_ptr, bias_ptr, output_ptr, N, C_in, H_in, W_in, C_out, kH, kW, stride, padding)"},
    {"conv2d_backward", py_conv2d_backward, METH_VARARGS,
     "conv2d_backward(grad_output_ptr, input_ptr, weight_ptr, grad_input_ptr, grad_weight_ptr, grad_bias_ptr, N, C_in, H_in, W_in, C_out, kH, kW, stride, padding)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef conv_module = {
    PyModuleDef_HEAD_INIT,
    "conv_ops",
    "Custom CUDA conv2d operations",
    -1,
    ConvMethods
};

PyMODINIT_FUNC PyInit_conv_ops(void) {
    return PyModule_Create(&conv_module);
}
