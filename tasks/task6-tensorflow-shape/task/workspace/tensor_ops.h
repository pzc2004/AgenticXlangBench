/*
 * tensor_ops.h -- Public header for the tensor_ops C extension.
 *
 * This header documents the shape-inference and forward-computation
 * contracts.  It is NOT included by the .c file (CPython extensions
 * are self-contained); it exists solely for human reference and for
 * any future C-level callers.
 */
#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * conv2d
 * ------
 * input  : float32 [N, C_in,  H,    W   ]
 * weight : float32 [C_out, C_in, KH, KW ]
 * output : float32 [N, C_out, H_out, W_out]
 *
 *   H_out = (H - KH + 2*padding) / stride + 1
 *   W_out = (W - KW + 2*padding) / stride + 1
 */

/*
 * relu
 * ----
 * input  : float32 [... any shape ...]
 * output : float32 [... same shape ...]
 *   out[i] = max(0, in[i])
 */

/*
 * pool  (average pooling, no padding)
 * ----
 * input  : float32 [N, C, H, W]
 * output : float32 [N, C, H_out, W_out]
 *
 *   H_out = (H - kernel_size) / stride + 1
 *   W_out = (W - kernel_size) / stride + 1
 */

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_OPS_H */
