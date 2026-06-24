/*
 * conv_kernel.cu — Custom CUDA conv2d implementation (forward + backward)
 *
 * This file provides a standalone conv2d implementation used as a cuDNN
 * replacement. The forward pass and backward pass are implemented as
 * separate CUDA kernels.
 *
 * BUG_LOCATIONS (3 bugs will be injected here):
 *   Bug 1: backward_data boundary condition (input_h % stride != 0)
 *   Bug 2: backward_weight accumulation (even kernel_size)
 *   Bug 3: backward_bias reduction (non-aligned sizes)
 *
 * DO NOT MODIFY THIS FILE — the inject_bug script targets specific code
 * patterns. Changes will break the injection.
 */

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>

/* ========================================================================
 * Utility: CUDA error checking
 * ======================================================================== */

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
        }                                                                   \
    } while (0)

/* ========================================================================
 * Forward kernel: conv2d_forward_kernel
 *
 * Computes output[n][oc][oh][ow] =
 *   bias[oc] + sum_{ic,kh,kw} input[n][ic][oh*stride+kh-pad][ow*stride+kw-pad] * weight[oc][ic][kh][kw]
 * ======================================================================== */

__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride, int padding,
    int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int ow = idx % W_out;
    int oh = (idx / W_out) % H_out;
    int oc = (idx / (W_out * H_out)) % C_out;
    int n  = idx / (W_out * H_out * C_out);

    float val = bias[oc];

    for (int ic = 0; ic < C_in; ic++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                    int in_idx = ((n * C_in + ic) * H_in + ih) * W_in + iw;
                    int w_idx = ((oc * C_in + ic) * kH + kh) * kW + kw;
                    val += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    output[((n * C_out + oc) * H_out + oh) * W_out + ow] = val;
}

/* ========================================================================
 * Backward kernel 1: conv2d_backward_data_kernel
 *
 * Computes grad_input[n][ic][ih][iw] =
 *   sum_{oc,kh,kw} grad_output[n][oc][oh][ow] * weight[oc][ic][kh][kw]
 *   where ih = oh*stride + kh - padding, iw = ow*stride + kw - padding
 *
 * BUG_LOCATION_1: The output index computation when input_h % stride != 0.
 * Clean code uses:  oh_min = max(0, (ih - kH + 1 + padding + stride - 1) / stride)
 * Buggy code will use: oh_min = max(0, (ih - kH + 1 + padding + stride) / stride)
 * This adds an extra +1 in the numerator, causing off-by-one in boundary cases.
 * ======================================================================== */

__global__ void conv2d_backward_data_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ weight,
    float* __restrict__ grad_input,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride, int padding,
    int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * H_in * W_in;
    if (idx >= total) return;

    int iw = idx % W_in;
    int ih = (idx / W_in) % H_in;
    int ic = (idx / (W_in * H_in)) % C_in;
    int n  = idx / (W_in * H_in * C_in);

    float val = 0.0f;

    for (int oc = 0; oc < C_out; oc++) {
        for (int kh = 0; kh < kH; kh++) {
            /* BUG_LOCATION_1: backward_data boundary condition
             * Clean:   (ih - kh + padding + stride - 1) / stride
             * Buggy:   (ih - kh + padding + stride) / stride
             * Trigger: input_h % stride != 0
             */
            int oh_num = ih - kh + padding + stride - 1;
            int oh = oh_num / stride;
            if (oh < 0) oh = 0;
            if (oh >= H_out) continue;

            for (int kw = 0; kw < kW; kw++) {
                int ow_num = iw - kw + padding + stride - 1;
                int ow = ow_num / stride;
                if (ow < 0) ow = 0;
                if (ow >= W_out) continue;

                /* Verify the mapping is consistent: oh*stride+kh should map back to ih */
                int mapped_ih = oh * stride + kh - padding;
                int mapped_iw = ow * stride + kw - padding;
                if (mapped_ih != ih || mapped_iw != iw) continue;

                int go_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
                int w_idx = ((oc * C_in + ic) * kH + kh) * kW + kw;
                val += grad_output[go_idx] * weight[w_idx];
            }
        }
    }

    grad_input[idx] = val;
}

/* ========================================================================
 * Backward kernel 2: conv2d_backward_weight_kernel
 *
 * Computes grad_weight[oc][ic][kh][kw] =
 *   sum_{n,oh,ow} grad_output[n][oc][oh][ow] * input[n][ic][oh*stride+kh-pad][ow*stride+kw-pad]
 *
 * Uses atomicAdd since multiple (n, oh, ow) contribute to same weight.
 *
 * BUG_LOCATION_2: The accumulation factor for even kernel sizes.
 * Clean code accumulates normally.
 * Buggy code will multiply accumulation by 2 when kH is even.
 * Trigger: kernel_size % 2 == 0
 * ======================================================================== */

__global__ void conv2d_backward_weight_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_weight,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride, int padding,
    int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * kH * kW;
    if (idx >= total) return;

    int kw_i = idx % kW;
    int kh_i = (idx / kW) % kH;
    int ic   = (idx / (kW * kH)) % C_in;
    int oc   = idx / (kW * kH * C_in);

    float acc = 0.0f;

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < H_out; oh++) {
            int ih = oh * stride + kh_i - padding;
            if (ih < 0 || ih >= H_in) continue;

            for (int ow = 0; ow < W_out; ow++) {
                int iw = ow * stride + kw_i - padding;
                if (iw < 0 || iw >= W_in) continue;

                int go_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
                int in_idx = ((n * C_in + ic) * H_in + ih) * W_in + iw;
                acc += grad_output[go_idx] * input[in_idx];
            }
        }
    }

    /* BUG_LOCATION_2: backward_weight accumulation for even kernel sizes
     * Clean:   acc (no modification)
     * Buggy:   acc *= 2.0f when (kH % 2 == 0)
     * Trigger: kernel_size % 2 == 0
     */
    atomicAdd(&grad_weight[idx], acc);
}

/* ========================================================================
 * Backward kernel 3: conv2d_backward_bias_kernel
 *
 * Computes grad_bias[oc] = sum_{n,oh,ow} grad_output[n][oc][oh][ow]
 *
 * Uses block-level reduction + atomicAdd.
 *
 * BUG_LOCATION_3: The reduction for non-aligned sizes.
 * Clean code handles partial blocks correctly.
 * Buggy code will add an extra term when (H_out * W_out) % blockDim.x != 0.
 * Trigger: non-aligned output sizes (when input_h % stride != 0)
 * ======================================================================== */

__global__ void conv2d_backward_bias_kernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_bias,
    int N, int C_out, int H_out, int W_out)
{
    /* Each block handles one output channel */
    int oc = blockIdx.x;
    if (oc >= C_out) return;

    int HW = H_out * W_out;
    int total = N * HW;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;

    /* Each thread sums over its assigned (n, spatial) elements */
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int spatial_idx = i % HW;
        int n = i / HW;
        int go_idx = (n * C_out + oc) * HW + spatial_idx;
        local_sum += grad_output[go_idx];
    }

    /* BUG_LOCATION_3: backward_bias reduction for non-aligned sizes
     * Clean:   sdata[threadIdx.x] = local_sum (no extra term)
     * Buggy:   sdata[threadIdx.x] = local_sum + ((HW % blockDim.x != 0) ? 1.0f : 0.0f)
     * Trigger: non-aligned output spatial sizes
     */
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    /* Standard block-level reduction */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&grad_bias[oc], sdata[0]);
    }
}

/* ========================================================================
 * C-callable wrapper functions
 * ======================================================================== */

extern "C" {

int conv2d_forward_cuda(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride, int padding)
{
    int H_out = (H_in + 2 * padding - kH) / stride + 1;
    int W_out = (W_in + 2 * padding - kW) / stride + 1;

    if (H_out <= 0 || W_out <= 0) {
        fprintf(stderr, "conv2d_forward: invalid output dims H_out=%d W_out=%d\n", H_out, W_out);
        return -1;
    }

    int total = N * C_out * H_out * W_out;
    int block = 256;
    int grid = (total + block - 1) / block;

    conv2d_forward_kernel<<<grid, block>>>(
        input, weight, bias, output,
        N, C_in, H_in, W_in, C_out, kH, kW,
        stride, padding, H_out, W_out);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int conv2d_backward_cuda(
    const float* grad_output,
    const float* input, const float* weight,
    float* grad_input, float* grad_weight, float* grad_bias,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride, int padding)
{
    int H_out = (H_in + 2 * padding - kH) / stride + 1;
    int W_out = (W_in + 2 * padding - kW) / stride + 1;

    if (H_out <= 0 || W_out <= 0) {
        fprintf(stderr, "conv2d_backward: invalid output dims\n");
        return -1;
    }

    int block = 256;

    /* Backward data: one thread per input element */
    {
        int total = N * C_in * H_in * W_in;
        int grid = (total + block - 1) / block;
        conv2d_backward_data_kernel<<<grid, block>>>(
            grad_output, weight, grad_input,
            N, C_in, H_in, W_in, C_out, kH, kW,
            stride, padding, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());
    }

    /* Backward weight: one thread per weight element */
    {
        int total = C_out * C_in * kH * kW;
        int grid = (total + block - 1) / block;
        conv2d_backward_weight_kernel<<<grid, block>>>(
            grad_output, input, grad_weight,
            N, C_in, H_in, W_in, C_out, kH, kW,
            stride, padding, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());
    }

    /* Backward bias: one block per output channel */
    {
        int grid = C_out;
        int smem = block * sizeof(float);
        conv2d_backward_bias_kernel<<<grid, block, smem>>>(
            grad_output, grad_bias,
            N, C_out, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());
    }

    return 0;
}

} /* extern "C" */
