// LayerNorm forward CUDA kernel (with off-by-one bug)
// Bug: loop uses 'j <= N' instead of 'j < N', causing out-of-bounds access

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void layer_norm_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ output,
    const int N,        // feature dimension
    const float eps
) {
    const int i = blockIdx.x;  // batch index

    // Step 1: compute mean
    float mean = 0.0f;
    // BUG HERE: j <= N instead of j < N (off-by-one)
    for (int j = threadIdx.x; j <= N; j += blockDim.x) {
        mean += static_cast<float>(input[i * N + j]);
    }
    // Warp reduce
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        mean += __shfl_down_sync(0xffffffff, mean, offset);
    }
    if (threadIdx.x == 0) {
        // Store mean in shared memory (reuse output temporarily)
        output[i * N] = static_cast<scalar_t>(mean / N);
    }
    __syncthreads();
    mean = static_cast<float>(output[i * N]);

    // Step 2: compute variance
    float var = 0.0f;
    // BUG HERE: j <= N instead of j < N (off-by-one)
    for (int j = threadIdx.x; j <= N; j += blockDim.x) {
        float diff = static_cast<float>(input[i * N + j]) - mean;
        var += diff * diff;
    }
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        var += __shfl_down_sync(0xffffffff, var, offset);
    }
    if (threadIdx.x == 0) {
        output[i * N] = static_cast<scalar_t>(var / N);
    }
    __syncthreads();
    var = static_cast<float>(output[i * N]);
    float rstd = rsqrtf(var + eps);

    // Step 3: normalize
    // BUG HERE: j <= N instead of j < N (off-by-one)
    for (int j = threadIdx.x; j <= N; j += blockDim.x) {
        float x = static_cast<float>(input[i * N + j]);
        float g = (gamma != nullptr) ? static_cast<float>(gamma[j]) : 1.0f;
        float b = (beta != nullptr) ? static_cast<float>(beta[j]) : 0.0f;
        output[i * N + j] = static_cast<scalar_t>((x - mean) * rstd * g + b);
    }
}

torch::Tensor layer_norm_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    const int batch_size = input.size(0);
    const int N = input.size(1);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "layer_norm_forward", [&] {
        layer_norm_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N,
            eps
        );
    });

    return output;
}
