#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>  // for int64_t

// The actual CUDA kernel
__global__ void histogram_kernel(
    const int8_t* bin_indices,
    const float* gradients,
    float* grad_hist,
    float* hess_hist,
    int64_t N, int64_t F, int64_t B
) {
    int64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= N) return;

    for (int64_t f = 0; f < F; ++f) {
        int64_t idx = sample * F + f;
        int8_t b = bin_indices[idx];
        if (b >= 0 && b < B) {
            atomicAdd(&grad_hist[f * B + b], gradients[sample]);
            atomicAdd(&hess_hist[f * B + b], 1.0f);
        }
    }
}

// Exported C++ function that launches the kernel
void launch_histogram_kernel_cuda(
    const at::Tensor& bin_indices,
    const at::Tensor& gradients,
    at::Tensor& grad_hist,
    at::Tensor& hess_hist,
    int num_bins
) {
    int64_t N = bin_indices.size(0);
    int64_t F = bin_indices.size(1);
    int threads = 256;
    int blocks = static_cast<int>((N + threads - 1) / threads);  // still int

    histogram_kernel<<<blocks, threads>>>(
        bin_indices.data_ptr<int8_t>(),
        gradients.data_ptr<float>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        N, F, num_bins
    );
}
