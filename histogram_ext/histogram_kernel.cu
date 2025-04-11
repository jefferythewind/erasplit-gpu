#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA kernel: tiled, 64-bit safe
__global__ void histogram_tiled_kernel(
    const int8_t* __restrict__ bin_indices,   // [N, F]
    const float* __restrict__ gradients,      // [N]
    float* __restrict__ grad_hist,            // [F * B]
    float* __restrict__ hess_hist,            // [F * B]
    int64_t F, int64_t B, int64_t tile_size
) {
    int64_t feature_tiles = (F + tile_size - 1) / tile_size;
    int64_t row = static_cast<int64_t>(blockIdx.x) / feature_tiles;
    int64_t tile = static_cast<int64_t>(blockIdx.x) % feature_tiles;
    int64_t feat = tile * tile_size + threadIdx.x;

    if (feat >= F) return;

    int8_t bin = bin_indices[row * F + feat];
    if (bin >= 0 && bin < B) {
        int64_t idx = feat * B + bin;
        atomicAdd(&grad_hist[idx], gradients[row]);
        atomicAdd(&hess_hist[idx], 1.0f);
    }
}

// Host function exposed to PyTorch
void launch_histogram_kernel_cuda(
    const at::Tensor& bin_indices,   // int8 [N, F]
    const at::Tensor& gradients,     // float32 [N]
    at::Tensor& grad_hist,           // float32 [F * B]
    at::Tensor& hess_hist,           // float32 [F * B]
    int num_bins
) {
    CHECK_INPUT(bin_indices);
    CHECK_INPUT(gradients);
    CHECK_INPUT(grad_hist);
    CHECK_INPUT(hess_hist);

    int64_t N = bin_indices.size(0);
    int64_t F = bin_indices.size(1);
    int64_t tile_size = 256;
    int64_t feature_tiles = (F + tile_size - 1) / tile_size;
    int64_t total_blocks = N * feature_tiles;

    histogram_tiled_kernel<<<
        static_cast<int>(total_blocks),
        static_cast<int>(tile_size)
    >>>(
        bin_indices.data_ptr<int8_t>(),
        gradients.data_ptr<float>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        F, num_bins, tile_size
    );

    // Optional: check for kernel launch failure
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
