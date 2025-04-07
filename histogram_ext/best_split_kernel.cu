#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void best_split_kernel(
    const float* __restrict__ G,  // [F x B]
    const float* __restrict__ H,  // [F x B]
    int F,
    int B,
    float min_split_gain,
    float min_child_samples,
    float eps,
    int* out_feature,
    int* out_bin,
    void* shared_mem
) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;

    // Cast shared memory
    extern __shared__ char smem[];
    float* gains = reinterpret_cast<float*>(smem);
    int* features = reinterpret_cast<int*>(&gains[blockDim.x]);
    int* bins = reinterpret_cast<int*>(&features[blockDim.x]);

    // Calculate total G and H for this feature
    float G_total = 0.0f, H_total = 0.0f;
    for (int b = 0; b < B; ++b) {
        G_total += G[f * B + b];
        H_total += H[f * B + b];
    }

    float G_L = 0.0f, H_L = 0.0f;
    float best_gain = min_split_gain;
    int best_bin = -1;

    for (int b = 0; b < B - 1; ++b) {
        G_L += G[f * B + b];
        H_L += H[f * B + b];
        float G_R = G_total - G_L;
        float H_R = H_total - H_L;

        if (H_L > min_child_samples && H_R > min_child_samples) {
            float gain = (G_L * G_L) / (H_L + eps) + (G_R * G_R) / (H_R + eps);
            if (gain > best_gain) {
                best_gain = gain;
                best_bin = b;
            }
        }
    }

    gains[threadIdx.x] = best_gain;
    features[threadIdx.x] = f;
    bins[threadIdx.x] = best_bin;
    __syncthreads();

    // Thread 0 in each block finds best among its block
    if (threadIdx.x == 0) {
        float block_best_gain = min_split_gain;
        int block_best_feature = -1;
        int block_best_bin = -1;
        for (int i = 0; i < blockDim.x && blockIdx.x * blockDim.x + i < F; ++i) {
            if (gains[i] > block_best_gain) {
                block_best_gain = gains[i];
                block_best_feature = features[i];
                block_best_bin = bins[i];
            }
        }

        // Write to global outputs
        *out_feature = block_best_feature;
        *out_bin = block_best_bin;
    }
}

extern "C" void launch_best_split_kernel_cuda(
    const at::Tensor& G,
    const at::Tensor& H,
    int F,
    int B,
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor& out_feature,
    at::Tensor& out_bin
) {
    int threads = 256;
    int blocks = (F + threads - 1) / threads;

    size_t shared_mem_bytes = threads * (sizeof(float) + 2 * sizeof(int));

    best_split_kernel<<<blocks, threads, shared_mem_bytes>>>(
        G.data_ptr<float>(),
        H.data_ptr<float>(),
        F,
        B,
        min_split_gain,
        min_child_samples,
        eps,
        out_feature.data_ptr<int>(),
        out_bin.data_ptr<int>(),
        nullptr  // shared memory pointer not needed; just launch size
    );
}