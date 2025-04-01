#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

struct BestSplit {
    float gain;
    int feature;
    int bin;
};

__global__ void best_split_kernel(
    const float* __restrict__ G,  // shape: [F x B]
    const float* __restrict__ H,  // shape: [F x B]
    int F,
    int B,
    float min_split_gain,
    float min_child_samples,
    float eps,
    float* block_gains,   // output: one per block
    int* block_features,  // output: one per block
    int* block_bins       // output: one per block
) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    bool valid = (f < F);
    
    float best_gain = min_split_gain;
    int best_bin = -1;

    if (valid) {
        // Sanity check: ensure that our memory accesses for G and H are in-range.
        // (Caller must guarantee that G and H have at least F*B elements.)
        float G_total = 0.0f, H_total = 0.0f;
        for (int b = 0; b < B; ++b) {
            // Using f * B + b; check if this index is within range if needed.
            G_total += G[f * B + b];
            H_total += H[f * B + b];
        }
        float G_L = 0.0f, H_L = 0.0f;
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
    }
    
    // Allocate shared memory using a char buffer, then cast it to BestSplit*.
    extern __shared__ char s[];
    BestSplit* shared_results = reinterpret_cast<BestSplit*>(s);
    
    BestSplit my_result;
    my_result.gain = best_gain;
    my_result.feature = valid ? f : -1;
    my_result.bin = best_bin;
    shared_results[threadIdx.x] = my_result;
    __syncthreads();
    
    // Thread 0 in the block performs the reduction over the block.
    if (threadIdx.x == 0) {
        BestSplit block_best;
        block_best.gain = min_split_gain;
        block_best.feature = -1;
        block_best.bin = -1;
        for (int i = 0; i < blockDim.x; i++) {
            if (shared_results[i].gain > block_best.gain) {
                block_best = shared_results[i];
            }
        }
        block_gains[blockIdx.x] = block_best.gain;
        block_features[blockIdx.x] = block_best.feature;
        block_bins[blockIdx.x] = block_best.bin;
    }
}

__global__ void reduce_block_results_kernel(
    const float* block_gains,
    const int* block_features,
    const int* block_bins,
    int num_blocks,
    float min_split_gain,
    int* out_feature,
    int* out_bin
) {
    // Single-thread reduction over per-block results.
    if (threadIdx.x == 0) {
        float best_gain = min_split_gain;
        int best_feature = -1;
        int best_bin = -1;
        for (int i = 0; i < num_blocks; i++) {
            float gain = block_gains[i];
            if (gain > best_gain) {
                best_gain = gain;
                best_feature = block_features[i];
                best_bin = block_bins[i];
            }
        }
        out_feature[0] = best_feature;
        out_bin[0] = best_bin;
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
    TORCH_CHECK(G.is_contiguous(), "G must be contiguous");
    TORCH_CHECK(H.is_contiguous(), "H must be contiguous");
    TORCH_CHECK(out_feature.is_contiguous(), "out_feature must be contiguous");
    TORCH_CHECK(out_bin.is_contiguous(), "out_bin must be contiguous");
    TORCH_CHECK(out_feature.numel() >= 1, "out_feature must have at least one element");
    TORCH_CHECK(out_bin.numel() >= 1, "out_bin must have at least one element");

    // Use a block size of 256 threads.
    int threads = 64;
    int blocks = (F + threads - 1) / threads;
    threads = std::min(threads, 1024);
    blocks = std::max(blocks, 1);

    auto options_float = G.options();
    at::Tensor block_gains = at::empty({blocks}, options_float);
    auto options_int = at::TensorOptions().dtype(at::kInt).device(G.device());
    at::Tensor block_features = at::empty({blocks}, options_int);
    at::Tensor block_bins = at::empty({blocks}, options_int);

    // Allocate shared memory: one BestSplit per thread.
    size_t shared_mem_size = threads * sizeof(BestSplit);

    best_split_kernel<<<blocks, threads, shared_mem_size>>>(
        G.data_ptr<float>(),
        H.data_ptr<float>(),
        F,
        B,
        min_split_gain,
        min_child_samples,
        eps,
        block_gains.data_ptr<float>(),
        block_features.data_ptr<int>(),
        block_bins.data_ptr<int>()
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string error_msg = "CUDA error in first kernel: " + std::string(cudaGetErrorString(error));
        throw std::runtime_error(error_msg);
    }

    reduce_block_results_kernel<<<1, 32>>>(
        block_gains.data_ptr<float>(),
        block_features.data_ptr<int>(),
        block_bins.data_ptr<int>(),
        blocks,
        min_split_gain,
        out_feature.data_ptr<int>(),
        out_bin.data_ptr<int>()
    );

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string error_msg = "CUDA error in second kernel: " + std::string(cudaGetErrorString(error));
        throw std::runtime_error(error_msg);
    }

    cudaDeviceSynchronize();
}
