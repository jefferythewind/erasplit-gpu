#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <algorithm>  // for std::min

// Optimized kernel that uses shared memory to accumulate histograms per block.
// The kernel processes a batch of features (feature_count) for a given chunk of samples.
__global__ void histogram_kernel_batched_shared(
    const int32_t* __restrict__ bin_indices,
    const float* __restrict__ gradients,
    float* __restrict__ grad_hist,
    float* __restrict__ hess_hist,
    int N, int F, int B,
    int feature_start, int feature_count
) {
    extern __shared__ float shared_mem[];
    // Allocate shared memory:
    // First part for grad histogram, second part for hess histogram.
    float* local_grad = shared_mem;
    float* local_hess = shared_mem + feature_count * B;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int total_bins = feature_count * B;

    // Zero shared memory histograms
    for (int i = tid; i < total_bins; i += num_threads) {
        local_grad[i] = 0.0f;
        local_hess[i] = 0.0f;
    }
    __syncthreads();

    // Process samples in a grid-stride loop
    for (int sample = blockIdx.x * blockDim.x + tid; sample < N; sample += gridDim.x * blockDim.x) {
        // For each feature in the current feature batch
        for (int f_offset = 0; f_offset < feature_count; ++f_offset) {
            int f = feature_start + f_offset;
            if (f < F) {
                int bin_idx = bin_indices[sample * F + f];
                if (bin_idx >= 0 && bin_idx < B) {
                    // Accumulate in shared memory
                    atomicAdd(&local_grad[f_offset * B + bin_idx], gradients[sample]);
                    atomicAdd(&local_hess[f_offset * B + bin_idx], 1.0f);
                }
            }
        }
    }
    __syncthreads();

    // Merge the shared memory histograms into global memory
    for (int i = tid; i < total_bins; i += num_threads) {
        int feature_offset = i / B;
        int bin_idx = i % B;
        // Global histogram index for feature feature_start + feature_offset.
        atomicAdd(&grad_hist[(feature_start + feature_offset) * B + bin_idx], local_grad[i]);
        atomicAdd(&hess_hist[(feature_start + feature_offset) * B + bin_idx], local_hess[i]);
    }
}

// ---------------------- Dispatch Functions ---------------------------

// Chunking version using the shared memory kernel.
void launch_histogram_kernel_with_chunking(
    const at::Tensor& bin_indices_full,
    const at::Tensor& gradients,
    at::Tensor& grad_hist,
    at::Tensor& hess_hist,
    int num_bins
) {
    int N = bin_indices_full.size(0);
    int F = bin_indices_full.size(1);
    int B = num_bins;
    
    // Tuning parameters (adjust based on your GPU and problem size)
    const int SAMPLE_CHUNK_SIZE = 100000;   // Samples per chunk
    const int FEATURE_BATCH_SIZE = 1024;    // Features processed per kernel launch (shared mem limited)
    
    // Create an asynchronous CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int sample_start = 0; sample_start < N; sample_start += SAMPLE_CHUNK_SIZE) {
        int sample_count = std::min(SAMPLE_CHUNK_SIZE, N - sample_start);
        
        // Use slicing (zero-copy) to get the chunk views
        at::Tensor bin_indices_chunk = bin_indices_full.slice(0, sample_start, sample_start + sample_count);
        at::Tensor gradients_chunk = gradients.slice(0, sample_start, sample_start + sample_count);
        
        // Configure kernel launch parameters for this sample chunk
        int threads = 256;
        int blocks = (sample_count + threads - 1) / threads;
        blocks = std::min(blocks, 65535);
        
        // Process the feature dimension in batches
        for (int feature_start = 0; feature_start < F; feature_start += FEATURE_BATCH_SIZE) {
            int feature_count = std::min(FEATURE_BATCH_SIZE, F - feature_start);
            // Calculate shared memory size: two arrays of size feature_count * B
            size_t sharedMemSize = 2 * feature_count * B * sizeof(float);
            
            histogram_kernel_batched_shared<<<blocks, threads, sharedMemSize, stream>>>(
                bin_indices_chunk.data_ptr<int32_t>(),
                gradients_chunk.data_ptr<float>(),
                grad_hist.data_ptr<float>(),
                hess_hist.data_ptr<float>(),
                sample_count, F, B,
                feature_start, feature_count
            );
        }
        // Synchronize once per sample chunk
        cudaStreamSynchronize(stream);
    }
    cudaStreamDestroy(stream);
}

// Extreme memory-saving version: one feature at a time using shared memory kernel.
void launch_histogram_kernel_cuda_extreme_memory_saving(
    const at::Tensor& bin_indices,
    const at::Tensor& gradients,
    at::Tensor& grad_hist,
    at::Tensor& hess_hist,
    int num_bins
) {
    int N = bin_indices.size(0);
    int F = bin_indices.size(1);
    int B = num_bins;
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    blocks = std::min(blocks, 65535);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Process one feature at a time.
    for (int feature = 0; feature < F; feature++) {
        int feature_count = 1;
        size_t sharedMemSize = 2 * feature_count * B * sizeof(float);
        histogram_kernel_batched_shared<<<blocks, threads, sharedMemSize, stream>>>(
            bin_indices.data_ptr<int32_t>(),
            gradients.data_ptr<float>(),
            grad_hist.data_ptr<float>(),
            hess_hist.data_ptr<float>(),
            N, F, B,
            feature, feature_count
        );
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

// Batched version (without chunking) for when memory is plentiful.
void launch_histogram_kernel_cuda_memory_efficient(
    const at::Tensor& bin_indices,
    const at::Tensor& gradients,
    at::Tensor& grad_hist,
    at::Tensor& hess_hist,
    int num_bins
) {
    int N = bin_indices.size(0);
    int F = bin_indices.size(1);
    int B = num_bins;
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    blocks = std::min(blocks, 65535);
    
    const int FEATURE_BATCH_SIZE = 128;
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int feature_start = 0; feature_start < F; feature_start += FEATURE_BATCH_SIZE) {
        int feature_count = std::min(FEATURE_BATCH_SIZE, F - feature_start);
        size_t sharedMemSize = 2 * feature_count * B * sizeof(float);
        histogram_kernel_batched_shared<<<blocks, threads, sharedMemSize, stream>>>(
            bin_indices.data_ptr<int32_t>(),
            gradients.data_ptr<float>(),
            grad_hist.data_ptr<float>(),
            hess_hist.data_ptr<float>(),
            N, F, B,
            feature_start, feature_count
        );
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

// Adaptive helper that selects the implementation based on available memory.
void launch_histogram_kernel_auto_memory_efficient(
    const at::Tensor& bin_indices,
    const at::Tensor& gradients,
    at::Tensor& grad_hist,
    at::Tensor& hess_hist,
    int num_bins
) {
    int device;
    cudaGetDevice(&device);
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    
    int N = bin_indices.size(0);
    int F = bin_indices.size(1);
    // Estimate memory requirement for bin_indices and gradients.
    size_t required_memory = (size_t)N * F * sizeof(int32_t) + (size_t)N * sizeof(float);
    
    // Zero out global histograms.
    grad_hist.zero_();
    hess_hist.zero_();
    
    if (free_memory > 2 * required_memory) {
        launch_histogram_kernel_cuda_memory_efficient(bin_indices, gradients, grad_hist, hess_hist, num_bins);
    } else if (free_memory > required_memory) {
        launch_histogram_kernel_cuda_extreme_memory_saving(bin_indices, gradients, grad_hist, hess_hist, num_bins);
    } else {
        launch_histogram_kernel_with_chunking(bin_indices, gradients, grad_hist, hess_hist, num_bins);
    }
}

// Public function exposed to Python.
void launch_histogram_kernel_cuda(
    const at::Tensor& bin_indices,
    const at::Tensor& gradients,
    at::Tensor& grad_hist,
    at::Tensor& hess_hist,
    int num_bins
) {
    launch_histogram_kernel_auto_memory_efficient(
        bin_indices,
        gradients,
        grad_hist,
        hess_hist,
        num_bins
    );
}
