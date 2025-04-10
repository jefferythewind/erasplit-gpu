#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void predict_forest_cuda_kernel(
    const int8_t* __restrict__ bin_indices,    // [N, F]
    const int16_t* __restrict__ features,      // [total_nodes]
    const int16_t* __restrict__ thresholds,    // [total_nodes]
    const int32_t* __restrict__ lefts,         // [total_nodes]
    const int32_t* __restrict__ rights,        // [total_nodes]
    const float* __restrict__ leaf_values,     // [total_nodes]
    const int32_t* __restrict__ tree_offsets,  // [n_trees]
    float* __restrict__ out_preds,             // [N, n_trees]
    int64_t N,
    int64_t F,
    int64_t n_trees
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tree = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || tree >= n_trees) return;

    int node = tree_offsets[tree];

    while (features[node] != -1) {
        int f = features[node];
        int8_t b = thresholds[node];
        int8_t val = bin_indices[row * F + f];
        node = (val <= b) ? lefts[node] : rights[node];
    }

    out_preds[row * n_trees + tree] = leaf_values[node];
}

void launch_predict_forest_cuda(
    at::Tensor bin_indices,       // [N, F] int8
    at::Tensor features,          // [total_nodes] int16
    at::Tensor thresholds,        // [total_nodes] int16
    at::Tensor lefts,             // [total_nodes] int32
    at::Tensor rights,            // [total_nodes] int32
    at::Tensor leaf_values,       // [total_nodes] float32
    at::Tensor tree_offsets,      // [n_trees] int32
    at::Tensor out_preds          // [N, n_trees] float32
) {
    const int64_t N = bin_indices.size(0);
    const int64_t F = bin_indices.size(1);
    const int64_t n_trees = tree_offsets.size(0);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (n_trees + threads.y - 1) / threads.y);

    predict_forest_cuda_kernel<<<blocks, threads>>>(
        bin_indices.data_ptr<int8_t>(),
        features.data_ptr<int16_t>(),
        thresholds.data_ptr<int16_t>(),
        lefts.data_ptr<int32_t>(),
        rights.data_ptr<int32_t>(),
        leaf_values.data_ptr<float>(),
        tree_offsets.data_ptr<int32_t>(),
        out_preds.data_ptr<float>(),
        N, F, n_trees
    );
}
