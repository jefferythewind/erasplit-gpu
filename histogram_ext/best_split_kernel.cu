#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void best_split_kernel_serial(
    const float* __restrict__ G,  // [F x B]
    const float* __restrict__ H,  // [F x B]
    int F,
    int B,
    float min_split_gain,
    float min_child_samples,
    float eps,
    int* out_feature,
    int* out_bin
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Compute G_total and H_total once, using feature 0
    float G_total = 0.0f, H_total = 0.0f;
    for (int b = 0; b < B; ++b) {
        G_total += G[0 * B + b];
        H_total += H[0 * B + b];
    }

    float best_gain = min_split_gain;
    int best_feature = -1;
    int best_bin = -1;

    float G_L = 0.0f;
    float H_L = 0.0f;
    float G_R = 0.0f;
    float H_R = 0.0f;
    float gain = 0.0f;
    int offset = 0;

    for (int f = 0; f < F; ++f) {
        G_L = 0.0f;
        H_L = 0.0f;
        offset = f * B;

        for (int b = 0; b < B - 1; ++b) {
            G_L += G[offset + b];
            H_L += H[offset + b];
            G_R = G_total - G_L;
            H_R = H_total - H_L;

            if (H_L >= min_child_samples && H_R >= min_child_samples) {
                gain = (G_L * G_L) / (H_L + eps) + (G_R * G_R) / (H_R + eps);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_feature = f;
                    best_bin = b;
                }
            }
        }
    }

    *out_feature = best_feature;
    *out_bin = best_bin;
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
    best_split_kernel_serial<<<1, 1>>>(
        G.data_ptr<float>(),
        H.data_ptr<float>(),
        F,
        B,
        min_split_gain,
        min_child_samples,
        eps,
        out_feature.data_ptr<int>(),
        out_bin.data_ptr<int>()
    );
}
