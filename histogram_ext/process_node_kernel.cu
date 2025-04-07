#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

// The actual kernel
__global__ void process_node_kernel(
    const int8_t* bin_indices,
    const float* Y,
    float* gradients,
    float* tree_matrix,
    int64_t* node_indices,       // in-place buffer
    int num_node_samples_root,   // root N
    int F,
    int B,
    int T,                       // max nodes per tree
    int max_depth,
    float learning_rate,
    float min_split_gain,
    float min_child_weight,
    float eps,
    int32_t* node_jobs,          // shape: [T x 3] → (depth, start, end)
    int tree_offset,             // offset into tree_matrix
    float* grad_hist,            // shape: [F x B]
    float* hess_hist             // shape: [F x B]
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int job_index = 0;
    int next_job_index = 1;

    while (job_index < next_job_index) {
        int depth = node_jobs[job_index * 3 + 0];
        int start = node_jobs[job_index * 3 + 1];
        int end   = node_jobs[job_index * 3 + 2];
        int num_node_samples = end - start;

        int node_id = tree_offset + job_index;
        int64_t* node_indices_ptr = &node_indices[start];

        //printf("Node %d [%d:%d] samples: ", job_index, start, end);
        //for (int i = start; i < end; ++i)
            //printf("%lld ", node_indices[i]);
        //printf("\n");

        if (num_node_samples == 0) {
            job_index++;
            continue;
        }

        // Zero the shared histograms
        for (int i = 0; i < F * B; ++i) {
            grad_hist[i] = 0.0f;
            hess_hist[i] = 0.0f;
        }

        float G_total = 0.0f, H_total = 0.0f;

        for (int i = 0; i < num_node_samples; ++i) {
            int64_t idx = node_indices_ptr[i];
            float y = Y[idx];
            float g = gradients[idx];
            float residual = y - g;
            G_total += residual;
            H_total += 1.0f;

            for (int f = 0; f < F; ++f) {
                int8_t b = bin_indices[idx * F + f];
                if (b >= 0 && b < B) {
                    atomicAdd(&grad_hist[f * B + b], residual);
                    atomicAdd(&hess_hist[f * B + b], 1.0f);
                }
            }
        }

        if (depth == max_depth) {
            float leaf_val = G_total / (H_total + eps);
            tree_matrix[node_id * 7 + 0] = 1.0f;
            tree_matrix[node_id * 7 + 1] = leaf_val;
            tree_matrix[node_id * 7 + 6] = (float)num_node_samples;

            for (int i = 0; i < num_node_samples; ++i) {
                int64_t idx = node_indices_ptr[i];
                gradients[idx] += learning_rate * leaf_val;
            }

            job_index++;
            continue;
        }

        float best_gain = min_split_gain;
        int best_feature = -1, best_bin = -1;

        for (int f = 0; f < F; ++f) {
            float G_L = 0.0f, H_L = 0.0f;
            for (int b = 0; b < B - 1; ++b) {
                int offset = f * B + b;
                G_L += grad_hist[offset];
                H_L += hess_hist[offset];
                float G_R = G_total - G_L;
                float H_R = H_total - H_L;
                //printf("Checking split f=%d b=%d: H_L=%.2f H_R=%.2f\n", f, b, H_L, H_R);
                if (H_L >= min_child_weight && H_R >= min_child_weight) {
                    float gain = (G_L * G_L) / (H_L + eps) + (G_R * G_R) / (H_R + eps);
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_feature = f;
                        best_bin = b;
                    }
                }
            }
        }

        if (best_feature == -1) {
            float leaf_val = G_total / (H_total + eps);
            tree_matrix[node_id * 7 + 0] = 1.0f;
            tree_matrix[node_id * 7 + 1] = leaf_val;
            tree_matrix[node_id * 7 + 6] = (float)num_node_samples;

            for (int i = 0; i < num_node_samples; ++i) {
                int64_t idx = node_indices_ptr[i];
                gradients[idx] += learning_rate * leaf_val;
            }

            job_index++;
            continue;
        }

        tree_matrix[node_id * 7 + 0] = 0.0f;
        tree_matrix[node_id * 7 + 2] = (float)best_feature;
        tree_matrix[node_id * 7 + 3] = (float)best_bin;
        tree_matrix[node_id * 7 + 6] = (float)num_node_samples;

        int left_child_id  = tree_offset + next_job_index;
        int right_child_id = tree_offset + next_job_index + 1;

        tree_matrix[node_id * 7 + 4] = (float)left_child_id;
        tree_matrix[node_id * 7 + 5] = (float)right_child_id;

        int left_count = 0;
        int right_count = num_node_samples - 1;
        
        while (left_count <= right_count) {
            int64_t idx = node_indices_ptr[left_count];
            int8_t b = bin_indices[idx * F + best_feature];
            if (b <= best_bin) {
                left_count++;
            } else {
                // Swap with right
                int64_t temp = node_indices_ptr[right_count];
                node_indices_ptr[right_count] = idx;
                node_indices_ptr[left_count] = temp;
                right_count--;
            }
        }

        node_jobs[next_job_index * 3 + 0] = depth + 1;
        node_jobs[next_job_index * 3 + 1] = start;
        node_jobs[next_job_index * 3 + 2] = start + left_count;
        next_job_index++;

        node_jobs[next_job_index * 3 + 0] = depth + 1;
        node_jobs[next_job_index * 3 + 1] = start + left_count;
        node_jobs[next_job_index * 3 + 2] = end;
        next_job_index++;

        job_index++;
    }
}


void launch_process_node_kernel_cuda(
    const at::Tensor& bin_indices,
    const at::Tensor& Y,
    at::Tensor& gradients,
    at::Tensor& tree_matrix,
    at::Tensor& node_indices,
    int F,
    int B,
    int T,
    int max_depth,
    float learning_rate,
    float min_split_gain,
    float min_child_weight,
    float eps,
    at::Tensor& node_jobs,
    int tree_offset,
    at::Tensor& gradient_histogram,
    at::Tensor& hessian_histogram
) {
    int num_node_samples_root = node_indices.size(0);

    process_node_kernel<<<1, 1>>>(
        bin_indices.data_ptr<int8_t>(),
        Y.data_ptr<float>(),
        gradients.data_ptr<float>(),
        tree_matrix.data_ptr<float>(),
        node_indices.data_ptr<int64_t>(),
        num_node_samples_root,
        F,
        B,
        T,
        max_depth,
        learning_rate,
        min_split_gain,
        min_child_weight,
        eps,
        node_jobs.data_ptr<int32_t>(),
        tree_offset,
        gradient_histogram.data_ptr<float>(),
        hessian_histogram.data_ptr<float>()
    );
}

