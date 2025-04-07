#include <torch/extension.h>
#include <vector>

// Declare launcher
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
);

// Declare the function from histogram_kernel.cu
void launch_histogram_kernel_cuda(
    const at::Tensor& bin_indices,
    const at::Tensor& gradients,
    at::Tensor& grad_hist,
    at::Tensor& hess_hist,
    int num_bins
);

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
);

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_histogram", &launch_histogram_kernel_cuda, "Histogram (CUDA)");
    m.def("compute_split", &launch_best_split_kernel_cuda, "Best Split (CUDA)");
    m.def("node_kernel", &launch_process_node_kernel_cuda, "GPU Tree Growing Launcher");
}