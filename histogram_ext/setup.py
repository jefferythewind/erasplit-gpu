from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='node_kernel',
    ext_modules=[
        CUDAExtension(
            name='node_kernel',
            sources=[
                # 'process_node_kernel.cu',
                'histogram_kernel.cu',
                'best_split_kernel.cu',
                'predict_forest_kernel.cu',
                # 'split_gain_kernel.cu',
                'node_kernel.cpp',
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
