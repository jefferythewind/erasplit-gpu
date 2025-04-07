from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='node_kernel',
    ext_modules=[
        CUDAExtension(
            name='node_kernel',
            sources=[
                'node_kernel.cpp',
                'process_node_kernel.cu',
                'histogram_kernel.cu',
                'best_split_kernel.cu'
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
