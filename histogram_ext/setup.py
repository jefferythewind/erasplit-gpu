from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='histogram_ext',
    ext_modules=[
        CUDAExtension(
            name='histogram_ext',
            sources=[
                'histogram.cpp',
                'histogram_kernel.cu',
                'best_split_kernel.cu',
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
