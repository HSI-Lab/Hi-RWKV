from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='wkv',
    ext_modules=[
        CUDAExtension(
            name='wkv',
            sources=['model/cuda/wkv_op.cpp', 'model/cuda/wkv_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-res-usage', '--maxrregcount=60', '--use_fast_math', '-Xptxas=-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
