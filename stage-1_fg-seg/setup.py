from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sys
import os

class BuildExt(build_ext):
    """Custom build extension to add compiler flags"""
    
    c_opts = {
        'msvc': ['/O2', '/openmp', '/std:c++14'],
        'unix': ['-O3', '-fopenmp', '-std=c++14', '-march=native'],
    }
    
    l_opts = {
        'msvc': [],
        'unix': ['-fopenmp'],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        
        # Add CUDA-ready flags (optional, for future CUDA integration)
        if os.environ.get('USE_CUDA', '0') == '1':
            opts.extend(['-DUSE_CUDA'])
        
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        'pixel2voxel.voxel_ops',
        sources=[
            'pixel2voxel/native/bindings.cpp',
            'pixel2voxel/native/voxel_core.cpp',
        ],
        include_dirs=[
            pybind11.get_include(),
            'pixel2voxel/native',
        ],
        language='c++',
    ),
]

setup(
    name='pixel2voxel',
    version='0.1.0',
    author='Pixel2Voxel Team',
    description='C++ accelerated voxel operations for pixel2voxel',
    packages=['pixel2voxel'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    install_requires=['pybind11'],
    python_requires='>=3.8',
)
