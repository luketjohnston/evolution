from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="binary_forward",
      ext_modules=[
          cpp_extension.CUDAExtension("binary_forward", ["../kernels/binary/binary.cu"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

