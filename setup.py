"""Build configuration for the optional CUDA extension."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent

setup(
    ext_modules=[
        CUDAExtension(
            name="fused_hoad._C",
            sources=[
                str(ROOT / "csrc" / "bindings.cpp"),
                str(ROOT / "csrc" / "fused_residual.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

