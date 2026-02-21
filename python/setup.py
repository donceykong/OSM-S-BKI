"""
Build the composite_bki_cpp extension (pybind11) and install osm_bki package.
Scripts expect: import composite_bki_cpp; composite_bki_cpp.run_pipeline(...); composite_bki_cpp.PySemanticBKI(...)
"""

import os
import sys
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class build_ext_in_build_dir(build_ext):
    """Build extension into python/build/ instead of in-place or build/lib.*/."""
    def initialize_options(self):
        build_ext.initialize_options(self)
        self.build_lib = "build"


def get_include_dirs():
    root = Path(__file__).resolve().parent
    cpp_include = root.parent / "cpp" / "osm_bki" / "include"
    if not cpp_include.is_dir():
        raise RuntimeError(f"C++ include dir not found: {cpp_include}")
    return [str(cpp_include)]


def get_sources():
    root = Path(__file__).resolve().parent
    cpp_src = root.parent / "cpp" / "osm_bki" / "src"
    pybind_src = root / "osm_bki" / "pybind" / "osm_bki_bindings.cpp"
    sources = [
        str(pybind_src),
        str(cpp_src / "continuous_bki.cpp"),
        str(cpp_src / "file_io.cpp"),
        str(cpp_src / "dataset_utils.cpp"),
    ]
    for s in sources:
        if not Path(s).exists():
            raise RuntimeError(f"Source not found: {s}")
    return sources


class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()


def extra_compile_args():
    args = ["-std=c++17", "-O3"]
    # OpenMP
    import subprocess
    try:
        subprocess.check_output(["g++", "-fopenmp", "-E", "-"], stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        args.append("-fopenmp")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return args


def extra_link_args():
    args = []
    try:
        import subprocess
        subprocess.check_output(["g++", "-fopenmp", "-E", "-"], stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        args.append("-fopenmp")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return args


ext = Extension(
    "composite_bki_cpp",
    sources=get_sources(),
    include_dirs=[
        get_pybind_include(),
        *get_include_dirs(),
    ],
    language="c++",
    extra_compile_args=extra_compile_args(),
    extra_link_args=extra_link_args(),
)

setup(
    name="osm_bki",
    version="2.0.0",
    description="OSM-S-BKI / Composite BKI: semantic BKI for LiDAR with OSM priors",
    author="Composite BKI Team",
    license="MIT",
    python_requires=">=3.7",
    packages=["osm_bki"],
    package_dir={"osm_bki": "osm_bki"},
    ext_modules=[ext],
    install_requires=["numpy>=1.20.0", "pybind11>=2.6.0"],
    setup_requires=["pybind11>=2.6.0"],
    cmdclass={"build_ext": build_ext_in_build_dir},
)
