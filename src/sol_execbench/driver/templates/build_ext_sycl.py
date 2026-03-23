# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build script for SYCL/DPC++ solutions targeting Intel GPU (XPU).

Uses ``torch.utils.cpp_extension.load()`` with ``with_sycl=True`` to compile
SYCL C++ source files using the Intel DPC++ compiler (icpx).
"""

import json
from pathlib import Path

import torch.utils.cpp_extension as ext

from sol_execbench.core import Solution

HERE = Path.cwd().resolve()

# Parse solution — validates sources (e.g. forbidden keywords) at compile time.
solution = Solution(**json.loads((HERE / "solution.json").read_text()))
compile_options = solution.spec.compile_options

sycl_cflags = ["-O3", "-ffast-math"] + (
    compile_options.sycl_cflags if compile_options else []
)
cflags = ["-O3"] + (compile_options.cflags if compile_options else [])
ld_flags = compile_options.ld_flags if compile_options else []

# Collect C++/SYCL source files from current directory
sources = [
    str(p)
    for p in HERE.iterdir()
    if p.suffix in (".sycl", ".cpp", ".cc", ".cxx", ".c") and p.is_file()
]
if not sources:
    raise RuntimeError("No SYCL/C++ source files found in working directory")

extra_include_paths = [str(HERE)]

ext.load(
    name="benchmark_kernel",
    sources=sources,
    extra_sycl_cflags=sycl_cflags,
    extra_cflags=cflags,
    extra_ldflags=ld_flags,
    extra_include_paths=extra_include_paths,
    build_directory=str(HERE),
    with_sycl=True,
    verbose=True,
)

# Rename platform-suffixed .so → benchmark_kernel.so
so_files = [
    f for f in HERE.glob("benchmark_kernel*.so") if f.name != "benchmark_kernel.so"
]
if so_files:
    so_files[0].rename("benchmark_kernel.so")
elif not (HERE / "benchmark_kernel.so").exists():
    raise FileNotFoundError("benchmark_kernel.so not produced by compilation")
