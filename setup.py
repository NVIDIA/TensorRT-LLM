# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import platform

from setuptools import find_packages, setup
from setuptools.dist import Distribution

requirements_filename = "requirements-windows.txt" if platform.system(
) == "Windows" else "requirements.txt"
with open(requirements_filename) as f:
    requirements = f.read().splitlines()

    def extract_url(line):
        return next(filter(lambda x: x[0] != '-', line.split()))

    extra_URLs = []
    required_deps = []
    for line in requirements:
        if line[0] == "#":
            continue

        # handle -i and --extra-index-url options
        if "-i " in line or "--extra-index-url" in line:
            extra_URLs.append(extract_url(line))
        else:
            required_deps.append(line)


class BinaryDistribution(Distribution):

    def has_ext_modules(self):
        return False

    def is_pure(self):
        return True


setup_kwargs = {}

setup(
    name='tensorrt_llm',
    version='0.5.0',
    description='TensorRT-LLM: A TensorRT Toolbox for Large Language Models',
    install_requires=required_deps,
    dependency_links=extra_URLs,
    zip_safe=True,
    packages=find_packages(),
    package_data={
        'tensorrt_llm':
        (['libs/th_common.dll', 'libs/nvinfer_plugin_tensorrt_llm.dll']
         if platform.system() == "Windows" else
         ['libs/libth_common.so', 'libs/libnvinfer_plugin_tensorrt_llm.so']) +
        ['tools/plugin_gen/templates/*']
    },
    python_requires=">=3.7, <4",
    distclass=BinaryDistribution,
    **setup_kwargs,
)
