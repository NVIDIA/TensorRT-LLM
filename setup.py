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
from setuptools import find_packages, setup
from setuptools.dist import Distribution

with open("requirements.txt") as f:
    required_deps = f.read().splitlines()


class BinaryDistribution(Distribution):

    def has_ext_modules(self):
        return False

    def is_pure(self):
        return True


setup_kwargs = {}

setup(
    name='tensorrt_llm',
    version='0.1.3',
    description='TensorRT-LLM: A TensorRT Toolbox for Large Language Models',
    install_requires=required_deps,
    zip_safe=True,
    packages=find_packages(),
    package_data={
        'tensorrt_llm':
        ['libs/libth_common.so', 'libs/libnvinfer_plugin_tensorrt_llm.so']
    },
    python_requires=">=3.7, <4",
    distclass=BinaryDistribution,
    **setup_kwargs,
)
