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
import contextlib
import platform
from pathlib import Path

import torch

from ._utils import str_dtype_to_trt
from .logger import logger
from .plugin import _load_plugin_lib

net = None

_inited = False


def _init(log_level=None):
    global _inited
    if _inited:
        return
    _inited = True
    # Move to __init__
    if log_level is not None:
        logger.set_level(log_level)

    # load plugin lib
    _load_plugin_lib()

    # load FT decoder layer
    project_dir = str(Path(__file__).parent.absolute())
    if platform.system() == "Windows":
        ft_decoder_lib = project_dir + '/libs/th_common.dll'
    else:
        ft_decoder_lib = project_dir + '/libs/libth_common.so'
    if ft_decoder_lib == '':
        raise ImportError('FT decoder layer is unavailable')
    torch.classes.load_library(ft_decoder_lib)

    global net
    logger.info('TensorRT-LLM inited.')


def default_net():
    assert net, "Use builder to create network first, and use `set_network` or `net_guard` to set it to default"
    return net


def default_trtnet():
    return default_net().trt_network


def set_network(network):
    global net
    net = network


def switch_net_dtype(cur_dtype):
    prev_dtype = default_net().dtype
    default_net().dtype = cur_dtype
    return prev_dtype


@contextlib.contextmanager
def precision(dtype):
    if isinstance(dtype, str):
        dtype = str_dtype_to_trt(dtype)
    prev_dtype = switch_net_dtype(dtype)
    yield
    switch_net_dtype(prev_dtype)
