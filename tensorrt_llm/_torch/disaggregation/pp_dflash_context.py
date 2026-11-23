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
"""Whether this process serves a stage of a pipeline-parallel DFlash context server.

Such a server is a disaggregated context server at ``pipeline_parallel_size > 1``
whose speculative config is a one-model DFlash drafter on the TRTLLM attention
backend, and whose decoder is built by a modeling target that serves a pipeline
stage of a context server (the MuseSparkOpen ``pp_dflash_context_server()``
scope; see ``modeling_musespark_open.pipeline_parallel_dflash_context``). The
decoder's construction decides it from that configuration and records it here
(:func:`set_pp_dflash_context_server`); every component that changes its
behavior for such a server reads it through :func:`pp_dflash_context_enabled`,
after the model is built, so they cannot disagree on its value:

* the KV cache creator places every draft layer on the last stage, and every
  stage charges the drafter's block width;
* the KV cache manager of every stage keeps only the drafter's attention window
  of each prompt's drafter context (plus the reuse rewind and one block), which
  the model engine hands to the drafter context write, and caps reuse claims so
  that no request's window inherits a row that was not written
  (``KVCacheManagerV2.enable_draft_context_window``);
* the cache transceiver publishes the instance-wide draft-layer count;
* the executor's first rank admits a new prompt only while every stage has KV
  room for it (``pyexecutor/pp_admission.py``).

Every other deployment, ``pipeline_parallel_size == 1`` included, never sets it.

This module imports nothing from the package, so any of them can import it.
"""

_pp_dflash_context_server = False


def set_pp_dflash_context_server(enabled: bool) -> None:
    """Record whether this process serves a stage of a pipeline-parallel DFlash context server.

    Called by the decoder's construction, on every rank of the server alike.
    """
    global _pp_dflash_context_server
    _pp_dflash_context_server = bool(enabled)


def pp_dflash_context_enabled() -> bool:
    """Whether this process serves a stage of a pipeline-parallel DFlash context server.

    False until a decoder construction records it (:func:`set_pp_dflash_context_server`).
    """
    return _pp_dflash_context_server
