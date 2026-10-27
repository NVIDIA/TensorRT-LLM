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
"""CPU checks of the MiniMax-M3 MSA fmha_sm100 JIT-prefetch planner.

The planner's variant per longest context row, its build order, and the
kernels it lists for MiniMax-M3's standard context server (TP2, NVFP4 KV,
FP8 indexer). ``test_minimax_m3_fmha_jit_prefetch.py`` checks the variants
against the real ``fmha_sm100`` selection on a GPU.
"""

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import jit_prefetch_fmha as jpf

pytestmark = pytest.mark.cpu_only

FP8, BF16 = jpf._DTYPE_CODES["float8_e4m3fn"], jpf._DTYPE_CODES["bfloat16"]


def _name(key):
    """fmha_sm100's variant directory name for a get_fmha_variant key."""
    code, tile, single_wg, mode, page, split, pack = key
    return "_".join(
        str(i)
        for i in (
            {BF16: 0, FP8: 1}[code],
            {128: 0, 256: 1}[tile],
            0 if single_wg else 1,
            mode,
            {-1: 0, 128: 1}[page],
            int(split),
            (1, 2, 4, 6, 8, 16).index(pack),
        )
    )


@pytest.mark.parametrize(
    "qo_len, heads, kv_heads, mode, expected",
    [
        # Proxy max-score pass at TP2 (2 index heads over one MQA head).
        (1, 2, 1, jpf._ONLY_SCORE, "1_0_0_2_1_0_1"),
        (32, 2, 1, jpf._ONLY_SCORE, "1_0_0_2_1_0_1"),
        (33, 2, 1, jpf._ONLY_SCORE, "1_0_0_2_1_0_0"),
        (65, 2, 1, jpf._ONLY_SCORE, "1_0_1_2_1_0_0"),
        (129, 2, 1, jpf._ONLY_SCORE, "1_1_1_2_1_0_0"),
        # Dense layers at TP2 (32 q heads over 2 KV heads): pack factor 16 at 1-8
        # rows, 8 to 16, 4 to 32; then the 128- and 256-row tiles.
        (1, 32, 2, jpf._OFF, "1_0_0_3_1_0_5"),
        (8, 32, 2, jpf._OFF, "1_0_1_3_1_0_5"),
        (9, 32, 2, jpf._OFF, "1_0_1_3_1_0_4"),
        (17, 32, 2, jpf._OFF, "1_0_1_3_1_0_2"),
        (33, 32, 2, jpf._OFF, "1_0_0_3_1_0_0"),
        (65, 32, 2, jpf._OFF, "1_0_1_3_1_0_0"),
        (16384, 32, 2, jpf._OFF, "1_1_1_3_1_0_0"),
    ],
)
def test_row_variant_matches_fmha_sm100_names(qo_len, heads, kv_heads, mode, expected):
    assert _name(jpf.row_variant("float8_e4m3fn", heads, kv_heads, mode, 128, qo_len)) == expected


def test_split_proxy_variant_matches_fmha_sm100_names():
    """The KV-split form of the 128-row score-only kernel, which the indexer's
    segmented proxy plan runs whatever its split count."""
    assert _name(jpf.split_proxy_variant("float8_e4m3fn", 128)) == "1_0_1_2_1_1_0"
    assert _name(jpf.split_proxy_variant("bfloat16", 128)) == "0_0_1_2_1_1_0"


def test_block_sparse_long_rows_run_no_variant():
    """Block-sparse rows past 32 tokens are planned for the CuTe DSL sparse
    prefill, which builds no fmha_sm100 variant."""
    assert jpf.row_variant("float8_e4m3fn", 32, 2, jpf._SPARSE, 128, 32) is not None
    assert jpf.row_variant("float8_e4m3fn", 32, 2, jpf._SPARSE, 128, 33) is None


def test_rows_past_129_select_nothing_new():
    for heads, kv_heads, mode in ((2, 1, jpf._ONLY_SCORE), (32, 2, jpf._OFF), (16, 1, jpf._OFF)):
        ref = jpf.row_variant("float8_e4m3fn", heads, kv_heads, mode, 128, 129)
        for qo_len in (130, 255, 256, 257, 4096, 16384):
            assert jpf.row_variant("float8_e4m3fn", heads, kv_heads, mode, 128, qo_len) == ref


def _provider(monkeypatch, sites, nvfp4):
    """A provider over fake MSA layers: (q heads, KV heads, index heads,
    sparse layer) per layer."""
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import kernels, msa_backend
    from tensorrt_llm.bindings import DataType

    class FakeMsa(msa_backend.MiniMaxM3MsaSparseAttention):
        def __init__(self, q, kv, idx):  # noqa: D107 - no backend state needed
            self.m3_config = SimpleNamespace(
                num_q_heads=q, num_kv_heads=kv, num_index_heads=idx, block_size=128
            )

    class Layer:
        def __init__(self, q, kv, idx, sparse):
            self.attn = FakeMsa(q, kv, idx)
            self.is_sparse_attention_layer = sparse

    layers = [Layer(*site) for site in sites]
    model = SimpleNamespace(modules=lambda: iter(layers))
    kv_cache_manager = SimpleNamespace(
        tokens_per_block=128,
        dtype=DataType.NVFP4 if nvfp4 else DataType.FP8,
        indexer_kv_dtype="fp8",
    )
    monkeypatch.setattr("tensorrt_llm._utils.is_sm_100f", lambda *a: True)
    monkeypatch.setattr(kernels.msa_utils, "msa_package_available", lambda: True)
    return jpf.MiniMaxM3FmhaProvider(model, kv_cache_manager)


def test_provider_lists_the_standard_ctx_server_kernels(monkeypatch):
    """TP2 MiniMax-M3, NVFP4 KV (FP8 dense layers), FP8 indexer: the planner,
    the k2q CSR builder, and the 12 variants a cold server can build: the two
    of the indexer's segmented proxy plan and one per row class at each site."""
    provider = _provider(monkeypatch, [(32, 2, 2, False)] * 3 + [(32, 2, 2, True)] * 57, True)
    ops = [json.loads(s) for s in provider.specs]
    assert ops[0] == {"op": "plan"} and ops[1] == {"op": "k2q_csr"}
    names = [_name(op["args"]) for op in ops[2:]]
    assert len(names) == len(set(names))
    assert set(names) == {
        "1_0_0_3_1_0_5", "1_0_0_2_1_0_1", "1_1_1_3_1_0_0", "1_1_1_2_1_0_0",
        "1_0_0_3_1_0_0", "1_0_0_2_1_0_0", "1_0_1_3_1_0_2", "1_0_1_3_1_0_4",
        "1_0_1_3_1_0_0", "1_0_1_2_1_0_0", "1_0_1_3_1_0_5", "1_0_1_2_1_1_0",
    }  # fmt: skip
    # In the order warmup needs them: the segmented proxy plan's two variants
    # (unsplit, KV-split), which it loads before its first forward, then the
    # single-row and longest-row variants of every site.
    assert names[:2] == ["1_0_1_2_1_0_0", "1_0_1_2_1_1_0"]
    assert set(names[2:6]) == {"1_0_0_2_1_0_1", "1_0_0_3_1_0_5", "1_1_1_2_1_0_0", "1_1_1_3_1_0_0"}


def test_provider_fp8_cache_plans_block_sparse_short_rows(monkeypatch):
    provider = _provider(monkeypatch, [(32, 2, 2, False), (32, 2, 2, True)], False)
    ops = [json.loads(s) for s in provider.specs]
    assert {"op": "k2q_csr"} not in ops
    assert any(op.get("args", [None] * 4)[3] == jpf._SPARSE for op in ops)


def test_provider_without_msa_layers_lists_nothing(monkeypatch):
    assert not _provider(monkeypatch, [], True)


# FlashInfer modules: the MSA backend runs the dense layers' decode rows on
# FlashInfer's TRTLLM-GEN launcher (fmha_gen), which every rank builds in
# warmup's first forward with a generation request, context servers included.


def _fi_model(monkeypatch, msa_layers: int, sm_100f=True, msa=True, flashinfer=True):
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import kernels, msa_backend
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import (
        trtllm_gen_dense_decode as dense_decode,
    )

    class FakeMsa(msa_backend.MiniMaxM3MsaSparseAttention):
        def __init__(self):  # noqa: D107 - no backend state needed
            pass

    class Layer:
        def __init__(self):
            self.attn = FakeMsa()

    layers = [Layer() for _ in range(msa_layers)]
    monkeypatch.setattr("tensorrt_llm._utils.is_sm_100f", lambda *a: sm_100f)
    monkeypatch.setattr(kernels.msa_utils, "msa_package_available", lambda: msa)
    monkeypatch.setattr(dense_decode, "flashinfer_available", lambda: flashinfer)
    return SimpleNamespace(modules=lambda: iter([SimpleNamespace(), *layers]))


def test_msa_model_builds_the_trtllm_gen_launcher(monkeypatch):
    assert jpf.flashinfer_modules(_fi_model(monkeypatch, 60)) == ["fmha_gen"]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(msa_layers=0),  # no MSA layer (Triton sparse backend, other models)
        dict(msa_layers=60, sm_100f=False),
        dict(msa_layers=60, msa=False),
        dict(msa_layers=60, flashinfer=False),
    ],
    ids=["no-msa-layers", "not-sm100f", "no-fmha_sm100", "no-flashinfer"],
)
def test_other_engines_build_no_flashinfer_module(monkeypatch, kwargs):
    assert jpf.flashinfer_modules(_fi_model(monkeypatch, **kwargs)) == []


@pytest.mark.parametrize("prefetch", [False, True])
def test_start_model_builds_starts_fmha_gen_for_m3_only_with_prefetch(monkeypatch, prefetch):
    from unittest.mock import patch

    from tensorrt_llm._torch import jit_prefetch as jp
    from tensorrt_llm._torch import jit_prefetch_engine

    model = _fi_model(monkeypatch, 60)
    monkeypatch.setenv("TLLM_JIT_AHEAD", "1" if prefetch else "0")
    with patch.object(jp, "start_flashinfer_build", return_value=True) as start:
        started = jit_prefetch_engine.start_model_builds(model)
    assert started is prefetch
    if prefetch:
        # A layer's module: joined before the first forward, at this
        # process's priority.
        start.assert_called_once_with(("fmha_gen",), nice=0, layer=True)
    else:
        start.assert_not_called()


@pytest.mark.parametrize(
    "prefetch, tunes, sm_100f, started",
    [
        (True, True, True, True),
        (True, False, True, False),  # the engine runs no FlashInfer MXFP8 pass
        (True, True, False, False),  # SM120 has its own MXFP8 module
        (False, True, True, False),
    ],
)
def test_mxfp8_tune_modules_start_only_when_the_pass_runs(
    monkeypatch, prefetch, tunes, sm_100f, started
):
    from unittest.mock import patch

    from tensorrt_llm._torch import jit_prefetch as jp
    from tensorrt_llm._torch.modules import jit_prefetch_mxfp8 as mx

    monkeypatch.setenv("TLLM_JIT_AHEAD", "1" if prefetch else "0")
    monkeypatch.setattr("tensorrt_llm._utils.is_sm_100f", lambda *a: sm_100f)
    with patch.object(jp, "start_flashinfer_build", return_value=True) as start:
        assert mx.start_mxfp8_tune_builds(tunes) is started
    if started:
        start.assert_called_once_with(("trtllm_utils", "mxfp8_gemm_cutlass"))
    else:
        start.assert_not_called()


def _mxfp8_model(opt_in=True, cutlass=True, cp_size=1):
    """A decoder model with one MXFP8 Linear, opted into FlashInfer for decode
    CUDA graphs like DeepSeek-V4.1's Engram WKV and MiniMax-M3's layers."""
    from unittest.mock import patch

    from torch import nn

    from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
    from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod
    from tensorrt_llm.mapping import Mapping

    with patch(
        "tensorrt_llm._torch.modules.linear._mxfp8_cutlass_op_available", return_value=cutlass
    ):
        method = MXFP8LinearMethod()
    model = DecoderModelForCausalLM.__new__(DecoderModelForCausalLM)
    nn.Module.__init__(model)
    mapping = Mapping(world_size=cp_size, cp_size=cp_size) if cp_size > 1 else Mapping()
    model.model_config = SimpleNamespace(
        is_generation=True, is_encoder_decoder=False, mapping=mapping
    )
    model.wkv = nn.Module()
    model.wkv.quant_method = method
    model.wkv._use_flashinfer_mxfp8_decode_graph_default = opt_in
    return model


def _mxfp8_llm_args(compile_mode="eager", **overrides):
    """Engine arguments; ``compile_mode`` "all" compiles every batch,
    "prefill_only" only the piecewise prefill CUDA graphs."""
    from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

    args = dict(
        encode_only=False,
        mm_encoder_only=False,
        enable_autotuner=True,
        cuda_graph_config=SimpleNamespace(),
        torch_compile_config=None,
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.DISABLED,
    )
    if compile_mode != "eager":
        args["torch_compile_config"] = SimpleNamespace(
            compile_only_piecewise_graphs=compile_mode == "prefill_only"
        )
        if compile_mode == "prefill_only":
            args["prefill_cuda_graph_backend"] = PrefillCudaGraphBackend.PIECEWISE
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.mark.parametrize(
    "model_kw, args_kw, tunes",
    [
        ({}, {}, True),  # the GEN worker: decode CUDA graphs, eager decode
        ({}, dict(cuda_graph_config=None), False),  # the CTX worker: no decode graphs
        ({}, dict(enable_autotuner=False), False),
        (dict(opt_in=False), {}, False),  # no module opts its MXFP8 Linears in
        (dict(cutlass=False), {}, False),  # no CUTLASS MXFP8 op to run it on
        (dict(cp_size=2), {}, False),  # non-Helix CP skips warmup
        ({}, dict(encode_only=True), False),  # the encoder runner has no such pass
        ({}, dict(compile_mode="all"), False),  # torch.compile runs the decode batches
        ({}, dict(compile_mode="prefill_only"), True),  # ... only the prefill graphs
    ],
    ids=[
        "gen-worker",
        "no-cuda-graphs",
        "no-autotuner",
        "no-opt-in",
        "no-cutlass",
        "cp",
        "encoder",
        "compiled-decode",
        "compiled-prefill-only",
    ],
)
def test_mxfp8_tune_modules_predicted_at_load_as_warmup_decides(
    monkeypatch, model_kw, args_kw, tunes
):
    """``start_model_builds`` predicts the FlashInfer MXFP8 pass from the
    engine's arguments with the decoder runner's own selection, which its
    warmup applies to the engine's resolved settings: both agree."""
    from unittest.mock import patch

    import torch

    from tensorrt_llm._torch import jit_prefetch as jp
    from tensorrt_llm._torch import jit_prefetch_engine
    from tensorrt_llm._torch.modules import jit_prefetch_mxfp8 as mx
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import DecoderWarmup

    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "flashinfer_mm_mxfp8", object(), raising=False)
    model = _mxfp8_model(**model_kw)
    llm_args = _mxfp8_llm_args(**args_kw)
    assert mx.tunes_flashinfer_mxfp8_at_load(model, llm_args) is tunes
    if not llm_args.encode_only:
        # The engine's resolved settings for these arguments (the compile
        # ones are PyTorchModelEngine's, see test_pcg_fx_fallback_policy_*).
        compile_mode = args_kw.get("compile_mode", "eager")
        warmup = SimpleNamespace(
            model=model,
            mapping=model.model_config.mapping,
            _context_warmups_supported=True,
            _graphs=SimpleNamespace(enabled=llm_args.cuda_graph_config is not None),
            _config=SimpleNamespace(
                enable_autotuner=llm_args.enable_autotuner,
                torch_compile_enabled=compile_mode != "eager",
                torch_compile_prefill_only=compile_mode == "prefill_only",
            ),
        )
        assert DecoderWarmup._tunes_flashinfer_mxfp8(warmup) is tunes

    monkeypatch.setenv("TLLM_JIT_AHEAD", "1")
    monkeypatch.setattr("tensorrt_llm._utils.is_sm_100f", lambda *a: True)
    with patch.object(jp, "start_flashinfer_build", return_value=True) as start:
        jit_prefetch_engine.start_model_builds(
            SimpleNamespace(modules=model.modules), llm_args=None
        )
        start.assert_not_called()  # without the engine's arguments, warmup starts them
        with (
            patch.object(mx, "tunes_flashinfer_mxfp8_at_load", return_value=tunes),
            patch.object(jit_prefetch_engine, "warmup_builds_sampling_module", return_value=False),
        ):
            assert jit_prefetch_engine.start_model_builds(model, llm_args=llm_args) is tunes
    if tunes:
        start.assert_called_once_with(("trtllm_utils", "mxfp8_gemm_cutlass"))
    else:
        start.assert_not_called()


def _llm_args(algorithm="minimax_m3", implementation="msa", tokens_per_block=128, indexer="fp8"):
    return SimpleNamespace(
        sparse_attention_config=SimpleNamespace(
            algorithm=algorithm, implementation=implementation, indexer_kv_dtype=indexer
        ),
        kv_cache_config=SimpleNamespace(tokens_per_block=tokens_per_block),
    )


def _quant_model(kv_quant):
    from tensorrt_llm.models.modeling_utils import QuantConfig

    return SimpleNamespace(
        model_config=SimpleNamespace(quant_config=QuantConfig(kv_cache_quant_algo=kv_quant))
    )


@pytest.mark.parametrize("kv_quant, expected", [("NVFP4", "NVFP4"), ("FP8", "FP8"), (None, "BF16")])
def test_kv_cache_geometry_follows_the_executor(kv_quant, expected):
    """The page size, cache dtype and indexer dtype the KV cache manager will
    have, as the fmha_sm100 planner reads them, from the engine's arguments."""
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.quantization.mode import QuantAlgo

    geometry = jpf.kv_cache_geometry(
        _quant_model(QuantAlgo[kv_quant] if kv_quant else None), _llm_args()
    )
    assert geometry.tokens_per_block == 128
    assert geometry.dtype == getattr(DataType, expected)
    assert geometry.indexer_kv_dtype == "fp8"


@pytest.mark.parametrize(
    "llm_args",
    [_llm_args(implementation="triton"), _llm_args(algorithm="dsa"), _llm_args(tokens_per_block=0)],
    ids=["triton-backend", "other-algorithm", "no-page-size"],
)
def test_kv_cache_geometry_none_without_msa(llm_args):
    assert jpf.kv_cache_geometry(_quant_model(None), llm_args) is None


def test_load_time_plan_equals_the_warmup_plan(monkeypatch):
    """Planned from the engine's arguments while the weights load, the
    standard context server lists the same kernels as from its KV cache
    manager at warmup."""
    from tensorrt_llm.quantization.mode import QuantAlgo

    sites = [(32, 2, 2, False)] * 3 + [(32, 2, 2, True)] * 57
    at_warmup = _provider(monkeypatch, sites, True)
    model = SimpleNamespace(
        modules=_provider_model(sites).modules,
        model_config=_quant_model(QuantAlgo.NVFP4).model_config,
    )
    geometry = jpf.kv_cache_geometry(model, _llm_args())
    at_load = jpf.MiniMaxM3FmhaProvider(model, geometry)
    assert at_load.specs == at_warmup.specs


def _provider_model(sites):
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_backend

    class FakeMsa(msa_backend.MiniMaxM3MsaSparseAttention):
        def __init__(self, q, kv, idx):  # noqa: D107 - no backend state needed
            self.m3_config = SimpleNamespace(
                num_q_heads=q, num_kv_heads=kv, num_index_heads=idx, block_size=128
            )

    class Layer:
        def __init__(self, q, kv, idx, sparse):
            self.attn = FakeMsa(q, kv, idx)
            self.is_sparse_attention_layer = sparse

    layers = [Layer(*site) for site in sites]
    return SimpleNamespace(modules=lambda: iter(layers))


@pytest.mark.parametrize(
    "args_kw, sampling",
    [
        ({}, True),  # a decoder engine: warmup builds the sampling module
        (dict(encode_only=True), False),  # the encoder runner never samples
    ],
    ids=["decoder", "encoder"],
)
def test_sampling_module_starts_in_a_helper_of_its_own_at_load(monkeypatch, args_kw, sampling):
    """``start_model_builds`` starts FlashInfer's sampling module, which every
    decoder runner's warmup builds, in a helper of its own, next to the one
    building the model's own modules: those, which warmup joins before its
    first forward, at this process's priority. An engine whose runner never
    samples does not build it."""
    from unittest.mock import call, patch

    from tensorrt_llm._torch import jit_prefetch as jp
    from tensorrt_llm._torch import jit_prefetch_engine
    from tensorrt_llm._torch.modules import jit_prefetch_mxfp8 as mx

    model = _mxfp8_model(opt_in=False)
    model.modules = _fi_model(monkeypatch, 60).modules
    llm_args = _mxfp8_llm_args(**args_kw)
    assert jit_prefetch_engine.warmup_builds_sampling_module(model, llm_args) is sampling
    monkeypatch.setenv("TLLM_JIT_AHEAD", "1")
    with (
        patch.object(jp, "start_flashinfer_build", return_value=True) as start,
        patch.object(mx, "tunes_flashinfer_mxfp8_at_load", return_value=False),
    ):
        assert jit_prefetch_engine.start_model_builds(model, llm_args=llm_args)
    layer = call(("fmha_gen",), nice=0, layer=True)
    assert start.call_args_list == ([layer, call(("sampling",))] if sampling else [layer])
