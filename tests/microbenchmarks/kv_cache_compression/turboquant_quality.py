# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconstruction quality of the cold-page codecs on structured inputs, as a markdown table.

For every TurboQuant setting (and NVFP4 for reference) this encodes Pages of 128-wide K/V vectors
into pinned host cold pages through the provider interface and decodes them back, then reports:

* the normalized MSE ||x - x_hat||^2 / ||x||^2 for Gaussian vectors, vectors with a few large
  channels (like real K), sparse vectors, one-hot vectors, the all-equal vector, and vectors whose
  norms span 1e-6..1e6 within a Page;
* for ``prod``, the bias and the variance (times d) of <y, x_hat> - <y, x> for unit x and y;
* the normalized MSE after one and after four round trips (each re-encoding the decoded Page).

Run on one GPU::

    python3 tests/microbenchmarks/kv_cache_compression/turboquant_quality.py
"""

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.pyexecutor._util import create_kv_cache_compression_manager
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm.bindings.internal import kv_cache_compression as native
from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig, KvCacheConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, BufferConfig

DIM, HEADS, TOKENS, PAGES = 128, 8, 32, 16
SETTINGS = [("mse", b) for b in (2, 2.5, 3, 3.5, 4)] + [("prod", b) for b in (2, 3, 4)]
INPUTS = ("gaussian", "outliers", "sparse", "one_hot", "constant", "mixed_norms")


def _manager(config):
    pretrained = SimpleNamespace(model_type="llama")
    engine = SimpleNamespace(
        mapping=SimpleNamespace(has_cp_helix=lambda: False),
        cuda_graph_batch_sizes=[],
        cuda_graph_padding_enabled=False,
        spec_config=None,
        model=SimpleNamespace(
            model_config=SimpleNamespace(quant_config=None, pretrained_config=pretrained)
        ),
    )
    return create_kv_cache_compression_manager(
        config,
        model_engine=engine,
        kv_cache_config=KvCacheConfig(enable_block_reuse=True, host_cache_size=1 << 30),
    )


class _Codec:
    """One K/V layer with 2 * PAGES hot slots and PAGES pinned cold pages."""

    def __init__(self, manager):
        self.manager = manager
        shape = (2 * PAGES, HEADS, TOKENS, DIM)
        self.hot = {r: torch.zeros(shape, dtype=torch.bfloat16, device="cuda") for r in "kv"}
        size = HEADS * TOKENS * DIM * 2
        cache = SimpleNamespace(
            tokens_per_block=TOKENS,
            layers=(
                AttentionLayerConfig(
                    layer_id=0,
                    buffers=[
                        BufferConfig(role="key", size=size),
                        BufferConfig(role="value", size=size),
                    ],
                ),
            ),
        )
        self.state = manager.build_codec_state(
            cache,
            runtime_dtype=DataType.BF16,
            pp_layers=(0,),
            num_kv_heads_per_layer=(HEADS,),
            head_dim_per_layer=(DIM,),
            is_draft=False,
        )
        hot = {
            role: SimpleNamespace(raw_base=t.data_ptr(), raw_slot_bytes=size, raw_bytes=size)
            for role, t in zip(("key", "value"), self.hot.values())
        }
        (props,) = manager.configure(self.state, [SimpleNamespace(layers={0: hot})])
        self.device = (
            "cuda" if props.page_index_location == native.ColdPageIndexLocation.DEVICE else "cpu"
        )
        self.cold = torch.empty(
            PAGES * int(props.cold_page_bytes), dtype=torch.uint8, pin_memory=True
        )
        self.stream = torch.cuda.Stream()

    def _call(self, method, dst, src):
        pairs = torch.stack([torch.as_tensor(dst), torch.as_tensor(src)], 1).to(torch.int32)
        pairs = pairs.to(self.device)
        method(
            self.state,
            0,
            self.cold.data_ptr(),
            pairs.data_ptr(),
            len(pairs),
            self.stream.cuda_stream,
        )
        self.stream.synchronize()

    def round_trips(self, values, trips):
        """NMSE after each of ``trips`` round trips of ``values`` [PAGES, HEADS, TOKENS, DIM]."""
        for tensor in self.hot.values():
            tensor[:PAGES] = values.to(torch.bfloat16)
        pages, low, high = torch.arange(PAGES), torch.arange(PAGES), torch.arange(PAGES) + PAGES
        errors = []
        for trip in range(trips):
            self._call(self.manager.encode_cold_pages, pages, low if trip == 0 else high)
            self._call(self.manager.decode_cold_pages, high, pages)
            errors.append(_nmse(values.to(torch.bfloat16), self.hot["k"][PAGES:]))
        return errors


def _nmse(x, y):
    x, y = x.double().flatten(0, -2), y.double().flatten(0, -2)
    norm = (x * x).sum(-1)
    keep = norm > 0
    return float((((x - y) ** 2).sum(-1)[keep] / norm[keep]).mean())


def _inputs(kind, generator):
    shape = (PAGES, HEADS, TOKENS, DIM)
    x = torch.randn(shape, generator=generator, device="cuda")
    if kind == "outliers":
        x[..., 3] *= 25.0
        x[..., 71] *= 10.0
    elif kind == "sparse":
        x *= torch.rand(shape, generator=generator, device="cuda").argsort(-1) < 4
    elif kind == "one_hot":
        index = torch.randint(0, DIM, shape[:-1], generator=generator, device="cuda")
        x = torch.nn.functional.one_hot(index, DIM).float() * 3.0
    elif kind == "constant":
        x = torch.full(shape, 0.75, device="cuda")
    elif kind == "mixed_norms":
        x *= 10.0 ** torch.randint(-6, 7, (*shape[:-1], 1), generator=generator, device="cuda")
    return x


def _inner_products(codec, generator):
    x = _inputs("outliers", generator)
    x /= x.norm(dim=-1, keepdim=True)
    codec.round_trips(x, 1)
    y = torch.randn(x.shape, generator=generator, device="cuda", dtype=torch.float64)
    y /= y.norm(dim=-1, keepdim=True)
    error = ((codec.hot["k"][PAGES:].double() - x.to(torch.bfloat16).double()) * y).sum(-1)
    return float(error.mean()), float(error.var()) * DIM


def main():
    rows = []
    for mode, bits in SETTINGS + [("nvfp4", None)]:
        if mode == "nvfp4":
            config = ColdPageQuantizationCompressionConfig(quant="nvfp4")
        else:
            config = ColdPageQuantizationCompressionConfig(
                quant="turboquant", turboquant_bits=bits, turboquant_mode=mode
            )
        codec = _Codec(_manager(config))
        generator = torch.Generator(device="cuda").manual_seed(0)
        errors = [codec.round_trips(_inputs(kind, generator), 1)[0] for kind in INPUTS]
        bias, variance = _inner_products(codec, generator) if mode != "mse" else (None, None)
        trips = codec.round_trips(_inputs("outliers", generator), 4)
        rows.append((mode, bits, errors, bias, variance, trips[0], trips[-1]))
    print("| mode | bits | " + " | ".join(INPUTS) + " | IP bias | IP var x d | 1 trip | 4 trips |")
    print("|---|---|" + "---|" * (len(INPUTS) + 4))
    for mode, bits, errors, bias, variance, first, last in rows:
        ip = (f"{bias:+.1e} | {variance:.3f}") if bias is not None else "- | -"
        print(
            f"| {mode} | {bits if bits is not None else 4.5} | "
            + " | ".join(f"{e:.4f}" for e in errors)
            + f" | {ip} | {first:.4f} | {last:.4f} |"
        )


if __name__ == "__main__":
    main()
