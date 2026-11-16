# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconstruction quality of the cold-page codecs on structured inputs, as a markdown table.

For every TurboQuant setting (and NVFP4 for reference) this encodes Pages of 128-wide K/V vectors
into pinned host cold pages through the provider interface and decodes them back, then reports:

* the normalized MSE ||x - x_hat||^2 / ||x||^2 for Gaussian vectors, vectors with a few large
  channels (like real K), sparse vectors, one-hot vectors, the all-equal vector, and vectors whose
  norms span 1e-6..1e6 within a Page;
* the normalized MSE after 1, 4 and 16 round trips, each re-encoding the decoded Page as an
  onboarded Page that is offloaded again would be.

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
BITS = (2, 2.5, 3, 3.5, 4)
INPUTS = ("gaussian", "outliers", "sparse", "one_hot", "constant", "mixed_norms")
TRIPS = (1, 4, 16)


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
        # The codec runs on its own stream: order it after the hot-buffer writes.
        self.stream.wait_stream(torch.cuda.current_stream())
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
        """NMSE after each of ``trips`` round trips of ``values`` [PAGES, HEADS, TOKENS, DIM].

        K holds ``values`` and V the same vectors in another Page order, so that a decoder that
        mixes up the buffers shows; the NMSE is over the K and V vectors together.
        """
        source = {"k": values.to(torch.bfloat16), "v": values.roll(1, 0).to(torch.bfloat16)}
        for role, tensor in self.hot.items():
            tensor[:PAGES] = source[role]
        pages, low, high = torch.arange(PAGES), torch.arange(PAGES), torch.arange(PAGES) + PAGES
        errors = []
        for trip in range(trips):
            self._call(self.manager.encode_cold_pages, pages, low if trip == 0 else high)
            self._call(self.manager.decode_cold_pages, high, pages)
            reference = torch.cat([source[role] for role in "kv"])
            decoded = torch.cat([self.hot[role][PAGES:] for role in "kv"])
            errors.append(_nmse(reference, decoded))
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


def main():
    rows = []
    for fmt, bits in [("turboquant", b) for b in BITS] + [("nvfp4", None)]:
        if fmt == "nvfp4":
            config = ColdPageQuantizationCompressionConfig(quant="nvfp4")
        else:
            config = ColdPageQuantizationCompressionConfig(quant="turboquant", turboquant_bits=bits)
        codec = _Codec(_manager(config))
        generator = torch.Generator(device="cuda").manual_seed(0)
        errors = [codec.round_trips(_inputs(kind, generator), 1)[0] for kind in INPUTS]
        trips = codec.round_trips(_inputs("outliers", generator), max(TRIPS))
        rows.append((fmt, bits, errors, [trips[n - 1] for n in TRIPS]))
    trip_columns = " | ".join(f"{n} trip{'s' if n > 1 else ''}" for n in TRIPS)
    print("| format | bits | " + " | ".join(INPUTS) + f" | {trip_columns} |")
    print("|---|---|" + "---|" * (len(INPUTS) + len(TRIPS)))
    for fmt, bits, errors, trips in rows:
        print(
            f"| {fmt} | {bits if bits is not None else 4.5} | "
            + " | ".join(f"{e:.4f}" for e in errors + trips)
            + " |"
        )


if __name__ == "__main__":
    main()
