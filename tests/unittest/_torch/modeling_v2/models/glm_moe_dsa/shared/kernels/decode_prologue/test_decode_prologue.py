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
"""decode_prologue target prologue versus the native prologue, byte for byte (TP1 oracle).

Native: the literal ``DecoderRunner._preprocess_inputs`` (two
``on_update_kv_lens()`` calls around the overlap scheduler's accepted-count
correction) and the target model's native ``Embedding``. Candidate: decode_prologue's one
launch. Both run on identically prepared CUDA-graph DSA metadata of a real
``DSACacheManager`` (served GEN DSA configuration) with the GLM-5.2
embedding table. Before every run every destination buffer is poisoned
identically (0xA5) in both arms; afterwards every metadata attribute either
arm changed, every destination buffer in full, the position ids, the input ids,
the offsets and the embedding rows must be bitwise equal.

Cases: accepted counts 0-5 (KV offsets -5..0), contexts 2K-1M with lengths
at k*256-1, k*256 and k*256+1 (DeepGEMM's SPLIT_KV), B in {1, 2}, a padded
B = 2 graph whose second row is a short dummy, every DSL atom-split factor the
picker yields (6, 3, 2), the dense top-k (skipped indexer) graph, out-of-range
input ids, the single-call prologue without the correction (overlap scheduler
off, or spec decode off for the step), the DeepGEMM paged-MQA path without the
DSL atom split (factor 1); eager and CUDA graph with changing inputs at every
replay.
"""

import os
import re
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

import _decode_prologue_oracle as orc  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.decode_prologue import (  # noqa: E402, E501
    prologue,
)

SYNTHETIC = os.environ.get("GLM_MOE_DSA_DECODE_PROLOGUE_SYNTHETIC_EMBEDDING") == "1"


@pytest.fixture(scope="module")
def weight():
    return orc.embedding_weight(synthetic=SYNTHETIC)


def _boundary(k: int, d: int) -> int:
    return k * 256 + d


# (past KV lengths at prepare, list of per-replay post-correction KV lengths)
CONTEXT_CASES = {
    # B = 1: atom split factor 6 (2K), 3 (~12.8K), 2 (>= 32K); SPLIT_KV edges.
    "b1_2k": ([2100], [[_boundary(9, d)] for d in (-1, 0, 1)]),
    "b1_12k": ([12700], [[_boundary(50, d)] for d in (-1, 0, 1)]),
    "b1_32k": ([32768], [[_boundary(128, d)] for d in (-1, 0, 1, 7)]),
    "b1_131k": ([131072], [[_boundary(512, d)] for d in (-1, 0, 1)]),
    "b1_491k": (
        [491061],
        [[491067], [_boundary(1918, 1)], [_boundary(1919, 0)], [_boundary(1919, -1)]],
    ),
    "b1_1m": ([1048000], [[_boundary(4094, d)] for d in (-1, 0, 1)]),
    # B = 2 live rows (different contexts).
    "b2_mixed": ([491061, 32768], [[491067, 32774], [_boundary(1919, 1), _boundary(128, -1)]]),
    # Padded: a B = 2 graph whose second row is a short dummy request.
    "b2_padded": ([131072, 1], [[131078, 7], [_boundary(512, 0), 7]]),
}

ACCEPTED_SWEEP = [0, 1, 2, 3, 4, 5]


def _run_eager_pair(native, decode_prologue, step):
    for arm in (native, decode_prologue):
        arm.set_step(*step)
        arm.poison()
    torch.cuda.synchronize()
    nb, zb = orc.state(native), orc.state(decode_prologue)
    out_n = orc.run_native(native)
    out_z = orc.run_decode_prologue(decode_prologue)
    torch.cuda.synchronize()
    na, za = orc.state(native), orc.state(decode_prologue)
    bad = orc.compare(nb, na, zb, za)
    assert out_n.shape == out_z.shape and out_n.dtype == out_z.dtype
    assert torch.equal(orc._bits(out_n), orc._bits(out_z)), "embedding rows differ"
    return bad


@pytest.mark.parametrize("case", sorted(CONTEXT_CASES))
def test_eager_matches_native(case, weight):
    past, kv_steps = CONTEXT_CASES[case]
    cm, native, decode_prologue = orc.make_pair(past, weight)
    try:
        md = decode_prologue.md
        reason = prologue.prologue_ineligibility(
            decode_prologue.runner, decode_prologue.inputs(), True
        )
        assert reason is None, reason
        print(
            f"[decode_prologue] {case}: B={md.num_seqs} N={md.num_tokens} expand_for_dsl={md.expand_for_dsl} "
            f"factor={md.dsl_expand_factor} width={md.gen_token_stride} "
            f"dense={md.skip_indexer_for_gen_reqs} num_sms={md.num_sms}"
        )
        gen = torch.Generator().manual_seed(len(case))
        trials = 0
        for kv_base in kv_steps:
            for a in ACCEPTED_SWEEP:
                accepted = [a] + [(a + 3) % 6 for _ in kv_base[1:]]
                if case == "b2_padded":
                    accepted[1] = 5  # a padding row has offset 0
                step = orc.step_inputs(gen, kv_base, accepted, native.num_tokens, id_edges=a == 0)
                bad = _run_eager_pair(native, decode_prologue, step)
                assert not bad, (case, kv_base, accepted, bad)
                trials += 1
        print(f"[decode_prologue] {case}: {trials} eager trials identical")
    finally:
        cm.shutdown()


@pytest.mark.parametrize("case", ["b1_491k", "b1_32k", "b1_2k", "b2_mixed", "b2_padded"])
def test_cuda_graph_replays_match_native(case, weight):
    past, kv_steps = CONTEXT_CASES[case]
    cm, native, decode_prologue = orc.make_pair(past, weight)
    try:
        g_n, out_n = orc.capture(native, orc.run_native)
        g_z, out_z = orc.capture(decode_prologue, orc.run_decode_prologue)
        gen = torch.Generator().manual_seed(7)
        replays = 0
        for rep in range(4):
            for kv_base in kv_steps:
                accepted = [int(x) for x in torch.randint(0, 6, (len(kv_base),), generator=gen)]
                if case == "b2_padded":
                    accepted[1] = 5
                step = orc.step_inputs(gen, kv_base, accepted, native.num_tokens, id_edges=rep == 1)
                for arm in (native, decode_prologue):
                    arm.set_step(*step)
                    arm.poison()
                torch.cuda.synchronize()
                nb, zb = orc.state(native), orc.state(decode_prologue)
                g_n.replay()
                g_z.replay()
                torch.cuda.synchronize()
                bad = orc.compare(nb, orc.state(native), zb, orc.state(decode_prologue))
                assert not bad, (case, rep, kv_base, accepted, bad)
                assert torch.equal(orc._bits(out_n), orc._bits(out_z)), (case, rep, "embedding")
                replays += 1
        assert replays >= 8
        print(f"[decode_prologue] {case}: {replays} graph replays identical")
    finally:
        cm.shutdown()


def test_dense_topk_graph_matches_native(weight):
    """Short context: the indexer is skipped and the dense top-k rows are written."""
    cm, native, decode_prologue = orc.make_pair([1000], weight, max_kv=4096)
    try:
        assert decode_prologue.md.skip_indexer_for_gen_reqs
        reason = prologue.prologue_ineligibility(
            decode_prologue.runner, decode_prologue.inputs(), True
        )
        assert reason is None, reason
        g_n, out_n = orc.capture(native, orc.run_native)
        g_z, out_z = orc.capture(decode_prologue, orc.run_decode_prologue)
        gen = torch.Generator().manual_seed(3)
        for kv_base in ([1006], [1001], [2040], [7]):
            for a in ACCEPTED_SWEEP:
                step = orc.step_inputs(gen, kv_base, [a], native.num_tokens)
                for arm in (native, decode_prologue):
                    arm.set_step(*step)
                    arm.poison()
                torch.cuda.synchronize()
                nb, zb = orc.state(native), orc.state(decode_prologue)
                g_n.replay()
                g_z.replay()
                torch.cuda.synchronize()
                bad = orc.compare(nb, orc.state(native), zb, orc.state(decode_prologue))
                assert not bad, (kv_base, a, bad)
                assert torch.equal(orc._bits(out_n), orc._bits(out_z))
    finally:
        cm.shutdown()


def _eager_and_graph(
    native, decode_prologue, kv_steps, run_n, run_z, label, sweep=(0, 2, 5), reps=2
):
    """Eager trials over ``sweep`` accepted counts, then CUDA-graph replays; returns the counts."""
    gen = torch.Generator().manual_seed(11)
    trials = 0
    for kv_base in kv_steps:
        for a in sweep:
            accepted = [a] + [(a + 3) % 6 for _ in kv_base[1:]]
            step = orc.step_inputs(gen, kv_base, accepted, native.num_tokens, id_edges=a == 0)
            for arm in (native, decode_prologue):
                arm.set_step(*step)
                arm.poison()
            torch.cuda.synchronize()
            nb, zb = orc.state(native), orc.state(decode_prologue)
            out_n, out_z = run_n(native), run_z(decode_prologue)
            torch.cuda.synchronize()
            bad = orc.compare(nb, orc.state(native), zb, orc.state(decode_prologue))
            assert not bad, (label, kv_base, accepted, bad)
            assert torch.equal(orc._bits(out_n), orc._bits(out_z)), (label, "embedding")
            trials += 1

    g_n, out_n = orc.capture(native, run_n)
    g_z, out_z = orc.capture(decode_prologue, run_z)
    replays = 0
    for rep in range(reps):
        for kv_base in kv_steps:
            accepted = [int(x) for x in torch.randint(0, 6, (len(kv_base),), generator=gen)]
            step = orc.step_inputs(gen, kv_base, accepted, native.num_tokens, id_edges=rep == 1)
            for arm in (native, decode_prologue):
                arm.set_step(*step)
                arm.poison()
            torch.cuda.synchronize()
            nb, zb = orc.state(native), orc.state(decode_prologue)
            g_n.replay()
            g_z.replay()
            torch.cuda.synchronize()
            bad = orc.compare(nb, orc.state(native), zb, orc.state(decode_prologue))
            assert not bad, (label, rep, kv_base, accepted, bad)
            assert torch.equal(orc._bits(out_n), orc._bits(out_z)), (label, rep, "embedding")
            replays += 1
    return trials, replays


@pytest.mark.parametrize("mode", ["no_overlap", "no_spec_decode"])
@pytest.mark.parametrize("case", ["b1_491k", "b1_2k", "b2_mixed"])
def test_without_correction_matches_native(case, mode, weight):
    """Without the overlap correction the native prologue is one ``on_update_kv_lens()``.

    ``no_overlap``: spec decode with the overlap scheduler disabled;
    ``no_spec_decode``: the step runs with ``enable_spec_decode=False``.
    """
    past, kv_steps = CONTEXT_CASES[case]
    cm, native, decode_prologue = orc.make_pair(past, weight)
    try:
        spec = mode != "no_spec_decode"
        if mode == "no_overlap":
            for arm in (native, decode_prologue):
                arm.runner._config.disable_overlap_scheduler = True
        reason = prologue.prologue_ineligibility(
            decode_prologue.runner, decode_prologue.inputs(), spec
        )
        assert reason is None, reason

        def run_n(arm):
            return orc.run_native(arm, enable_spec_decode=spec)

        def run_z(arm):
            return orc.run_decode_prologue(arm, enable_spec_decode=spec)

        trials, replays = _eager_and_graph(
            native, decode_prologue, kv_steps, run_n, run_z, (case, mode)
        )
        print(
            f"[decode_prologue] {case} {mode}: {trials} eager trials and {replays} graph replays identical"
        )
    finally:
        cm.shutdown()


@pytest.mark.parametrize("case", ["b1_491k", "b1_2k", "b2_mixed"])
def test_without_dsl_expansion_matches_native(case, weight):
    """DeepGEMM paged-MQA logits without the DSL atom split: no expanded schedule (factor 1)."""
    past, kv_steps = CONTEXT_CASES[case]
    cm, native, decode_prologue = orc.make_pair(
        past, weight, sparse_update={"use_cute_dsl_paged_mqa_logits": False}
    )
    try:
        md = decode_prologue.md
        assert not md.expand_for_dsl and not md.use_expanded_buffers_for_mtp
        reason = prologue.prologue_ineligibility(
            decode_prologue.runner, decode_prologue.inputs(), True
        )
        assert reason is None, reason
        trials, replays = _eager_and_graph(
            native,
            decode_prologue,
            kv_steps,
            orc.run_native,
            orc.run_decode_prologue,
            (case, "no_dsl"),
            sweep=tuple(ACCEPTED_SWEEP),
        )
        print(
            f"[decode_prologue] {case} no_dsl: {trials} eager trials and {replays} graph replays identical"
        )
    finally:
        cm.shutdown()


class _HelixMapping:
    @staticmethod
    def has_cp_helix():
        return True


def test_declines_leave_native_untouched(weight):
    """Out-of-domain steps: target_prologue returns False and writes nothing."""
    cm, native, decode_prologue = orc.make_pair([32768], weight)
    try:
        md = decode_prologue.md
        inputs = decode_prologue.inputs()
        mrope = dict(inputs)
        mrope["position_ids"] = torch.zeros((3, 1, md.num_tokens), dtype=torch.int32, device="cuda")
        # (expected reason, attribute owner, attribute, value, inputs)
        cases = [
            ("expanded MTP buffers", md, "use_expanded_buffers_for_mtp", True, inputs),
            ("ragged verification", md, "enable_ragged_verification", True, inputs),
            (
                "helix context parallelism",
                decode_prologue.runner,
                "mapping",
                _HelixMapping(),
                inputs,
            ),
            ("not a generation-only step", md, "_num_contexts", 1, inputs),
            ("position ids are not a [1, N] int32 row", None, None, None, mrope),
        ]
        for expect, owner, name, value, case_inputs in cases:
            saved = getattr(owner, name) if owner is not None else None
            if owner is not None:
                setattr(owner, name, value)
            try:
                reason = prologue.prologue_ineligibility(decode_prologue.runner, case_inputs, True)
                before = orc.state(decode_prologue)
                done = prologue.target_prologue(
                    decode_prologue.runner,
                    case_inputs,
                    enable_spec_decode=True,
                    runtime_draft_len=orc.MTP_K,
                )
                torch.cuda.synchronize()
                after = orc.state(decode_prologue)
            finally:
                if owner is not None:
                    setattr(owner, name, saved)
            assert reason == expect, (expect, reason)
            assert not done, expect
            assert not orc.changed(before, after), (expect, orc.changed(before, after))
            print(f"[decode_prologue] decline '{expect}' ok")
    finally:
        cm.shutdown()


def test_take_embedding_requires_the_same_ids(weight):
    cm, native, decode_prologue = orc.make_pair([32768], weight)
    try:
        inputs = decode_prologue.inputs()
        assert prologue.target_prologue(
            decode_prologue.runner, inputs, enable_spec_decode=True, runtime_draft_len=5
        )
        other_ids = inputs["input_ids"].clone()
        assert prologue.take_embedding(decode_prologue.md, other_ids, decode_prologue.embed) is None
        # The packet is consumed either way.
        assert (
            prologue.take_embedding(decode_prologue.md, inputs["input_ids"], decode_prologue.embed)
            is None
        )
    finally:
        cm.shutdown()


_KINDS = ("KERNEL", "MEMCPY", "MEMSET", "EMPTY", "HOST", "EVENT", "MEM_ALLOC", "MEM_FREE")


def _dot_nodes(path):
    """(kind, label text) per node of a ``cudaGraphDebugDotPrint`` file, in file order."""
    text = Path(path).read_text(errors="replace")
    nodes = []
    for label in re.findall(r'"graph_\d+_node_\d+"\[[^\]]*?label="(.*?)"\];', text, flags=re.S):
        flat = re.sub(r"\\[<>]|[{}|]|\s+", " ", label).strip()
        kind = next((k for k in _KINDS if flat.startswith(k)), "OTHER")
        width = re.search(r"Width\s+(\d+)", flat)
        name = re.search(r"topoId:\s*\d+\)\s+(\S+)", flat)
        if kind == "MEMCPY" and width:
            flat = f"MEMCPY {width.group(1)} B"
        elif kind == "KERNEL" and name:
            flat = name.group(1)
        nodes.append((kind, flat[:160]))
    return nodes


def test_native_chain_census(weight, tmp_path):
    """Print the captured native prologue's nodes (memcpy sizes included) and decode_prologue's."""
    cm, native, decode_prologue = orc.make_pair([491061], weight)
    try:
        out_dir = Path(os.environ.get("GLM_MOE_DSA_DECODE_PROLOGUE_DOT_DIR", str(tmp_path)))
        out_dir.mkdir(parents=True, exist_ok=True)
        dot_n = str(out_dir / "native_prologue.dot")
        dot_z = str(out_dir / "decode_prologue.dot")
        orc.capture(native, orc.run_native, debug_dot=dot_n)
        orc.capture(decode_prologue, orc.run_decode_prologue, debug_dot=dot_z)
        for tag, path in (("native", dot_n), ("decode_prologue", dot_z)):
            nodes = _dot_nodes(path)
            kinds = [k for k, _ in nodes]
            print(
                f"[decode_prologue] census {tag}: {len(nodes)} nodes: "
                f"{kinds.count('KERNEL')} kernels, {kinds.count('MEMCPY')} memcpy, "
                f"{kinds.count('MEMSET')} memset; dot {path}"
            )
            for i, (kind, detail) in enumerate(nodes):
                print(f"[decode_prologue]   {tag} {i:2d} {kind} {detail}")
    finally:
        cm.shutdown()
