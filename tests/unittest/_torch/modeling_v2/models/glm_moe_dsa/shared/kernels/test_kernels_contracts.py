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
"""Contract table of the target's kernels (``kernels/contracts.py``).

The module is loaded by file path so the test needs neither torch nor the
compiled extensions; kernel sources are read as text.
"""

import importlib.util
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[8]
_TARGET = ("tensorrt_llm", "_torch", "_experimental", "modeling_v2", "models", "glm_moe_dsa")
_KERNELS = (*_TARGET, "shared", "kernels")
_TESTS = ("tests", "unittest", "_torch", "modeling_v2", "models", "glm_moe_dsa")
_KERNEL_TESTS = (*_TESTS, "shared", "kernels")


def _load(name):
    path = _ROOT.joinpath(*_KERNELS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"glm_moe_dsa_kernels_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_table():
    contracts = _load("contracts")
    owners = {k: c.owner for k, c in contracts.CONTRACTS.items()}
    assert owners == {
        "query_front": "absorbed_query",
        "index_selection": "indexer_topk",
        "sparse_attention": "sparse_mla_decode",
        "attention_output_tail": "attention_output_tail",
        "router": "router_gemm",
        "routed_moe": "routed_moe",
        "post_moe_hop": "post_moe_hop",
        "draft_front": "draft_front",
    }
    # Rules are optional, so a four-field contract is still valid.
    plain = contracts.Contract("x", "absorbed_query", ("a",), ("b",))
    assert plain.rules == ()


def test_attention_tail_rules():
    contracts = _load("contracts")
    tail = contracts.CONTRACTS["attention_output_tail"]
    assert tail.outputs == ("partial[R,6144]",)
    assert tail.rules == (
        contracts.ATTENTION_TAIL_COMMUNICATION_RULE,
        contracts.ATTENTION_TAIL_PDL_RULE,
    )
    comm = contracts.ATTENTION_TAIL_COMMUNICATION_RULE
    assert "No communication is hidden inside attention_output_tail." in comm
    assert "router_gemm/native all-reduce consumes the identical partial." in comm
    pdl = contracts.ATTENTION_TAIL_PDL_RULE
    for phrase in (
        "attention_output_tail may be a PDL dependent of the latent's producer.",
        "Before its griddepcontrol.wait a attention_output_tail kernel reads only immutable weights",
        "writes only its own shared memory",
        "no async-proxy global writes",
        "Every latent read and every global write (v scratch, partial) follows the wait.",
        "executes griddepcontrol.launch_dependents before it writes the partial",
        "must execute griddepcontrol.wait before reading the partial",
        "may be reused after attention_output_tail returns",
    ):
        assert phrase in pdl, phrase
    switches = re.search(r"Kill switches: (.*)\.$", pdl).group(1)
    assert switches == (
        "TRTLLM_ENABLE_PDL=0 (all, entry included), TailConfig(trigger_dependents=False) (exit)"
    )


def _source(*parts):
    path = _ROOT.joinpath(*parts)
    if not path.is_file():
        pytest.skip(f"{path} is not in this tree")
    return path.read_text()


def test_attention_tail_names_exist_in_the_tree():
    """The kill switches and the into-hop protocol header the rules name exist."""
    kernels = _source(*_KERNELS, "attention_output_tail", "kernels.py")
    assert re.search(
        r"def pdl_entry_enabled\(\) -> bool:\n(?:.*\n)*?    return pdl_enabled\(\)$", kernels, re.M
    )
    assert 'os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"' in kernels
    assert re.search(r"^\s+trigger_dependents: bool = True$", kernels, re.M)
    rule = _load("contracts").ATTENTION_TAIL_COMMUNICATION_RULE
    header = re.search(r"(cpp/\S+\.h)", rule).group(1)
    assert "griddepcontrol.wait" in _source(*header.split("/"))


def test_routed_moe_and_post_moe_hop_rules():
    """The routed-input and deferred-norm seams and the token-major routed-MoE form."""
    contracts = _load("contracts")
    c = contracts.CONTRACTS
    assert c["routed_moe"].rules == (
        contracts.ROUTED_MOE_TOKEN_MAJOR_RULE,
        contracts.POST_MOE_HOP_ROUTED_INPUT_RULE,
    )
    assert c["post_moe_hop"].rules == (
        contracts.POST_MOE_HOP_ROUTED_INPUT_RULE,
        contracts.POST_MOE_HOP_DEFERRED_NORM_RULE,
        contracts.POST_MOE_HOP_SHARED_PROTOCOL_RULE,
    )
    assert c["query_front"].rules == (contracts.POST_MOE_HOP_DEFERRED_NORM_RULE,)
    for phrase in ("identity t*8+k", "reads fc2_output only through the permutation"):
        assert phrase in contracts.ROUTED_MOE_TOKEN_MAJOR_RULE, phrase
    for phrase in (
        "parsed only by dispatch.feature_variants",
        "agreed by every rank",
        "zero when routed_moe's kernel starts",
        "the hop clears after reading them",
        "routed_moe never reuses a buffer it wrote into",
        "only after its own griddepcontrol.wait",
    ):
        assert phrase in contracts.POST_MOE_HOP_ROUTED_INPUT_RULE, phrase
    for phrase in (
        "The packet is not a tensor",
        "materialize()",
        "reads only immutable weights before its griddepcontrol.wait",
    ):
        assert phrase in contracts.POST_MOE_HOP_DEFERRED_NORM_RULE, phrase
    assert "must change both in the same commit" in contracts.POST_MOE_HOP_SHARED_PROTOCOL_RULE


def test_routed_moe_and_post_moe_hop_names_exist_in_the_tree():
    """The names the routed-input and deferred-norm rules cite exist where the rules say."""
    hop = _source(*_KERNELS, "post_moe_hop", "hop.py")
    assert "def enable_routed_input(" in hop
    assert 'ROUTED_MODES = ("row", "identity", "fp32")' in hop
    assert 'dispatch.feature_variants(("routed_moe_into_hop",))["routed_moe_into_hop"]' in hop
    packet = _source(*_KERNELS, "deferred_norm", "packet.py")
    assert "class DeferredNormHidden:" in packet and "def materialize(self)" in packet
    ptx = _source(*_KERNELS, "deferred_norm", "ptx.py")
    assert "from ..post_moe_hop.kernels import (" in ptx
    hop_kernel = _source(*_KERNELS, "deferred_norm", "hop_kernel.py")
    assert "griddepcontrol_launch_dependents" in hop_kernel
    assert "trigger_early" in hop_kernel
    for name in ("test_deferred_norm_hop_mpi.py", "test_deferred_norm_routed_mpi.py"):
        assert "mixed_with_post_moe_hop" in _source(*_KERNEL_TESTS, "deferred_norm", name)
