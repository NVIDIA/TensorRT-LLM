# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The flow's own leg B: no operator script stands between the witness and the hop.

The equivalence witness's leg B was a launcher each run wrote for its model,
and every change to what the witness asks of leg B (pinned rank pairs, the
state ablation) failed every launcher not rewritten with it. These tests hold
the package's launcher to that contract: it reads leg A's workload from the
witness's variables, splits the devices into a context and a generation
engine, pins each hop to its rank pair, prints what the witness parses, and
destroys the received state only where it can reach it -- refusing anywhere
else. The engines are fakes: a fake ``tensorrt_llm`` whose "state" crosses
the hop in the context-only response, and is read back on the generation
engine from a recurrent cache the ablation hook can zero. No GPU is used.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import shlex
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import witness
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import disagg_leg_b as leg_b
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment

_PROMPTS = ("The lighthouse keeper climbed the stairs", "Plants turn sunlight into sugar")


def _environ(tmp_path: Path, prompts=_PROMPTS, **extra) -> dict[str, str]:
    prompts_file = tmp_path / "prompts.json"
    prompts_file.write_text(json.dumps({"prompts": list(prompts)}), encoding="utf-8")
    return {
        leg_b.PROMPTS_FILE_ENV: str(prompts_file),
        leg_b.ARGV_ENV: '/repo/examples/llm-api/quickstart_advanced.py --model_dir="/ckpt" '
        "--max_tokens=40 --tp_size=2",
        leg_b.MAX_TOKENS_ENV: "40",
        leg_b.TIMEOUT_ENV: "900",
        **extra,
    }


# ------------------------------------------------------------- the workload


def test_the_names_it_reads_are_the_ones_the_witness_hands_over():
    assert {
        leg_b.PROMPTS_FILE_ENV: witness.LEG_B_PROMPTS_FILE_ENV,
        leg_b.MAX_TOKENS_ENV: witness.LEG_B_MAX_TOKENS_ENV,
        leg_b.MEMBER_ENV: witness.LEG_B_MEMBER_ENV,
        leg_b.ARGV_ENV: witness.LEG_B_ARGV_ENV,
        leg_b.TIMEOUT_ENV: witness.LEG_B_TIMEOUT_ENV,
        leg_b.MAX_BATCH_SIZE_ENV: witness.LEG_B_MAX_BATCH_SIZE_ENV,
        leg_b.LEGS_ENV_ENV: witness.LEG_B_LEGS_ENV_ENV,
        leg_b.RANK_PAIRS_ENV: witness.LEG_B_RANK_PAIRS_ENV,
        leg_b.ABLATE_ENV: witness.LEG_B_ABLATE_ENV,
        leg_b.ENGINE_LAUNCH_ENV: witness.LEG_B_ENGINE_LAUNCH_ENV,
        leg_b.TRANSCEIVER_ENV: witness.LEG_B_TRANSCEIVER_ENV,
        leg_b.FULL_OUTPUT_ENV: witness.WITNESS_OUTPUT_ENV,
    } == {name: name for name in (
        leg_b.PROMPTS_FILE_ENV, leg_b.MAX_TOKENS_ENV, leg_b.MEMBER_ENV, leg_b.ARGV_ENV,
        leg_b.TIMEOUT_ENV, leg_b.MAX_BATCH_SIZE_ENV, leg_b.LEGS_ENV_ENV, leg_b.RANK_PAIRS_ENV,
        leg_b.ABLATE_ENV, leg_b.ENGINE_LAUNCH_ENV, leg_b.TRANSCEIVER_ENV, leg_b.FULL_OUTPUT_ENV,
    )}  # fmt: skip
    assert leg_b.ABLATE_STATE == witness.ABLATE_STATE
    assert witness.DISAGG_LEG_B == Path(leg_b.__file__).resolve()


def test_the_workload_is_leg_as_handed_over(tmp_path):
    workload = leg_b.read_workload(
        _environ(
            tmp_path,
            **{
                leg_b.RANK_PAIRS_ENV: "0:0,1:1",
                leg_b.MAX_BATCH_SIZE_ENV: "1",
                leg_b.LEGS_ENV_ENV: json.dumps({"TARGET_REPLAY": "0"}),
                leg_b.TRANSCEIVER_ENV: json.dumps({"backend": "UCX"}),
                leg_b.FULL_OUTPUT_ENV: "1",
            },
        )
    )
    assert workload.prompts == _PROMPTS
    assert workload.quickstart == "/repo/examples/llm-api/quickstart_advanced.py"
    assert workload.argv == ("--model_dir=/ckpt", "--max_tokens=40", "--tp_size=2")
    assert workload.pairs == ((0, 0), (1, 1)) and workload.one_at_a_time
    assert workload.legs_env == {"TARGET_REPLAY": "0"}
    assert workload.transceiver == {"backend": "UCX"} and workload.full_output
    assert not workload.ablate and workload.max_tokens == 40
    assert leg_b.read_workload(_environ(tmp_path)).transceiver == leg_b.DEFAULT_TRANSCEIVER


@pytest.mark.parametrize(
    "change,message",
    [
        ({leg_b.PROMPTS_FILE_ENV: ""}, "run it through `trtllm-modeling-v2 gate witness`"),
        ({leg_b.RANK_PAIRS_ENV: "0-1"}, "is not `<ctx rank>:<gen rank>`"),
        ({leg_b.RANK_PAIRS_ENV: "0:0,1:1,0:1"}, "3 rank pair(s) for 2 prompt(s)"),
        ({leg_b.ABLATE_ENV: "kv"}, "the one ablation leg B performs is state"),
        ({leg_b.TRANSCEIVER_ENV: "[1]"}, "is not a JSON object"),
        ({leg_b.ARGV_ENV: "--tp_size=2"}, "does not start with the quickstart"),
    ],
)
def test_a_workload_it_cannot_run_is_refused_by_name(tmp_path, change, message):
    with pytest.raises(leg_b.LegBError) as caught:
        leg_b.read_workload({**_environ(tmp_path), **change})
    assert message in str(caught.value)


def test_each_engine_has_the_ranks_of_leg_as_argv():
    assert leg_b.engine_ranks(["--tp_size=2", "--pp_size=2", "--moe_ep_size=2"]) == 4
    assert leg_b.engine_ranks(["--tp_size", "4", "--enable_attention_dp"]) == 4
    assert leg_b.engine_ranks(["--model_dir=/ckpt"]) == 1


def test_the_devices_split_in_two_halves_and_too_few_is_refused():
    devices = leg_b.visible_devices({"CUDA_VISIBLE_DEVICES": "4,5,6,7"})
    assert leg_b.split_devices(devices, 2) == (["4", "5"], ["6", "7"])
    assert leg_b.visible_devices({}, count=lambda: 2) == ["0", "1"]
    with pytest.raises(leg_b.LegBError, match="needs 4 devices and sees 2"):
        leg_b.split_devices(["0", "1"], 2)


def test_only_the_generation_engine_of_an_ablation_run_gets_the_hook(tmp_path):
    workload = leg_b.read_workload(
        _environ(
            tmp_path,
            **{leg_b.ABLATE_ENV: "state", leg_b.LEGS_ENV_ENV: json.dumps({"PINNED": "1"})},
        )
    )
    base = {"PYTHONPATH": "/tree", leg_b.ABLATE_ENV: "state", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    common = dict(address=("127.0.0.1", 4000), authkey=b"k", workload=workload)
    site = tmp_path / "site"
    gen = leg_b.engine_environment(base, role="gen", devices=["2", "3"], site=site, **common)
    ctx = leg_b.engine_environment(base, role="ctx", devices=["0", "1"], site=site, **common)
    assert gen["PYTHONPATH"] == os.pathsep.join([str(site), "/tree"])
    assert ctx["PYTHONPATH"] == "/tree" and leg_b.ABLATE_ENV not in ctx
    assert (gen["CUDA_VISIBLE_DEVICES"], ctx["CUDA_VISIBLE_DEVICES"]) == ("2,3", "0,1")
    assert gen["PINNED"] == ctx["PINNED"] == "1"
    assert gen[leg_b.ADDRESS_ENV] == "127.0.0.1:4000"
    launch = "mpirun -n 2 trtllm-llmapi-launch /py"
    assert leg_b.engine_command(launch, "gen").startswith(f"{launch} ")
    assert leg_b.engine_command(launch, "gen").endswith("disagg_leg_b.py engine --role gen")


# ---------------------------------------------------------------- the hops


class _Engine:
    """A coordinator-side engine that answers from ``answer(request)`` and records every call."""

    def __init__(self, role, answer):
        self.role, self.answer, self.calls = role, answer, []

    def call(self, requests, deadline):
        self.calls.append(list(requests))
        return [self.answer(r) for r in requests]


def _ctx_answer(request):
    params = pickle.dumps({"opaque": request["index"]})
    return {
        "index": request["index"],
        "params": params,
        "prompt_token_ids": [7, request["index"]],
        "ctx_dp_rank": request["rank"],
        "token_ids": [1],
        "text": " A",
    }


def _gen_answer(request):
    return {"index": request["index"], "token_ids": [1, 2, 3], "text": " A B C"}


def test_each_prompt_crosses_from_its_context_rank_to_its_generation_rank(tmp_path):
    workload = leg_b.read_workload(_environ(tmp_path, **{leg_b.RANK_PAIRS_ENV: "0:1,1:0"}))
    ctx, gen = _Engine("ctx", _ctx_answer), _Engine("gen", _gen_answer)
    hops = leg_b.drive(ctx, gen, workload, deadline=0)
    [ctx_batch], [gen_batch] = ctx.calls, gen.calls
    assert [r["rank"] for r in ctx_batch] == [0, 1] and [r["rank"] for r in gen_batch] == [1, 0]
    assert [r["prompt"] for r in ctx_batch] == list(_PROMPTS)
    assert pickle.loads(gen_batch[1]["params"]) == {"opaque": 1}
    assert gen_batch[1]["prompt_token_ids"] == [7, 1] and gen_batch[1]["max_tokens"] == 40
    assert [(h.pair, h.ctx_rank, h.token_ids) for h in hops] == [
        ((0, 1), 0, [1, 2, 3]),
        ((1, 0), 1, [1, 2, 3]),
    ]


def test_at_batch_size_one_each_prompt_crosses_alone(tmp_path):
    workload = leg_b.read_workload(_environ(tmp_path, **{leg_b.MAX_BATCH_SIZE_ENV: "1"}))
    ctx, gen = _Engine("ctx", _ctx_answer), _Engine("gen", _gen_answer)
    leg_b.drive(ctx, gen, workload, deadline=0)
    assert [len(c) for c in ctx.calls] == [1, 1] and [len(c) for c in gen.calls] == [1, 1]


def test_a_prompt_the_context_engine_cannot_hand_over_is_a_failed_hop(tmp_path):
    workload = leg_b.read_workload(_environ(tmp_path))

    def ctx_answer(request):
        if request["index"] == 0:
            return {"index": 0, "error": "RuntimeError: no transceiver"}
        return {**_ctx_answer(request), "params": None}

    gen = _Engine("gen", _gen_answer)
    hops = leg_b.drive(_Engine("ctx", ctx_answer), gen, workload, deadline=0)
    assert not gen.calls, "nothing reached the generation engine"
    assert "the context-only request failed: RuntimeError: no transceiver" in hops[0].error
    assert "returned no disaggregated params" in hops[1].error
    assert leg_b.format_hop(hops[0], full_output=True).startswith("[0] Witness leg B failed: ")


def test_what_it_prints_is_what_the_witness_reads():
    text = " The keeper climbed slowly, counting each of the stairs as he went."
    pinned = leg_b.Hop(
        index=0, prompt="p" * 100, pair=(0, 1), ctx_rank=0, token_ids=[5, 6], text=text
    )
    plain = leg_b.Hop(index=1, prompt="q", pair=(1, 0), token_ids=[7], text=" B")
    out = leg_b.format_hop(pinned, full_output=True) + leg_b.format_hop(plain, full_output=True)
    assert witness.parse_generated_texts(out) == (repr(text[:200]), repr(" B"))
    assert witness.parse_witness_token_ids(out) == ((5, 6), (7,))
    assert witness.parse_witness_texts(out) == (text, " B")
    assert witness.parse_witness_ranks(out) == {0: (0, 1), 1: (1, 0)}
    assert witness.unpinned_hops(out, [(0, 1), (1, 0)]) == ""
    # The context engine served the hop on another rank: the witness sees it.
    moved = leg_b.format_hop(
        leg_b.Hop(index=0, prompt="p", pair=(1, 1), ctx_rank=0), full_output=False
    )
    assert "#0 reported ctx=0 gen=1 for 1:1" in witness.unpinned_hops(moved, [(1, 1)])
    assert "Witness token ids" not in leg_b.format_hop(plain, full_output=False)


# --------------------------------------------------------------- the engines


class _Future:
    def __init__(self, output=None, error=None):
        self.output, self.error = output, error

    def result(self, timeout=None):
        if self.error:
            raise self.error
        return self.output


class _Llm:
    def __init__(self):
        self.calls = []

    def generate_async(self, inputs, sampling, **kwargs):
        self.calls.append((inputs, sampling, kwargs))
        params = kwargs["disaggregated_params"]
        handed = types.SimpleNamespace(ctx_dp_rank=1, opaque=b"state")
        sequence = types.SimpleNamespace(
            token_ids=[9, 8], text=" x", disaggregated_params=handed if params.kind == "c" else None
        )
        if inputs == "boom":
            return _Future(error=RuntimeError("engine refused"))
        return _Future(types.SimpleNamespace(outputs=[sequence], prompt_token_ids=[3, 4]))


class _Params:
    def __init__(self, request_type):
        self.request_type, self.kind = request_type, "c"


_API = leg_b.Api(
    disaggregated_params=_Params,
    scheduling_params=lambda **fields: fields,
)


def test_the_context_engine_sends_context_only_requests_pinned_strictly():
    llm = _Llm()
    base = types.SimpleNamespace(max_tokens=64)
    args = types.SimpleNamespace(apply_chat_template=False)
    requests = [
        {"index": 0, "prompt": "p", "max_tokens": 40, "rank": 1},
        {"index": 1, "prompt": "boom", "max_tokens": None, "rank": None},
    ]
    replies = leg_b.serve(llm, base, args, "ctx", requests, _API)
    (inputs, sampling, kwargs), (_, other, unpinned) = llm.calls
    assert inputs == "p" and sampling.max_tokens == 40 and base.max_tokens == 64
    assert kwargs["disaggregated_params"].request_type == "context_only"
    assert kwargs["scheduling_params"] == {"attention_dp_rank": 1, "attention_dp_relax": False}
    assert "scheduling_params" not in unpinned and other.max_tokens == 64
    assert replies[0]["prompt_token_ids"] == [3, 4] and replies[0]["ctx_dp_rank"] == 1
    assert pickle.loads(replies[0]["params"]).opaque == b"state"
    assert replies[1] == {"index": 1, "error": "RuntimeError: engine refused"}


def test_the_generation_engine_takes_the_handed_params_and_the_token_ids():
    llm = _Llm()
    handed = _Params("context_only")
    request = {
        "index": 0,
        "params": pickle.dumps(handed),
        "prompt_token_ids": [3, 4],
        "max_tokens": 40,
        "rank": 0,
    }
    [reply] = leg_b.serve(llm, types.SimpleNamespace(max_tokens=8), None, "gen", [request], _API)
    [(inputs, _, kwargs)] = llm.calls
    assert inputs == {"prompt_token_ids": [3, 4]}
    assert kwargs["disaggregated_params"].request_type == "generation_only"
    assert reply == {"index": 0, "token_ids": [9, 8], "text": " x"}


# --------------------------------------------------------------- the ablation


class _Tensor:
    """Rows of numbers; ``t[i].zero_()`` zeroes row ``i`` in place, as a torch view would."""

    def __init__(self, rows):
        self.rows = [list(r) for r in rows]

    def __getitem__(self, i):
        tensor = self

        class _Row:
            def zero_(self):
                tensor.rows[i] = [0] * len(tensor.rows[i])

        return _Row()


class _Recurrent:
    def __init__(self):
        self.mamba_layer_offsets = {0: 0, 2: 1}
        self.mamba_cache_index = {11: 1}
        self.conv = {layer: _Tensor([[5], [6]]) for layer in (0, 2)}
        self.ssm = {layer: _Tensor([[7], [8]]) for layer in (0, 2)}

    def get_conv_states(self, layer):
        return self.conv[layer]

    def get_ssm_states(self, layer):
        return self.ssm[layer]


class _Paged:
    def __init__(self):
        self.layer_offsets = {1: 0, 3: 1}
        self.pools = {layer: _Tensor([[1], [2], [3]]) for layer in (1, 3)}

    def get_buffers(self, layer):
        return self.pools[layer]

    def get_batch_cache_indices(self, request_ids, layer_idx=None):
        return [[0, 2]]


def _executor(*managers):
    keys = ("KV_CACHE_MANAGER", "SPEC_RESOURCE_MANAGER")
    return types.SimpleNamespace(
        resource_manager=types.SimpleNamespace(resource_managers=dict(zip(keys, managers)))
    )


_REQUEST = types.SimpleNamespace(py_request_id=11, is_disagg_generation_transmission_complete=True)


def test_a_recurrent_model_loses_conv_and_ssm_state_at_the_requests_slot():
    cache = _Recurrent()
    ablation = leg_b.resolve_ablation(_executor(cache))
    assert isinstance(ablation, leg_b.RecurrentAblation)
    assert ablation.destroy(_REQUEST) == "the conv and SSM state of 2 recurrent layer(s)"
    for layer in (0, 2):
        assert cache.conv[layer].rows == [[5], [0]] and cache.ssm[layer].rows == [[7], [0]]


def test_a_model_with_no_recurrent_state_loses_its_kv_pages():
    cache = _Paged()
    ablation = leg_b.resolve_ablation(_executor(cache))
    assert isinstance(ablation, leg_b.PageAblation)
    assert ablation.destroy(_REQUEST) == "the KV pages of 2 attention layer(s) (2 page(s) each)"
    assert all(pool.rows == [[0], [2], [0]] for pool in cache.pools.values())


def test_a_cache_it_cannot_reach_is_refused_not_skipped():
    opaque = types.SimpleNamespace(get_buffers=lambda layer: None)
    with pytest.raises(leg_b.AblationRefused, match="leg B cannot reach the state"):
        leg_b.resolve_ablation(_executor(opaque))


def _executor_module(cache, *, method=True):
    calls = []

    class PyExecutor:
        def __init__(self):
            self.resource_manager = _executor(cache).resource_manager

    if method:

        def prepare(self, batch):
            calls.append([list(cache.conv[0].rows) for _ in batch.generation_requests])

        PyExecutor._prepare_disagg_gen_transmission_complete = prepare
    return types.SimpleNamespace(PyExecutor=PyExecutor), calls


def test_the_hook_destroys_the_state_before_the_first_decode_step(capsys):
    cache = _Recurrent()
    module, calls = _executor_module(cache)
    leg_b.patch_executor(module)
    executor = module.PyExecutor()
    waiting = types.SimpleNamespace(
        py_request_id=12, is_disagg_generation_transmission_complete=False
    )
    batch = types.SimpleNamespace(generation_requests=[_REQUEST, waiting])
    executor._prepare_disagg_gen_transmission_complete(batch)
    assert calls == [[[[5], [0]], [[5], [0]]]], "zeroed before the executor took the request up"
    assert capsys.readouterr().out == (
        "Witness ablated: the conv and SSM state of 2 recurrent layer(s)\n"
    )


@pytest.mark.parametrize("method", [True, False])
def test_an_engine_whose_state_cannot_be_destroyed_does_not_come_up(capsys, method):
    cache = _Recurrent() if not method else types.SimpleNamespace()
    module, _ = _executor_module(cache, method=method)
    leg_b.patch_executor(module)
    with pytest.raises(leg_b.AblationRefused):
        module.PyExecutor()
    assert capsys.readouterr().out.startswith(f"{leg_b.REFUSED} ")


# ------------------------------------------------- the generation engine's hook

_EXECUTOR_MODULE = """
import types


class PyExecutor:
    def __init__(self, cache):
        self.resource_manager = types.SimpleNamespace(resource_managers={"KV_CACHE_MANAGER": cache})

    def _prepare_disagg_gen_transmission_complete(self, batch):
        return "taken up"
"""


def _site(tmp_path: Path) -> Path:
    return leg_b.write_site(tmp_path / "site")


def test_the_hook_reaches_the_executor_in_a_fresh_process_and_the_next_sitecustomize_runs(tmp_path):
    package = tmp_path / "pkgs"
    module = package / "tensorrt_llm" / "_torch" / "pyexecutor" / "py_executor.py"
    module.parent.mkdir(parents=True)
    for parent in (module.parent, module.parent.parent, module.parent.parent.parent):
        (parent / "__init__.py").write_text("", encoding="utf-8")
    module.write_text(_EXECUTOR_MODULE, encoding="utf-8")
    # A session hook beside it, which chains back to the first sitecustomize it
    # finds that is not itself -- leg B's -- as the run's resolve hook does.
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            import os, sys
            sys._other_hook_runs = getattr(sys, "_other_hook_runs", 0) + 1
            here = os.path.dirname(os.path.abspath(__file__))
            for entry in sys.path:
                candidate = os.path.join(os.path.abspath(entry or "."), "sitecustomize.py")
                if os.path.dirname(candidate) != here and os.path.isfile(candidate):
                    exec(compile(open(candidate).read(), candidate, "exec"),
                         {"__name__": "sitecustomize", "__file__": candidate})
                    break
            """
        ),
        encoding="utf-8",
    )
    probe = textwrap.dedent(
        """
        import sys, types
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

        class Rows:
            def __init__(self, rows):
                self.rows = rows

            def __getitem__(self, i):
                rows = self.rows
                return types.SimpleNamespace(zero_=lambda: rows.__setitem__(i, [0]))

        cache = types.SimpleNamespace(
            mamba_layer_offsets={0: 0, 2: 1},
            mamba_cache_index={11: 1},
            conv={0: Rows([[5], [6]]), 2: Rows([[5], [6]])},
            ssm={0: Rows([[7], [8]]), 2: Rows([[7], [8]])},
        )
        cache.get_conv_states = lambda layer: cache.conv[layer]
        cache.get_ssm_states = lambda layer: cache.ssm[layer]
        executor = PyExecutor(cache)
        request = types.SimpleNamespace(py_request_id=11,
                                        is_disagg_generation_transmission_complete=True)
        print(executor._prepare_disagg_gen_transmission_complete(
            types.SimpleNamespace(generation_requests=[request])))
        print("other hook ran", sys._other_hook_runs, "conv", cache.conv[2].rows)
        """
    )

    def run(ablate: str) -> str:
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(_site(tmp_path)), str(other), str(package)]),
            leg_b.ABLATE_ENV: ablate,
        }
        done = subprocess.run(
            [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stderr
        return done.stdout

    ablated = run("state")
    assert "Witness ablated: the conv and SSM state of 2 recurrent layer(s)\ntaken up\n" in ablated
    assert "other hook ran 1 conv [[5], [0]]" in ablated
    untouched = run("")
    assert "Witness ablated" not in untouched
    assert "other hook ran 1 conv [[5], [6]]" in untouched


# ------------------------------------------------------- leg B, end to end

_FAKE_MODEL = """
def _ids(text):
    return [100 + ord(c) % 50 for c in text]


def _state(ids):
    # What prefill leaves behind: a recurrent state the whole prompt shaped.
    return 1 + sum(i * (n + 1) for n, i in enumerate(ids)) % 89


def _decode(state, count):
    return [1000 + (state * 7 + k * 13) % 97 for k in range(count)]
"""

_FAKE_LLMAPI = """
import os
import pickle
import types
from dataclasses import dataclass

from tensorrt_llm._fake_model import _decode, _ids, _state
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


@dataclass
class SamplingParams:
    max_tokens: int = 16


@dataclass
class DisaggregatedParams:
    request_type: str = None
    ctx_request_id: int = None
    opaque_state: bytes = None
    first_gen_tokens: list = None
    ctx_dp_rank: int = None


@dataclass
class SchedulingParams:
    attention_dp_rank: int = None
    attention_dp_relax: bool = True


class CacheTransceiverConfig:
    def __init__(self, **fields):
        self.fields = fields


class _Done:
    def __init__(self, output):
        self.output = output

    def result(self, timeout=None):
        return self.output

    def __iter__(self):
        yield self.output


class LLM:
    def __init__(self, model=None, **kwargs):
        self.kwargs = kwargs
        self.args = types.SimpleNamespace()
        transceiver = kwargs.get("cache_transceiver_config")
        if transceiver is not None:
            print(f"fake engine: cache transceiver {sorted(transceiver.fields.items())}, "
                  f"overlap {'off' if kwargs.get('disable_overlap_scheduler') else 'on'}",
                  flush=True)
        self.executor = PyExecutor()

    def generate_async(self, inputs, sampling_params, disaggregated_params=None,
                       scheduling_params=None, streaming=False):
        ids = list(inputs["prompt_token_ids"]) if isinstance(inputs, dict) else _ids(inputs)
        kind = disaggregated_params.request_type if disaggregated_params else None
        rank = scheduling_params.attention_dp_rank if scheduling_params else 0
        handed = None
        if kind == "context_only":
            state = _state(ids)
            tokens = _decode(state, 1)
            handed = DisaggregatedParams(request_type="context_only", ctx_request_id=7,
                                         opaque_state=pickle.dumps(state),
                                         first_gen_tokens=tokens, ctx_dp_rank=rank)
        elif kind == "generation_only":
            received = pickle.loads(disaggregated_params.opaque_state)
            if os.environ.get("FAKE_HOP_DROPS_STATE") == "1":
                received = 0
            tokens = _decode(self.executor.receive(received), sampling_params.max_tokens)
        else:
            tokens = _decode(_state(ids), sampling_params.max_tokens)
        sequence = types.SimpleNamespace(token_ids=tokens, text="".join(f" w{t}" for t in tokens),
                                         disaggregated_params=handed)
        return _Done(types.SimpleNamespace(outputs=[sequence], prompt_token_ids=ids))

    def shutdown(self):
        pass
"""

_FAKE_EXECUTOR = '''
import types


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, i):
        rows = self.rows

        class _Row:
            def zero_(self):
                rows[i] = 0

        return _Row()


class _Cache:
    def __init__(self):
        self.mamba_layer_offsets = {0: 0, 1: 1}
        self.conv = {layer: _Rows([0] * 4) for layer in self.mamba_layer_offsets}
        self.ssm = {layer: _Rows([0] * 4) for layer in self.mamba_layer_offsets}
        self.mamba_cache_index = {}

    def get_conv_states(self, layer):
        return self.conv[layer]

    def get_ssm_states(self, layer):
        return self.ssm[layer]


class PyExecutor:
    def __init__(self):
        self.cache = _Cache()
        self.resource_manager = types.SimpleNamespace(
            resource_managers={"KV_CACHE_MANAGER": self.cache})
        self.served = 0

    def _prepare_disagg_gen_transmission_complete(self, batch):
        return None

    def receive(self, state):
        """The transfer lands in the request's slot; decoding reads it back from there."""
        request_id, self.served = self.served, self.served + 1
        slot = request_id % 4
        self.cache.mamba_cache_index[request_id] = slot
        for layer in self.cache.mamba_layer_offsets:
            self.cache.conv[layer].rows[slot] = state
            self.cache.ssm[layer].rows[slot] = state
        request = types.SimpleNamespace(py_request_id=request_id,
                                        is_disagg_generation_transmission_complete=True)
        self._prepare_disagg_gen_transmission_complete(
            types.SimpleNamespace(generation_requests=[request]))
        return min(self.cache.conv[0].rows[slot], self.cache.ssm[1].rows[slot])
'''

_FAKE_QUICKSTART = """
import argparse

from tensorrt_llm.llmapi import LLM, SamplingParams


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir")
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--disable_overlap_scheduler", action="store_true")
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--prompt", nargs="+", default=["hello"])
    args, _ = parser.parse_known_args()
    return args


def setup_llm(args, **kwargs):
    llm = LLM(model=args.model_dir, disable_overlap_scheduler=args.disable_overlap_scheduler,
              **kwargs)
    return llm, SamplingParams(max_tokens=args.max_tokens)


def main():
    args = parse_arguments()
    llm, sampling = setup_llm(args)
    for i, prompt in enumerate(args.prompt):
        for output in llm.generate_async(prompt, sampling, streaming=True):
            pass
        text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt[:80]!r}..., Generated text: {text[:200]!r}...")
"""


def _fake_tree(tmp_path: Path) -> tuple[Path, Path]:
    site = tmp_path / "site-packages"
    files = {
        "tensorrt_llm/__init__.py": "",
        "tensorrt_llm/_fake_model.py": _FAKE_MODEL,
        "tensorrt_llm/_torch/__init__.py": "",
        "tensorrt_llm/_torch/pyexecutor/__init__.py": "",
        "tensorrt_llm/_torch/pyexecutor/py_executor.py": _FAKE_EXECUTOR,
        "tensorrt_llm/llmapi/__init__.py": _FAKE_LLMAPI,
    }
    for name, text in files.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    repo = tmp_path / "repo"
    quickstart = repo / "examples" / "llm-api" / "quickstart_advanced.py"
    quickstart.parent.mkdir(parents=True)
    quickstart.write_text(_FAKE_QUICKSTART, encoding="utf-8")
    return site, repo


def _coordinate(tmp_path: Path, **extra) -> tuple[int, str]:
    site, repo = _fake_tree(tmp_path)
    environ = {
        **os.environ,
        **_environ(tmp_path),
        leg_b.ARGV_ENV: f"{repo}/examples/llm-api/quickstart_advanced.py --model_dir=/ckpt "
        "--max_tokens=12",
        leg_b.MAX_TOKENS_ENV: "12",
        leg_b.ENGINE_LAUNCH_ENV: shlex.quote(sys.executable),
        leg_b.FULL_OUTPUT_ENV: "1",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "PYTHONPATH": str(site),
        **extra,
    }
    done = subprocess.run(
        [sys.executable, str(leg_b.SCRIPT)],
        env=environ,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    return done.returncode, done.stdout + done.stderr


def _aggregated(prompts=_PROMPTS, tokens=12) -> tuple[tuple[int, ...], ...]:
    """What leg A's one engine generates for ``prompts`` under the fake model."""
    model: dict = {}
    exec(compile(_FAKE_MODEL, "fake_model", "exec"), model)
    return tuple(
        tuple(model["_decode"](model["_state"](model["_ids"](p)), tokens)) for p in prompts
    )


def test_leg_b_hands_each_prompt_over_and_matches_one_engine(tmp_path):
    code, out = _coordinate(tmp_path)
    assert code == 0, out
    assert witness.parse_witness_token_ids(out) == _aggregated()
    assert "context engine on devices 0, generation engine on 1, 1 rank(s) each" in out
    transceiver = "cache transceiver [('backend', 'DEFAULT'), ('transceiver_runtime', 'auto')]"
    assert f"{transceiver}, overlap off" in out and f"{transceiver}, overlap on" in out
    assert len(witness.parse_generated_texts(out)) == 2 and "Witness ablated" not in out


def test_leg_b_shows_a_hop_that_loses_the_state(tmp_path):
    code, out = _coordinate(tmp_path, FAKE_HOP_DROPS_STATE="1")
    assert code == 0, out
    lost = witness.parse_witness_token_ids(out)
    assert all(a != b for a, b in zip(lost, _aggregated())), "a lost state moves every continuation"


def test_under_the_ablation_the_received_state_is_destroyed_before_decoding(tmp_path):
    code, out = _coordinate(tmp_path, **{leg_b.ABLATE_ENV: "state"})
    assert code == 0, out
    assert witness.parse_witness_ablations(out) == (
        "the conv and SSM state of 2 recurrent layer(s)",
    )
    ablated = witness.parse_witness_token_ids(out)
    assert all(a != b for a, b in zip(ablated, _aggregated()))


def test_an_engine_that_dies_before_it_is_up_is_named(tmp_path):
    code, out = _coordinate(
        tmp_path,
        **{leg_b.ENGINE_LAUNCH_ENV: f"{shlex.quote(sys.executable)} -c 'raise SystemExit(3)'"},
    )
    assert code == 2
    assert "engine exited 3 before it was up" in out


def test_too_few_devices_is_refused_before_any_engine_starts(tmp_path):
    started = []
    environ = {**_environ(tmp_path), "CUDA_VISIBLE_DEVICES": "0,1,2"}
    out = io.StringIO()
    code = leg_b.coordinate(environ, out=out, start=lambda *a: started.append(a))
    assert code == 2 and not started
    assert "needs 4 devices and sees 3" in out.getvalue()


# ------------------------------------------------------ the witness runs it


_TASK_BLOCK = {"prompts": list(_PROMPTS), "max_tokens": 12, "min_generated_tokens": 8}


def _task(repo: Path, tmp_path: Path, block: dict) -> dict:
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir(exist_ok=True)
    return {
        "trtllm_repo_path": str(repo),
        "checkpoint_path": str(checkpoint),
        "parallel": "tp1",
        "features": ["disagg"],
        "runtime": {"extra_flags": []},
        "accuracy": {"witness": {"disagg": block}},
    }


def _witness(tmp_path: Path, block: dict) -> witness.GateResult:
    site, repo = _fake_tree(tmp_path)
    environment = RunEnvironment(
        kind="local",
        python=sys.executable,
        world_size=1,
        env={"PYTHONPATH": str(site), "CUDA_VISIBLE_DEVICES": "0,1"},
    )
    return witness.run_declared_witness(
        environment,
        unit="disagg",
        member="tp1",
        task_data=_task(repo, tmp_path, block),
        workdir=tmp_path / "witness",
    )


def test_a_witness_with_no_launcher_of_its_own_passes_on_the_flows_leg_b(tmp_path):
    result = _witness(tmp_path, dict(_TASK_BLOCK))
    assert result.passed, result.detail
    assert "2 continuation(s) token-identical across the hop" in result.detail
    assert "with the received state destroyed (the conv and SSM state of 2 recurrent" in (
        result.detail
    )


def test_the_flows_leg_b_fails_a_hop_that_drops_the_state(tmp_path):
    block = {**_TASK_BLOCK, "contrast_env": {"FAKE_HOP_DROPS_STATE": "1"}}
    result = _witness(tmp_path, block)
    assert not result.passed
    assert "differ across the hop" in result.detail and "first differs at token 0" in result.detail
