# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two-leg gates for features that are output-identical by construction.

Written against the real tools' output formats, not against invented strings:
a parser tested only against strings its author invented passes its tests
while matching nothing the tools actually print.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.checks.witness import (
    ABLATE_STATE,
    DEVICE_MEMORY_MARKER,
    LEG_B_ABLATE_ENV,
    LEG_B_ARGV_ENV,
    LEG_B_DRAFT_LEN_ENV,
    LEG_B_ENGINE_LAUNCH_ENV,
    LEG_B_LEGS_ENV_ENV,
    LEG_B_MAX_BATCH_SIZE_ENV,
    LEG_B_MAX_TOKENS_ENV,
    LEG_B_MEMBER_ENV,
    LEG_B_PROMPTS_FILE_ENV,
    LEG_B_TIMEOUT_ENV,
    PREFIX_OBSERVABLE,
    WITNESS_COUNTER_ENV,
    WITNESS_COUNTER_MARKER,
    WITNESS_OUTPUT_ENV,
    compare_continuations,
    engine_launch,
    filler_words,
    first_difference,
    natural_text,
    parse_device_memory_mib,
    parse_generated_texts,
    parse_witness_counters,
    parse_witness_texts,
    parse_witness_token_ids,
    run_equivalence_witness,
    run_kv_quant_witness,
    run_replay_witness,
    run_residency_witness,
    with_max_batch_size,
    witness_command,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult

_TASK = {
    "trtllm_repo_path": "/repo",
    "checkpoint_path": "/ckpt",
    "parallel": "tp1",
    "features": [],
    "runtime": {"cuda_graph": True, "kv_cache_reuse": True, "extra_flags": []},
}


def _env():
    return RunEnvironment(kind="local", python="/opt/venv/bin/python", world_size=1)


def _ids(text: str) -> list[int]:
    """Token ids for ``text``: one per character, so ids differ where the text does."""
    return [ord(c) for c in text]


def _witness_lines(i: int, text: str) -> str:
    """What a quickstart that prints whole continuations adds under TRTLLM_WITNESS_FULL_OUTPUT=1."""
    return f"[{i}] Witness token ids: {_ids(text)}\n[{i}] Witness text: {text!r}\n"


#: The line format verbatim from `quickstart_advanced.py`.
#: ``ids=False`` is a tree whose quickstart prints no whole continuations.
def _quickstart(text_a: str, text_b: str, mib: float | None = None, *, ids: bool = True) -> str:
    out = ""
    for i, (prompt, text) in enumerate(
        (("The capital of France is", text_a), ("In one sentence, photosynthesis is", text_b))
    ):
        out += f"[{i}] Accept rate: 1.00 (tokens=32, iterations=32)\n"
        out += f"[{i}] Prompt: {prompt!r}..., Generated text: {text!r}...\n"
        if ids:
            out += _witness_lines(i, text)
    if mib is not None:
        out += f"{DEVICE_MEMORY_MARKER}: {mib}\n"
    return out


# ------------------------------------------------------------------- parsing


def test_continuations_are_read_in_order():
    texts = parse_generated_texts(_quickstart(" Paris.", " the process"))
    assert texts == ("' Paris.'", "' the process'")


def test_no_continuations_is_empty_not_an_error():
    assert parse_generated_texts("nothing here") == ()


def test_device_memory_is_read_from_a_marker_not_from_nvidia_smis_format():
    """The target decides WHEN to sample, because the gate cannot know.

    The number that matters is taken after the weights load and before the
    KV cache pool is allocated, and a gate sampling from outside has no way
    to tell when that was.
    """
    assert parse_device_memory_mib(f"{DEVICE_MEMORY_MARKER}: 183542\n") == 183542.0
    assert parse_device_memory_mib(f"{DEVICE_MEMORY_MARKER}=1024.5") == 1024.5
    assert parse_device_memory_mib("no marker") is None


def test_the_last_reading_wins():
    """A two-phase engine start prints one per phase; the real one is last."""
    both = f"{DEVICE_MEMORY_MARKER}: 100\n...\n{DEVICE_MEMORY_MARKER}: 200\n"
    assert parse_device_memory_mib(both) == 200.0


# ------------------------------------------------------------------- command


def test_the_witness_runs_quickstart_not_the_eval_harness():
    """`trtllm-eval` reports neither device memory nor the continuations.

    It also takes about twice as long a leg.
    """
    argv = witness_command(_TASK, _env(), unit="core")
    assert "quickstart_advanced.py" in argv
    assert "trtllm-eval" not in argv


def test_the_witness_carries_the_units_runtime_block():
    task = dict(_TASK) | {
        "runtime": {"default": {"kv_cache_reuse": True}, "mm": {"kv_cache_reuse": False}}
    }
    assert "--disable_kv_cache_reuse" in witness_command(task, _env(), unit="mm")
    assert "--disable_kv_cache_reuse" not in witness_command(task, _env(), unit="core")


# ----------------------------------------------------------------- residency


def _residency(
    on_mib, off_mib, on_text=(" A", " B"), off_text=(" A", " B"), expected=100.0, ids=True
):
    calls: list[str] = []

    def runner(command, env=None, timeout=0):
        calls.append(command)
        first = len(calls) == 1
        mib, text = (on_mib, on_text) if first else (off_mib, off_text)
        return CommandResult(returncode=0, stdout=_quickstart(text[0], text[1], mib, ids=ids))

    return run_residency_witness(
        _env(),
        task_data=_TASK,
        expected_mib=expected,
        contrast_env={"TARGET_OFFLOAD": "0"},
        runner=runner,
    ), calls


def test_a_real_offload_shows_the_predicted_memory_delta():
    result, calls = _residency(on_mib=1000, off_mib=1200, expected=100.0)
    assert result.passed
    assert result.measured == 200.0
    assert len(calls) == 2


def test_an_offload_that_did_nothing_fails_even_though_the_text_is_right():
    """The failure this gate exists for.

    A target that quietly loads the table onto the device anyway produces
    identical output and passes boot and accuracy. Only the memory delta
    distinguishes it.
    """
    result, _ = _residency(on_mib=1200, off_mib=1205, expected=100.0)
    assert not result.passed
    assert "delta 5 MiB < predicted 100 MiB" in result.detail


def test_legs_that_disagree_on_text_fail_regardless_of_the_memory():
    """Offloading is output-identical by construction.

    So a disagreement means the gather is wrong, and the accuracy tolerance
    would hide it.
    """
    result, _ = _residency(on_mib=1000, off_mib=1200, on_text=(" A", " B"), off_text=(" A", " C"))
    assert not result.passed
    assert "identical text" in result.detail


def test_a_missing_memory_reading_is_not_a_pass():
    result, _ = _residency(on_mib=None, off_mib=1200)
    assert not result.passed
    assert "is not a pass" in result.detail
    # ...and says who prints it, and when: only the target knows the moment.
    assert "before the KV cache pool is allocated" in result.detail


def test_legs_that_print_no_token_ids_cannot_pass_however_good_the_memory_delta():
    """A tree whose quickstart prints only the 200-character line.

    The two legs agree on that prefix and the delta is right, but agreement
    on a prefix is not token identity, and a 2-token answer would agree
    there just as well.
    """
    result, _ = _residency(on_mib=1000, off_mib=1200, ids=False)
    assert not result.passed
    assert "delta 200 MiB >= predicted 100 MiB" in result.detail
    assert f"agree as {PREFIX_OBSERVABLE}, which is not token identity" in result.detail
    assert "offloaded and resident leg(s)" in result.detail
    # The verdict says what to print, under which variable, and from where.
    assert f"{WITNESS_OUTPUT_ENV}=1" in result.detail
    assert "`[i] Witness token ids: [...]`" in result.detail
    assert "examples/llm-api/quickstart_advanced.py" in result.detail


def test_the_control_leg_runs_with_the_offload_switched_off():
    """The changed variable has to be the feature, and only the feature."""
    seen: list[dict] = []

    def runner(command, env=None, timeout=0):
        seen.append({"cmd": command})
        return CommandResult(returncode=0, stdout=_quickstart(" A", " B", 1000 + 200 * len(seen)))

    run_residency_witness(
        _env(),
        task_data=_TASK,
        expected_mib=1.0,
        contrast_env={"TARGET_OFFLOAD": "0"},
        runner=runner,
    )
    assert "TARGET_OFFLOAD=0" not in seen[0]["cmd"]
    assert "TARGET_OFFLOAD=0" in seen[1]["cmd"]


# --------------------------------------------------------------- equivalence

#: In a command: it is leg B's self-test run.
_ABLATE = f"{LEG_B_ABLATE_ENV}={ABLATE_STATE}"


def _ablated(text: str = " lost the thread", what: str = "conv, ssm") -> str:
    """Leg B's self-test run: its one continuation, and the line saying what it destroyed."""
    return _whole([text]) + f"[gen] disagg_leg_b[gen]: Witness ablated: {what}\n"


def _equivalence(agg_text, dis_text, dis_ok=True):
    calls: list[str] = []

    def runner(command, env=None, timeout=0):
        calls.append(command)
        if _ABLATE in command:
            return CommandResult(returncode=0, stdout=_ablated())
        first = len(calls) == 1
        if not first and not dis_ok:
            return CommandResult(returncode=1, stdout="", stderr="hang")
        text = agg_text if first else dis_text
        return CommandResult(returncode=0, stdout=_quickstart(*text))

    return run_equivalence_witness(
        _env(),
        task_data=_TASK,
        contrast_env={"TRTLLM_DISAGG": "1"},
        runner=runner,
    ), calls


def test_token_identical_output_across_the_hop_passes():
    result, calls = _equivalence((" Paris.", " the process"), (" Paris.", " the process"))
    assert result.passed
    assert "token-identical" in result.detail
    # Both legs, then leg B's self-test run with the received state destroyed.
    assert len(calls) == 3 and _ABLATE in calls[2]
    assert "with the received state destroyed (conv, ssm)" in result.detail


def test_fluent_but_different_output_fails():
    """Not 'similar' — identical.

    Missing recurrent state is recomputed from zero and the model recovers
    into text that is fluent and wrong, which a short benchmark under a
    five-point tolerance does not catch.
    """
    result, _ = _equivalence((" Paris.", " the process"), (" Paris.", " a process"))
    assert not result.passed
    assert "fluent, wrong text" in result.detail


def test_a_disagg_leg_that_fails_names_the_usual_causes():
    """A disagg launch fails by hanging more often than by raising."""
    result, _ = _equivalence((" A", " B"), (" A", " B"), dis_ok=False)
    assert not result.passed
    assert "cache transceiver" in result.detail


def test_the_second_leg_runs_disaggregated_and_the_first_does_not():
    result, calls = _equivalence((" A", " B"), (" A", " B"))
    assert "TRTLLM_DISAGG=1" not in calls[0]
    assert "TRTLLM_DISAGG=1" in calls[1]


def test_the_disagg_leg_can_be_a_different_command_entirely():
    """Disaggregation is not an environment variable over one process.

    It is a context server, a generation server and a client. An env overlay
    alone would run leg B **aggregated**, and the witness would compare a run
    to itself and pass trivially — the worst possible outcome for a gate whose
    entire job is to catch a silent equality.
    """
    calls: list[str] = []

    def runner(command, env=None, timeout=0):
        calls.append(command)
        if _ABLATE in command:
            return CommandResult(returncode=0, stdout=_ablated())
        return CommandResult(returncode=0, stdout=_quickstart(" Paris.", " the process"))

    result = run_equivalence_witness(
        _env(),
        task_data=_TASK,
        contrast_env={"TRTLLM_DISAGG": "1"},
        contrast_command="/run/launch_disagg.sh --prompts boot",
        runner=runner,
    )
    assert result.passed
    assert "quickstart_advanced.py" in calls[0]
    assert "launch_disagg.sh" in calls[1]
    # Leg A's argv is HANDED to leg B in its environment; it is not what runs.
    assert "quickstart_advanced.py" not in calls[1].split("bash -c", 1)[1]


def test_without_a_contrast_command_both_legs_are_the_same_argv():
    """Recorded so the trivial-pass hazard above is visible in the tests.

    This is the correct shape for the residency witness, where the feature
    really is one switch, and the wrong one for disagg.
    """
    calls: list[str] = []

    def runner(command, env=None, timeout=0):
        calls.append(command)
        return CommandResult(returncode=0, stdout=_quickstart(" A", " B"))

    run_equivalence_witness(_env(), task_data=_TASK, contrast_env={"X": "1"}, runner=runner)
    assert calls[0].split("python")[-1] in calls[1]


# ------------------------------------------- equivalence: leg B as a launcher


def _whole(texts, ids=None) -> str:
    """Quickstart's output under TRTLLM_WITNESS_FULL_OUTPUT=1 (see the example script).

    ``ids`` are the token ids printed per continuation, one per character of
    its text when not given; ``False`` prints the whole text without them.
    """
    out = ""
    for i, text in enumerate(texts):
        out += f"[{i}] Prompt: 'p'..., Generated text: {text[:200]!r}...\n"
        if ids is not False:
            out += f"[{i}] Witness token ids: {list(_ids(text) if ids is None else ids[i])}\n"
        out += f"[{i}] Witness text: {text!r}\n"
    return out


def _run_equivalence(tmp_path, agg_out, dis_out, **kwargs):
    """Both legs, and leg B's self-test run, whose continuation moves."""
    calls: list[str] = []

    def runner(command, env=None, timeout=0):
        calls.append(command)
        if _ABLATE in command:
            return CommandResult(returncode=0, stdout=_ablated())
        return CommandResult(returncode=0, stdout=agg_out if len(calls) == 1 else dis_out)

    kwargs.setdefault("contrast_command", "/onboard/scripts/disagg_leg_b.py")
    kwargs.setdefault("workdir", tmp_path / "witness")
    kwargs.setdefault("task_data", _TASK)
    result = run_equivalence_witness(
        kwargs.pop("environment", _env()),
        contrast_env={"TRTLLM_DISAGG": "1"},
        runner=runner,
        **kwargs,
    )
    return result, calls


def _assignment(command: str, name: str) -> str:
    import shlex

    for word in shlex.split(command):
        if word.startswith(f"{name}="):
            return word.split("=", 1)[1]
    raise AssertionError(f"{name} not exported in {command}")


def test_the_contrast_command_runs_as_one_process_not_as_a_rank(tmp_path):
    """Leg B starts engines; it is not one.

    As engine work it would go out behind `mpirun -n <world>
    trtllm-llmapi-launch`: idle workers would hold the other ranks, and the
    command would inherit the wrapper's spawn-proxy address, which makes the
    engines it starts dial the outer proxy and hang.
    """
    four = RunEnvironment(
        kind="local", python="/opt/venv/bin/python", world_size=4, devices=(0, 1, 2, 3)
    )
    out = _whole([" A", " B"])
    result, calls = _run_equivalence(tmp_path, out, out, environment=four)
    assert result.passed
    assert "trtllm-llmapi-launch" in calls[0] and "mpirun" in calls[0]
    # The engines leg B starts are engine work: it is handed how the run starts one.
    launch = f"{LEG_B_ENGINE_LAUNCH_ENV}={shlex.quote(engine_launch(four))}"
    assert launch in calls[1] and "mpirun -n 4 " in engine_launch(four)
    leg_b = calls[1].replace(launch, "")
    assert "trtllm-llmapi-launch" not in leg_b and "mpirun" not in leg_b
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in leg_b
    for name in ("TLLM_SPAWN_PROXY_PROCESS", "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR"):
        assert f"-u {name} " in leg_b
    assert "/opt/venv/bin/python /onboard/scripts/disagg_leg_b.py" in leg_b


def test_a_contrast_command_that_is_not_python_runs_as_written(tmp_path):
    out = _whole([" A"])
    _, calls = _run_equivalence(
        tmp_path, out, out, contrast_command="/onboard/scripts/leg_b.sh --fast"
    )
    assert "/onboard/scripts/leg_b.sh --fast" in calls[1]
    assert "/opt/venv/bin/python /onboard/scripts/leg_b.sh" not in calls[1]


def test_leg_b_is_handed_what_leg_a_resolved(tmp_path):
    """A restated workload desynchronises at the next edit to either side."""
    import json

    out = _whole([" A", " B"])
    _, calls = _run_equivalence(tmp_path, out, out, member="tep4", max_tokens=128, timeout=2700)
    leg_b = calls[1]
    prompts = json.loads(Path(_assignment(leg_b, LEG_B_PROMPTS_FILE_ENV)).read_text())["prompts"]
    for prompt in prompts:
        assert shlex.quote(prompt) in calls[0]  # the very prompts leg A ran
    assert _assignment(leg_b, LEG_B_MAX_TOKENS_ENV) == "128"
    assert _assignment(leg_b, LEG_B_MEMBER_ENV) == "tep4"
    assert _assignment(leg_b, LEG_B_TIMEOUT_ENV) == "2700"
    argv = _assignment(leg_b, LEG_B_ARGV_ENV)
    assert "quickstart_advanced.py" in argv and "--max_tokens=128" in argv
    assert "--prompt" not in argv  # the prompts travel in the file
    # Nothing speculates here, so there is no depth to hand over.
    assert f"{LEG_B_DRAFT_LEN_ENV}=" not in leg_b
    # Leg A is not the one being handed anything.
    assert LEG_B_PROMPTS_FILE_ENV not in calls[0]


def test_leg_b_gets_the_draft_depth_when_leg_a_speculates(tmp_path):
    import yaml

    options = tmp_path / "dspark.yaml"
    options.write_text(
        yaml.safe_dump({"speculative_config": {"decoding_type": "DSpark", "max_draft_len": 5}})
    )
    task = dict(_TASK) | {"accuracy": {"llm_api_options": {"disagg": str(options)}}}
    out = _whole([" A"])
    _, calls = _run_equivalence(
        tmp_path, out, out, task_data=task, unit="disagg", features=["dspark"]
    )
    assert _assignment(calls[1], LEG_B_DRAFT_LEN_ENV) == "5"


def test_declared_prompt_lengths_reach_both_legs_beside_the_boot_prompts(tmp_path):
    """The disagg gate wants lengths hand-written prompts do not reach.

    Over one prefill chunk and past a page edge. A leg-B-only prompt cannot be
    added: the witness compares continuation lists and a count mismatch fails.
    """
    import json

    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import BootSpec

    spec = BootSpec(prompts=("What is the capital of France?",))
    out = _whole([" A", " B", " C"])
    _, calls = _run_equivalence(tmp_path, out, out, spec=spec, prompt_tokens=[5, 7])
    expected = ["What is the capital of France?", natural_text(5), natural_text(7)]
    for prompt in expected:
        assert shlex.quote(prompt) in calls[0]
    handed = json.loads(Path(_assignment(calls[1], LEG_B_PROMPTS_FILE_ENV)).read_text())
    assert handed["prompts"] == expected


def test_both_legs_are_asked_for_whole_continuations(tmp_path):
    out = _whole([" A"])
    _, calls = _run_equivalence(tmp_path, out, out)
    assert all(f"{WITNESS_OUTPUT_ENV}=1" in call for call in calls)


def test_a_difference_past_the_printed_prefix_fails(tmp_path):
    """The 200-character line is under a quarter of a 256-token continuation."""
    same = "x" * 200
    agg, dis = _whole([same + " tail one"], ids=False), _whole([same + " tail two"], ids=False)
    assert parse_generated_texts(agg) == parse_generated_texts(dis)
    result, _ = _run_equivalence(tmp_path, agg, dis)
    assert not result.passed
    assert "full text" in result.detail
    assert "first differs at character 206" in result.detail


def test_token_ids_win_over_text_when_both_legs_print_them(tmp_path):
    """Two token sequences can detokenize to one string; ids are the claim."""
    agg = _whole([" A B"], ids=[(1, 2, 3)])
    dis = _whole([" A B"], ids=[(1, 2, 4)])
    result, _ = _run_equivalence(tmp_path, agg, dis)
    assert not result.passed
    assert "token ids" in result.detail
    assert "first differs at token 2" in result.detail
    assert result.measured == 2


def test_a_leg_without_token_ids_cannot_pass_and_is_named(tmp_path):
    """Leg A from a quickstart that prints only the 200-character line.

    The legs agree on that prefix, which is under a quarter of a 256-token
    continuation: the rest of each is never compared.
    """
    result, _ = _run_equivalence(tmp_path, _quickstart(" A", " B", ids=False), _whole([" A", " B"]))
    assert not result.passed
    assert f"agree as {PREFIX_OBSERVABLE}, which is not token identity" in result.detail
    assert "the aggregated leg(s) did not print" in result.detail
    assert "examples/llm-api/quickstart_advanced.py" in result.detail
    assert "a contrast_command prints them itself" in result.detail


def test_whole_text_without_token_ids_cannot_pass_either(tmp_path):
    """Two token sequences can detokenize to one string; ids are the claim."""
    out = _whole([" A", " B"], ids=False)
    result, _ = _run_equivalence(tmp_path, out, out)
    assert not result.passed
    assert "agree as full text, which is not token identity" in result.detail
    assert "the aggregated and disagg leg(s)" in result.detail


def test_a_difference_in_a_prefix_still_fails_as_a_difference(tmp_path):
    """Without token ids a difference is still real; only agreement is refused."""
    agg, dis = _quickstart(" A", " B", ids=False), _quickstart(" A", " C", ids=False)
    result, _ = _run_equivalence(tmp_path, agg, dis)
    assert not result.passed
    assert "differ across the hop" in result.detail
    assert PREFIX_OBSERVABLE in result.detail and "not compared" in result.detail


def test_without_token_ids_the_minimum_length_is_not_skipped(tmp_path):
    """A 2-token answer agrees on its prefix as well as a 256-token one."""
    out = _quickstart(" A", " B", ids=False)
    result, _ = _run_equivalence(tmp_path, out, out, min_generated_tokens=32)
    assert not result.passed
    assert "nor held to min_generated_tokens=32" in result.detail


def test_a_leg_prints_token_ids_only_with_one_line_per_continuation():
    from agent_flow.workflows.trtllm_modeling_v2.checks.witness import prints_token_ids

    assert prints_token_ids(_whole([" A", " B"]))
    assert not prints_token_ids(_whole([" A", " B"], ids=False))
    assert not prints_token_ids(_quickstart(" A", " B", ids=False))
    assert not prints_token_ids(_whole([" A", " B"]) + _quickstart(" C", " D", ids=False))


def test_a_failing_verdict_says_where_each_continuation_first_differs(tmp_path):
    """Lost state shows at the first decode step; a late flip is numerics."""
    agg = _whole([" one two three", " same"], ids=[(1, 2, 3, 4, 5), (7, 7)])
    dis = _whole([" one two four", " same"], ids=[(1, 2, 3, 9, 5), (7, 7)])
    result, _ = _run_equivalence(tmp_path, agg, dis)
    assert not result.passed
    assert "continuation(s) [0]" in result.detail
    assert "#0 first differs at token 3" in result.detail
    assert "logit" in result.detail and "fluent, wrong text" in result.detail
    assert "around token 3" in result.stdout_tail


def test_a_count_mismatch_is_named_as_one(tmp_path):
    result, _ = _run_equivalence(tmp_path, _whole([" A", " B"]), _whole([" A"]))
    assert not result.passed
    assert "printed 2 continuation(s), disagg 1" in result.detail


def test_whole_continuations_are_parsed_verbatim():
    out = (
        "[0] Witness token ids: [128798, 17, 13]\n"
        "0: [1] Witness token ids: []\n"
        "[0] Witness text: 'it\\'s \"72\"\\nnext'\n"
    )
    assert parse_witness_token_ids(out) == ((128798, 17, 13), ())
    assert parse_witness_texts(out) == ('it\'s "72"\nnext',)


def test_first_difference_of_a_truncated_continuation_is_where_it_stops():
    assert first_difference((1, 2, 3), (1, 2, 3, 4)) == 3
    assert first_difference("abc", "abx") == 2
    assert compare_continuations(_whole(["x"]), _whole(["x"])).identical


# ------------------------------------------------------------- the draft depth


def test_a_witness_asks_for_the_depth_the_unit_was_built_for(tmp_path):
    """The witness reads the depth from the options YAML, as the boot gate does.

    A fixed depth of 1 is harmless for MTP and fatal for DSpark: `TorchLlmArgs`
    rejects a DSpark config whose `block_size` differs from `max_draft_len`, so
    every witness after a `dspark` unit would die at startup and report a leg
    that "did not complete" -- a message that reads as the unit's own defect.
    """
    import yaml

    options = tmp_path / "dspark.yaml"
    options.write_text(
        yaml.safe_dump({"speculative_config": {"decoding_type": "DSpark", "max_draft_len": 5}})
    )
    task = dict(_TASK) | {"accuracy": {"llm_api_options": {"dspark": str(options)}}}
    argv = witness_command(task, _env(), unit="dspark", features=["dspark"])
    assert "--spec_decode_algo=DSPARK" in argv
    assert "--spec_decode_max_draft_len=5" in argv


# -------------------------------------------------------- the counter contract


def _counter(**fields) -> str:
    return f"{WITNESS_COUNTER_MARKER} " + " ".join(f"{k}={v}" for k, v in fields.items())


def test_counter_lines_are_read_whatever_precedes_them():
    """A logger prefix or a rank tag in front of the marker changes nothing."""
    out = (
        "[TensorRT-LLM][INFO] " + _counter(forwarded_tokens=4100, decoder_tokens=128) + "\n"
        "[rank1] " + _counter(kv_bytes_per_token=890) + "\n"
        "an ordinary line\n"
    )
    assert parse_witness_counters(out) == [
        {"forwarded_tokens": 4100, "decoder_tokens": 128},
        {"kv_bytes_per_token": 890},
    ]


def test_the_filler_is_one_short_word_per_token_at_least():
    """The long prompt's length is DECLARED, so the workload is built to it."""
    text = filler_words(4096)
    assert len(text.split()) == 4096
    assert max(len(word) for word in text.split()) <= 5
    # 16384 words must stay under Linux's 128 KiB cap on one argv string.
    assert len(filler_words(16384).encode()) < 128 * 1024


# ------------------------------------------------------------ decoder replay

_REPLAY_OFF = "REPLAY_OFF_FOR_TEST=1"


def _leg_runner(on_out: str, off_out: str, *, marker: str = _REPLAY_OFF, fail_leg=None):
    """A runner whose output depends on whether the contrast env is in the command."""
    calls: list[str] = []

    def runner(command, env=None, timeout=0):
        calls.append(command)
        off = marker in command
        if fail_leg == ("OFF" if off else "ON"):
            return CommandResult(returncode=1, stdout="", stderr="boom")
        return CommandResult(returncode=0, stdout=(off_out if off else on_out))

    return runner, calls


def _lines(*rows: dict) -> str:
    return "".join(_counter(**row) + "\n" for row in rows) + _quickstart(" A", " B")


def _decoder(on_rows, off_rows, **kw):
    runner, calls = _leg_runner(_lines(*on_rows), _lines(*off_rows), **kw)
    result = run_replay_witness(
        _env(),
        task_data=_TASK,
        feature="decoder_replay",
        contrast_env={"REPLAY_OFF_FOR_TEST": "1"},
        window=128,
        prompt_tokens=4096,
        runner=runner,
        unit="decoder_replay",
    )
    return result, calls


# Quickstart runs every prompt twice -- a batched pass, then a streamed one --
# so a real log carries a cold request and a warm (prefix-hit) one.
_COLD = {"forwarded_tokens": 4100}
_WARM = {"forwarded_tokens": 36}


def test_a_bounded_decoder_passes():
    result, calls = _decoder(
        on_rows=[_COLD | {"decoder_tokens": 128}, _WARM | {"decoder_tokens": 36}],
        off_rows=[_COLD | {"decoder_tokens": 4100}, _WARM | {"decoder_tokens": 36}],
    )
    assert result.passed, result.detail
    assert result.name == "replay[decoder_replay]"
    assert len(calls) == 2


def test_a_target_that_implements_nothing_fails_the_on_leg():
    """The case this witness exists for: full prefill, accuracy green."""
    result, _ = _decoder(
        on_rows=[_COLD | {"decoder_tokens": 4100}],
        off_rows=[_COLD | {"decoder_tokens": 4100}],
    )
    assert not result.passed
    assert "not bounded" in result.detail


def test_a_counter_that_prints_a_formula_is_caught_by_the_off_leg():
    """`decoder_tokens=min(prompt, window)` passes ON; OFF is why there are two legs."""
    result, _ = _decoder(
        on_rows=[_COLD | {"decoder_tokens": 128}],
        off_rows=[_COLD | {"decoder_tokens": 128}],
    )
    assert not result.passed
    assert "did not turn the feature off" in result.detail


def test_a_prompt_that_never_ran_cold_at_length_proves_nothing():
    """Under the window, bounded and unbounded decoders do the same work."""
    short = {"forwarded_tokens": 100}
    result, _ = _decoder(
        on_rows=[short | {"decoder_tokens": 100}],
        off_rows=[short | {"decoder_tokens": 100}],
    )
    assert not result.passed
    assert "prompt_tokens=4096" in result.detail


def test_no_counter_line_is_a_named_failure_not_a_pass():
    runner, _ = _leg_runner(_quickstart(" A", " B"), _quickstart(" A", " B"))
    result = run_replay_witness(
        _env(),
        task_data=_TASK,
        feature="decoder_replay",
        contrast_env={"REPLAY_OFF_FOR_TEST": "1"},
        window=128,
        prompt_tokens=4096,
        runner=runner,
    )
    assert not result.passed
    assert "printed no witness counter" in result.detail


def test_a_counter_line_missing_a_field_names_it():
    result, _ = _decoder(on_rows=[{"forwarded_tokens": 4100}], off_rows=[_COLD])
    assert not result.passed
    assert "decoder_tokens" in result.detail


def test_a_storage_line_under_the_same_marker_is_not_a_replay_line():
    """One target may implement kv_quant and replay; each witness reads its own."""
    result, _ = _decoder(
        on_rows=[{"kv_bytes_per_token": 890}, _COLD | {"decoder_tokens": 128}],
        off_rows=[{"kv_bytes_per_token": 890}, _COLD | {"decoder_tokens": 4100}],
    )
    assert result.passed, result.detail


def test_both_legs_count_and_only_the_second_is_switched_off():
    _, calls = _decoder(
        on_rows=[_COLD | {"decoder_tokens": 128}],
        off_rows=[_COLD | {"decoder_tokens": 4100}],
    )
    assert all(f"{WITNESS_COUNTER_ENV}=1" in call for call in calls)
    assert all("TRTLLM_MODELING_V2=require" in call for call in calls)
    assert _REPLAY_OFF not in calls[0] and _REPLAY_OFF in calls[1]
    # The workload is the declared one, not the boot spec's two short prompts.
    assert "capital of France" not in calls[0]
    assert calls[0].count(" one ") >= 400


def test_a_leg_that_dies_is_named():
    runner, _ = _leg_runner(_lines(_COLD | {"decoder_tokens": 128}), "", fail_leg="OFF")
    result = run_replay_witness(
        _env(),
        task_data=_TASK,
        feature="decoder_replay",
        contrast_env={"REPLAY_OFF_FOR_TEST": "1"},
        window=128,
        prompt_tokens=4096,
        runner=runner,
    )
    assert not result.passed
    assert "replay-OFF leg did not complete" in result.detail


# ------------------------------------------------------------ encoder replay

_TASK_REUSE = dict(_TASK) | {"runtime": {"kv_cache_reuse": True, "extra_flags": []}}
_COLD_E = {
    "prompt_tokens": 4105,
    "reused_tokens": 0,
    "replayed_tokens": 0,
    "forwarded_tokens": 4105,
}


def _encoder(on_rows, off_rows, *, task=_TASK_REUSE, alignment=1):
    runner, calls = _leg_runner(_lines(*on_rows), _lines(*off_rows))
    result = run_replay_witness(
        _env(),
        task_data=task,
        feature="encoder_replay",
        contrast_env={"REPLAY_OFF_FOR_TEST": "1"},
        window=128,
        prefix_tokens=4096,
        reuse_alignment=alignment,
        runner=runner,
        unit="encoder_replay",
    )
    return result, calls


def _hit(reused: int, replayed: int, forwarded: int | None = None) -> dict:
    prompt = 4105
    if forwarded is None:
        forwarded = prompt - reused + replayed
    return {
        "prompt_tokens": prompt,
        "reused_tokens": reused,
        "replayed_tokens": replayed,
        "forwarded_tokens": forwarded,
    }


def test_a_prefix_hit_that_replays_the_window_passes():
    result, calls = _encoder(
        on_rows=[_COLD_E, _hit(4096, 128)],
        off_rows=[_COLD_E, _hit(4096, 0)],
    )
    assert result.passed, result.detail
    assert len(calls) == 2


def test_the_encoder_workload_is_two_prompts_sharing_the_prefix():
    _, calls = _encoder(on_rows=[_COLD_E, _hit(4096, 128)], off_rows=[_COLD_E])
    assert "What comes next" in calls[0] and "Say it backwards" in calls[0]


def test_a_hit_that_recomputes_the_whole_prefix_breaks_the_accounting():
    """The shortcut "SWA KV is missing" invites: report the hit, forward everything."""
    result, _ = _encoder(
        on_rows=[_COLD_E, _hit(4096, 128, forwarded=4105)],
        off_rows=[_COLD_E, _hit(4096, 0)],
    )
    assert not result.passed
    assert "prompt - reused + replayed" in result.detail


def test_no_prefix_hit_means_nothing_was_witnessed():
    """No reuse at all satisfies every identity trivially; it must not pass."""
    result, _ = _encoder(on_rows=[_COLD_E, _COLD_E], off_rows=[_COLD_E, _COLD_E])
    assert not result.passed
    assert "no request reused more than window=128" in result.detail


def test_a_replay_of_the_wrong_length_fails():
    result, _ = _encoder(
        on_rows=[_COLD_E, _hit(4096, 64)],
        off_rows=[_COLD_E, _hit(4096, 0)],
    )
    assert not result.passed
    assert "min(window=128, reused)" in result.detail


def test_the_off_leg_must_replay_nothing():
    result, _ = _encoder(
        on_rows=[_COLD_E, _hit(4096, 128)],
        off_rows=[_COLD_E, _hit(4096, 128)],
    )
    assert not result.passed
    assert "did not turn the" in result.detail


def test_an_odd_boundary_on_a_ratio_two_model_fails_when_declared():
    result, _ = _encoder(
        on_rows=[_COLD_E, _hit(4095, 128)],
        off_rows=[_COLD_E, _hit(4095, 0)],
        alignment=2,
    )
    assert not result.passed
    assert "not a multiple of 2" in result.detail
    # Undeclared, the same boundary is not this witness's business.
    result, _ = _encoder(
        on_rows=[_COLD_E, _hit(4095, 128)],
        off_rows=[_COLD_E, _hit(4095, 0)],
    )
    assert result.passed, result.detail


def test_block_reuse_off_refuses_before_launching_anything():
    task = dict(_TASK) | {"runtime": {"kv_cache_reuse": False, "extra_flags": []}}
    result, calls = _encoder(on_rows=[_COLD_E], off_rows=[_COLD_E], task=task)
    assert not result.passed
    assert calls == []
    assert "kv_cache_reuse" in result.detail


# -------------------------------------------------------------------- kv_quant

_BF16 = "KV_STORAGE_FOR_TEST=bf16"


def _kv(native_out, bf16_out, *, exact=True, tolerance=0.02):
    runner, calls = _leg_runner(native_out, bf16_out, marker=_BF16)
    result = run_kv_quant_witness(
        _env(),
        task_data=_TASK,
        expected_bytes_per_token=890,
        exact=exact,
        contrast_env={"KV_STORAGE_FOR_TEST": "bf16"},
        bytes_tolerance=tolerance,
        runner=runner,
        unit="kv_quant",
    )
    return result, calls


def _storage(bytes_per_token: int, a=" Paris.", b=" the process") -> str:
    return _counter(kv_bytes_per_token=bytes_per_token) + "\n" + _quickstart(a, b)


def test_native_storage_at_the_declared_figure_and_identical_output_passes():
    result, calls = _kv(_storage(890), _storage(3200))
    assert result.passed, result.detail
    assert len(calls) == 2
    assert all(f"{WITNESS_COUNTER_ENV}=1" in call for call in calls)
    assert _BF16 not in calls[0] and _BF16 in calls[1]


def test_a_unit_still_on_bf16_pools_fails_on_bytes():
    """The case this witness exists for: output identical, storage unchanged."""
    result, calls = _kv(_storage(3200), _storage(3200))
    assert not result.passed
    assert "3200 B/token against the declared 890" in result.detail
    assert len(calls) == 1


def test_padding_passes_and_a_layout_change_does_not():
    assert _kv(_storage(900), _storage(3200))[0].passed
    # The indexer scale padded from 4 bytes to 16: 890 -> 920.
    assert not _kv(_storage(920), _storage(3200))[0].passed


def test_a_contrast_that_selects_nothing_cannot_vouch_for_identity():
    result, _ = _kv(_storage(890), _storage(890))
    assert not result.passed
    assert "did not select bf16 storage" in result.detail


def test_native_storage_that_moves_the_output_stored_other_values():
    result, _ = _kv(_storage(890), _storage(3200, b=" a process"))
    assert not result.passed
    assert "stored values are not the reference's" in result.detail


def test_kv_quant_legs_that_print_no_token_ids_cannot_vouch_for_identity():
    native = _counter(kv_bytes_per_token=890) + "\n" + _quickstart(" A", " B", ids=False)
    bf16 = _counter(kv_bytes_per_token=3200) + "\n" + _quickstart(" A", " B", ids=False)
    result, calls = _kv(native, bf16)
    assert not result.passed
    assert len(calls) == 2
    assert "native storage at 890 B/token (declared 890) against 3200 B/token" in result.detail
    assert f"agree as {PREFIX_OBSERVABLE}, which is not token identity" in result.detail
    assert "native and bf16 leg(s)" in result.detail


def test_no_storage_line_is_a_named_failure():
    result, _ = _kv(_quickstart(" A", " B"), _storage(3200))
    assert not result.passed
    assert "printed no witness counter" in result.detail


def test_a_lossy_attention_path_drops_the_identity_leg_and_says_so():
    result, calls = _kv(_storage(890), _storage(3200, b=" different"), exact=False)
    assert result.passed
    assert len(calls) == 1
    assert "WEAKER CLAIM" in result.detail


# ----------------------------------------------- one schedule on both legs

_BATCHED = dict(_TASK) | {
    "runtime": {"cuda_graph": True, "extra_flags": ["--max_num_tokens=2048", "--max_batch_size=8"]}
}


def test_a_declared_batch_limit_replaces_the_runtime_blocks():
    """Batch composition is a run-to-run source; the witness states ONE limit.

    argparse would take the last of two `--max_batch_size` flags, but leg B
    reads leg A's argv for its limits, and a stale `=8` beside the `=1` is a
    second answer to one question.
    """
    argv = witness_command(_BATCHED, _env(), unit="kv_quant", max_batch_size=1)
    assert argv.count("--max_batch_size") == 1
    assert "--max_batch_size=1" in argv
    assert "--max_batch_size=8" not in argv
    assert "--max_num_tokens=2048" in argv


def test_without_a_declared_limit_the_argv_is_unchanged():
    """A task.yaml without the key keeps its exact commands."""
    argv = witness_command(_BATCHED, _env(), unit="kv_quant")
    assert "--max_batch_size=8" in argv and argv.count("--max_batch_size") == 1


def test_every_spelling_of_the_batch_flag_is_replaced():
    for flags in (
        ["--max_batch_size=8", "--x"],
        ["--max_batch_size 8", "--x"],
        ["--max_batch_size", "8", "--x"],
    ):
        assert with_max_batch_size(flags, 1) == ["--x", "--max_batch_size=1"]
    assert with_max_batch_size(["--max_batch_size=8"], None) == ["--max_batch_size=8"]


def test_the_equivalence_witness_runs_both_legs_at_the_declared_limit(tmp_path):
    out = _whole([" A", " B"])
    result, calls = _run_equivalence(tmp_path, out, out, task_data=_BATCHED, max_batch_size=1)
    assert result.passed
    assert "max_batch_size=1" in result.detail
    assert "--max_batch_size=1" in calls[0] and "--max_batch_size=8" not in calls[0]
    leg_b = calls[1]
    assert _assignment(leg_b, LEG_B_MAX_BATCH_SIZE_ENV) == "1"
    handed = _assignment(leg_b, LEG_B_ARGV_ENV)
    assert "--max_batch_size=1" in handed and "--max_batch_size=8" not in handed


def test_leg_b_is_not_handed_a_limit_nobody_declared(tmp_path):
    out = _whole([" A"])
    _, calls = _run_equivalence(tmp_path, out, out, task_data=_BATCHED)
    assert f"{LEG_B_MAX_BATCH_SIZE_ENV}=" not in calls[1]
    assert "--max_batch_size=8" in _assignment(calls[1], LEG_B_ARGV_ENV)


def test_an_env_only_equivalence_leg_b_gets_the_limit_in_its_argv():
    calls: list[str] = []

    def runner(command, env=None, timeout=0):
        calls.append(command)
        return CommandResult(returncode=0, stdout=_quickstart(" A", " B"))

    run_equivalence_witness(
        _env(), task_data=_BATCHED, contrast_env={"X": "1"}, runner=runner, max_batch_size=1
    )
    assert all("--max_batch_size=1" in call for call in calls)


# ----------------------------------------- an environment both legs share

_REPLAYS_OFF = {
    "TRTLLM_DSV41_ENCODER_SWA_BOUNDED_REPLAY": "0",
    "TRTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY": "0",
}


def test_legs_env_reaches_leg_a_and_is_handed_to_leg_b(tmp_path):
    """Leg B's engines guard the replays off, so leg A runs them off too.

    Otherwise the legs differ in the replays as well as in the hop, and token
    identity no longer holds by construction.
    """
    import json

    out = _whole([" A", " B"])
    result, calls = _run_equivalence(tmp_path, out, out, legs_env=_REPLAYS_OFF)
    assert result.passed
    agg, dis, ablated = calls
    # The self-test run is leg B again, under the same shared environment.
    assert json.loads(_assignment(ablated, LEG_B_LEGS_ENV_ENV)) == _REPLAYS_OFF
    for name, value in _REPLAYS_OFF.items():
        assert _assignment(agg, name) == value
        # In leg B's own environment, so a launcher that copies it passes it on...
        assert _assignment(dis, name) == value
    # ...and named, for one that sets engine variables of its own afterwards.
    assert json.loads(_assignment(dis, LEG_B_LEGS_ENV_ENV)) == _REPLAYS_OFF
    assert LEG_B_LEGS_ENV_ENV not in agg
    # The disagg switch is still leg B's alone.
    assert "TRTLLM_DISAGG" not in agg
    assert _assignment(dis, "TRTLLM_DISAGG") == "1"
    # A pass holds under these settings, so the verdict says which.
    assert "both legs under TRTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY=0, " in result.detail


def test_without_legs_env_neither_leg_changes(tmp_path):
    out = _whole([" A"])
    result, calls = _run_equivalence(tmp_path, out, out)
    assert LEG_B_LEGS_ENV_ENV not in calls[1]
    assert not any(name in call for name in _REPLAYS_OFF for call in calls)
    assert "both legs under" not in result.detail


def test_legs_env_names_the_failing_verdict_too(tmp_path):
    result, _ = _run_equivalence(
        tmp_path, _whole([" A"]), _whole([" B"]), legs_env={"REPLAY_FOR_TEST": 0}
    )
    assert not result.passed
    assert "both legs under REPLAY_FOR_TEST=0" in result.detail


def test_an_env_only_equivalence_applies_legs_env_to_both_legs():
    calls: list[str] = []

    def runner(command, env=None, timeout=0):
        calls.append(command)
        return CommandResult(returncode=0, stdout=_quickstart(" A", " B"))

    run_equivalence_witness(
        _env(), task_data=_TASK, contrast_env={"X": "1"}, runner=runner, legs_env={"Y": "0"}
    )
    assert all(_assignment(call, "Y") == "0" for call in calls)
    assert "X=1" not in calls[0] and _assignment(calls[1], "X") == "1"


def test_the_kv_quant_legs_share_the_declared_limit_and_print_whole_continuations():
    runner, calls = _leg_runner(_storage(890), _storage(3200), marker=_BF16)
    result = run_kv_quant_witness(
        _env(),
        task_data=_BATCHED,
        expected_bytes_per_token=890,
        contrast_env={"KV_STORAGE_FOR_TEST": "bf16"},
        runner=runner,
        unit="kv_quant",
        max_batch_size=1,
    )
    assert result.passed, result.detail
    assert "max_batch_size=1" in result.detail
    for call in calls:
        assert "--max_batch_size=1" in call and "--max_batch_size=8" not in call
        assert f"{WITNESS_OUTPUT_ENV}=1" in call


def test_the_residency_legs_share_the_declared_limit():
    calls: list[str] = []

    def runner(command, env=None, timeout=0):
        calls.append(command)
        return CommandResult(returncode=0, stdout=_quickstart(" A", " B", 1000 + 200 * len(calls)))

    result = run_residency_witness(
        _env(),
        task_data=_BATCHED,
        expected_mib=100.0,
        contrast_env={"TARGET_OFFLOAD": "0"},
        runner=runner,
        max_batch_size=1,
    )
    assert result.passed, result.detail
    assert all("--max_batch_size=1" in call for call in calls)
    assert all(f"{WITNESS_OUTPUT_ENV}=1" in call for call in calls)


# -------------------------------------- a failed comparison shows both legs

_AGG_TEXT = " The pass key is 41627. I found it by scanning for the only sentence."
_DIS_TEXT = " The pass key is 41627. I found it by scanning for the one sentence."


def test_a_failing_equivalence_verdict_quotes_both_legs_around_the_difference(tmp_path):
    """The index says WHERE; the words say whether it is lost state or a near-tie."""
    agg = _whole([_AGG_TEXT, " same"], ids=[(1, 2, 3, 4, 5, 6), (7,)])
    dis = _whole([_DIS_TEXT, " same"], ids=[(1, 2, 3, 4, 9, 6), (7,)])
    result, _ = _run_equivalence(tmp_path, agg, dis)
    assert not result.passed
    assert "#0 first differs at token 4" in result.detail
    # The one-line verdict quotes the words on both sides of the flip...
    assert (
        "around it, #0 text [27:69] around character 57: "
        "aggregated 'ound it by scanning for the only sentence.' vs "
        "disagg 'ound it by scanning for the one sentence.'"
    ) in result.detail
    # ...and the evidence block shows ids and text, both legs, labelled.
    tail = result.stdout_tail
    assert "continuation #0: first differs at token 4" in tail
    assert "token ids [0:6] around token 4:" in tail
    assert "aggregated: [1, 2, 3, 4, 5, 6]" in tail
    assert "disagg    : [1, 2, 3, 4, 9, 6]" in tail
    assert "text [" in tail and "the only sentence" in tail and "the one sentence" in tail


def test_ids_that_differ_under_identical_text_are_shown_as_ids(tmp_path):
    agg = _whole([" A B"], ids=[(1, 2, 3)])
    dis = _whole([" A B"], ids=[(1, 2, 4)])
    result, _ = _run_equivalence(tmp_path, agg, dis)
    assert (
        "around it, #0 token ids [0:3] around token 2: aggregated [1, 2, 3] vs disagg [1, 2, 4]"
    ) in result.detail
    assert "text: identical on both legs" in result.stdout_tail


def test_the_evidence_covers_several_differing_continuations_and_counts_the_rest(tmp_path):
    texts_a = [f" answer {i} alpha" for i in range(6)]
    texts_b = [f" answer {i} beta" for i in range(6)]
    result, _ = _run_equivalence(tmp_path, _whole(texts_a), _whole(texts_b))
    tail = result.stdout_tail
    for i in range(4):
        assert f"continuation #{i}:" in tail
    assert "continuation #4:" not in tail
    assert "2 more differing continuation(s)" in tail


def test_a_failing_kv_quant_verdict_names_the_first_difference_and_both_legs():
    """Not just which continuation differs: where, and what both legs had there."""
    ids_native, ids_bf16 = [(1,), (5, 6, 7)], [(1,), (5, 6, 8)]
    native = _counter(kv_bytes_per_token=890) + "\n" + _whole([" same", _AGG_TEXT], ids_native)
    bf16 = _counter(kv_bytes_per_token=3200) + "\n" + _whole([" same", _DIS_TEXT], ids_bf16)
    runner, _ = _leg_runner(native, bf16, marker=_BF16)
    result = run_kv_quant_witness(
        _env(),
        task_data=_TASK,
        expected_bytes_per_token=890,
        contrast_env={"KV_STORAGE_FOR_TEST": "bf16"},
        runner=runner,
        unit="kv_quant",
        max_batch_size=1,
    )
    assert not result.passed
    assert "continuation(s) [1]" in result.detail
    assert "#1 first differs at token 2" in result.detail
    assert "native 'ound it by scanning for the only sentence.'" in result.detail
    assert "bf16 'ound it by scanning for the one sentence.'" in result.detail
    assert "stored values are not the reference's" in result.detail
    assert "max_batch_size=1" in result.detail
    assert result.measured == 2
    assert "native: [5, 6, 7]" in result.stdout_tail
    assert "bf16  : [5, 6, 8]" in result.stdout_tail


def test_a_failing_residency_verdict_names_the_first_difference():
    """A difference in the 200-character prefix alone is still a difference."""
    result, _ = _residency(
        on_mib=1000, off_mib=1200, on_text=(" A", " B x"), off_text=(" A", " B y"), ids=False
    )
    assert not result.passed
    assert "identical text" in result.detail
    assert "#1 first differs at character 3" in result.detail
    assert "offloaded ' B x'" in result.detail and "resident ' B y'" in result.detail
    assert "around character 3" in result.stdout_tail


def test_the_shim_builds_the_named_target_and_prints_whole_continuations(capsys):
    """A stock quickstart takes no `modeling_v2` and prints only 200 characters."""
    import sys
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.checks.witness import QUICKSTART_SHIM

    sys.path.insert(0, str(QUICKSTART_SHIM.parent))
    import quickstart_shim

    class FakeLLM:
        def __init__(self, model, **kwargs):
            self.kwargs = kwargs

        def generate_async(self, prompt, params, streaming=False):
            seq = SimpleNamespace(token_ids=[1, 2], text="ab")
            return iter([SimpleNamespace(outputs=[seq])] * 2)

    llm = quickstart_shim.wrap_llm(FakeLLM, mode="require", full_output=True)("m")
    assert llm.kwargs == {"modeling_v2": "require"}
    assert len(list(llm.generate_async("p", None, streaming=True))) == 2
    list(llm.generate_async("q", None, streaming=True))
    out = capsys.readouterr().out
    assert "[0] Witness token ids: [1, 2]\n[0] Witness text: 'ab'" in out
    assert "[1] Witness token ids: [1, 2]" in out
    quiet = quickstart_shim.wrap_llm(FakeLLM, mode=None, full_output=False)("m")
    assert quiet.kwargs == {} and not list(quiet.generate_async("p", None)) == []
    assert capsys.readouterr().out == ""


# ------------------------------------- the declared witness: leg B, tree, reuse


def _declared_task(repo: str = "/repo", **block) -> dict:
    return {
        "trtllm_repo_path": repo,
        "checkpoint_path": "/ckpt",
        "parallel": "tp1",
        "features": ["disagg", "host_offload"],
        "runtime": {"extra_flags": []},
        "accuracy": {
            "witness": {
                "disagg": {"min_generated_tokens": 1, **block},
                "host_offload": {
                    "contrast_env": {"TARGET_OFFLOAD": "0"},
                    "expected_mib": 100,
                    "min_generated_tokens": 1,
                },
            }
        },
    }


_ONE = "[0] Prompt: 'p'..., Generated text: ' A'...\n[0] Witness token ids: [1, 2]\n"


def _declared_runner(*, reuse: str = ""):
    """Every leg prints the same continuation; leg B's self-test a moved one; ``reuse`` per leg."""
    seen: list[str] = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        if f"{LEG_B_ABLATE_ENV}={ABLATE_STATE}" in command:
            return CommandResult(0, stdout=_ONE.replace("[1, 2]", "[3]") + "Witness ablated: kv\n")
        memory = f"{DEVICE_MEMORY_MARKER} {1200 if 'TARGET_OFFLOAD=0' in command else 1000}\n"
        return CommandResult(0, stdout=memory + _ONE + reuse)

    run.seen = seen  # type: ignore[attr-defined]
    return run


def test_a_disagg_witness_with_no_launcher_runs_the_flows_leg_b(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.checks import witness

    run = _declared_runner()
    transceiver = {"backend": "UCX", "transceiver_runtime": "CPP"}
    result = witness.run_declared_witness(
        _env(),
        unit="disagg",
        member="tp1",
        task_data=_declared_task(cache_transceiver_config=transceiver),
        runner=run,
        workdir=tmp_path,
    )
    assert result.passed, result.detail
    _, leg_b, ablated = run.seen
    words = shlex.split(leg_b)
    assert f"{_env().python} {witness.DISAGG_LEG_B}" in leg_b, "the package's own launcher"
    # A one-rank engine has no launcher to create its worker, so each leg-B engine runs it in process.
    assert (
        f"{witness.LEG_B_ENGINE_LAUNCH_ENV}=TLLM_WORKER_USE_SINGLE_PROCESS=1 {_env().python}"
        in words
    )
    assert f"{witness.LEG_B_TRANSCEIVER_ENV}={json.dumps(transceiver, sort_keys=True)}" in words
    assert str(witness.DISAGG_LEG_B) in ablated


def test_a_launcher_of_the_runs_own_is_still_run_when_named(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.checks import witness

    run = _declared_runner()
    witness.run_declared_witness(
        _env(),
        unit="disagg",
        member="tp1",
        task_data=_declared_task(contrast_command="/onboard/leg_b.py", contrast_env={"X": "1"}),
        runner=run,
        workdir=tmp_path,
    )
    assert "/onboard/leg_b.py" in run.seen[1] and str(witness.DISAGG_LEG_B) not in run.seen[1]
    assert witness.LEG_B_TRANSCEIVER_ENV not in run.seen[1], "declared only when the block says"


def test_a_standalone_witness_names_the_tree_it_measured(tmp_path):
    import subprocess

    from agent_flow.workflows.trtllm_modeling_v2.checks import gates, witness

    tree = tmp_path / "tree"
    tree.mkdir()
    git = ["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    (tree / "README").write_text("x\n", encoding="utf-8")
    subprocess.run([*git, "add", "README"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    head = subprocess.run(
        [*git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    result = witness.run_declared_witness(
        _env(),
        unit="host_offload",
        member="tp1",
        task_data=_declared_task(str(tree)),
        runner=_declared_runner(),
        workdir=tmp_path / "w",
    )
    assert result.passed, result.detail
    assert result.tree_state.startswith(head[:7])
    assert f"[tree {result.tree_state}]" in result.summary()
    unknown = witness.run_declared_witness(
        _env(),
        unit="host_offload",
        member="tp1",
        task_data=_declared_task(str(tmp_path / "not-a-checkout")),
        runner=_declared_runner(),
        workdir=tmp_path / "w2",
    )
    assert unknown.tree_state == gates.UNKNOWN_TREE_STATE


def test_a_witness_leg_that_asks_for_reuse_and_reused_nothing_fails(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.checks import witness
    from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io

    def reuse(**fields):
        record = {"requests": 4, "reported": 4, "hits": 0, "resumable": 2, "concurrency": 1}
        record.update(fields, min_resume=64, probe=None)
        return f"{workload_io.REUSE_MARKER} {json.dumps(record)}\n"

    def run_with(line):
        return witness.run_declared_witness(
            _env(),
            unit="host_offload",
            member="tp1",
            task_data=_declared_task(),
            runner=_declared_runner(reuse=line),
            workdir=tmp_path,
        )

    nothing = run_with(reuse())
    assert not nothing.passed
    assert nothing.detail.startswith(
        "a witness leg: the unit asks for KV-cache block reuse and the engine of the offloaded "
        "leg reused nothing: 2 of 4 requests shared more than 64 prompt tokens"
    )
    assert run_with(reuse(hits=2)).passed, "the engine resumed: nothing to say"
