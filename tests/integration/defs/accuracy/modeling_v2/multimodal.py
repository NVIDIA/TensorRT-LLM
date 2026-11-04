# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native image checks within one accuracy-test engine: cache construction, content and ranks.

Path invariants fail the case: the prompt round trip, a fresh cache-salt namespace with no hit,
and hits that stay before the first image that differs. Missing cache telemetry, a missing warm
witness, a rank without image prefill evidence and a wrong answer about image content or order
are INCONCLUSIVE: greedy text may change across kernels or scheduling.
"""

import base64
import hashlib
import json
import re
import time
from contextlib import closing, contextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi import SchedulingParams
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
from tensorrt_llm.sampling_params import GuidedDecodingParams

# Output tokens per question: room for the reasoning block that precedes every answer at
# reasoning effort low.
ANSWER_TOKENS = 1024
# The cache pairs use one image size, so their prompts are token-identical and only the images
# differ: a hit may then extend no further than the first image that differs.
PAIR_SIZE = (224, 168)
# The structured-output case: one color from the panel's palette, as a JSON object.
_COLOR_SCHEMA = {
    "type": "object",
    "properties": {"color": {"type": "string", "enum": ["red", "blue", "green"]}},
    "required": ["color"],
    "additionalProperties": False,
}
# Shapes are drawn on white, so white and black are not answers.
_COLORS = re.compile(r"\b(red|blue|green|yellow|orange|purple|pink|brown|gray|grey|paris)\b")
_ORDER = re.compile(r"\b(first|second)\b")
_REFUSAL = re.compile(r"can't|cannot|don't|didn't|unable|blank|if you|not see")
_PREFIX = (
    "You are checking a test image. Look at the picture carefully, ignore the white background "
    "and answer only the question that follows the image, with no explanation."
)


def image_part(image):
    buffer = BytesIO()
    image.save(buffer, "PNG")
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        },
    }


def text_part(text):
    return {"type": "text", "text": text}


def shape(kind, size, color):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    width, height = size
    margin = max(8, min(width, height) // 8)
    bounds = (margin, margin, width - margin - 1, height - margin - 1)
    if kind == "circle":
        draw.ellipse(bounds, fill=color)
    elif kind == "square":
        draw.rectangle(bounds, fill=color)
    else:
        draw.polygon(
            [
                (width // 2, margin),
                (width - margin - 1, height - margin - 1),
                (margin, height - margin - 1),
            ],
            fill=color,
        )
    return image


def native_prompt(llm, messages, images):
    ids = llm.tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        return_tensors=None,
        reasoning_effort="low",
    )
    assert len(ids) + ANSWER_TOKENS <= llm.args.max_seq_len, (
        "Image expansion exceeded the sequence budget"
    )
    prompt = {"prompt_token_ids": ids}
    if images:
        prompt["multi_modal_data"] = {"image": images}
    return prompt


def final_answer(text):
    """The answer in ``text`` (decoded with special tokens kept), without the reasoning before it."""
    return ReasoningParserFactory.create_reasoning_parser("mistral_large_4").parse(text).content


def image_tokens(llm):
    """The [IMG] and [IMG_END] token ids of the native tokenizer."""
    tekken = llm.tokenizer.tokenizer
    lookup = getattr(tekken, "get_special_token", None) or tekken.get_control_token
    return lookup("[IMG]"), lookup("[IMG_END]")


def image_spans(ids, img, img_end):
    """(first [IMG], [IMG_END]) positions of each image, in prompt order."""
    spans, start = [], None
    for position, token in enumerate(ids):
        if token == img and start is None:
            start = position
        elif token == img_end:
            assert start is not None, "[IMG_END] without an image"
            spans.append((start, position))
            start = None
    assert start is None, "Image without [IMG_END]"
    return spans


def answer_matches(text, expected):
    """Whether the answer names the expected colors (or first / second) in order."""
    answer = text.lower().replace("’", "'")
    # A refusal can repeat the expected words as a hypothetical example.
    if _REFUSAL.search(answer):
        return False
    if expected in (["first"], ["second"]):
        return _ORDER.findall(answer)[:1] == expected
    words = _COLORS.findall(answer)
    # Repeated descriptions and a repeated final answer must keep the same order.
    runs = [word for index, word in enumerate(words) if index == 0 or word != words[index - 1]]
    return (
        bool(runs)
        and len(runs) % len(expected) == 0
        and runs == expected * (len(runs) // len(expected))
    )


def pair_failures(hits, bounds, witnesses):
    """Cache checks of one salted construction.

    hits: cached tokens per case name (None when telemetry was missing). bounds: the largest
    legal hit per case. witnesses: the hit a case must exceed to show the cache reused image
    blocks, i.e. the start of its first image: hits are whole blocks.
    """
    failures = []
    for name, hit in hits.items():
        if hit is None:
            failures.append(f"INCONCLUSIVE: {name}: missing cache telemetry")
        elif hit > bounds[name]:
            failures.append(f"FAIL: {name}: illegal cache hit {hit} > {bounds[name]}")
    for name, floor in witnesses.items():
        if hits[name] is not None and hits[name] <= floor:
            failures.append(f"INCONCLUSIVE: {name}: no warm image-block witness ({hits[name]})")
    return failures


def _timestamp(row):
    return datetime.strptime(row["timestamp"], "%m-%d-%Y %H:%M:%S.%f").timestamp()


def rank_evidence(iterations, windows, enable_attention_dp, margin=2.0):
    """Require each rank to prefill its cold image prompt inside the window it was pinned to.

    iterations: rows of the engine's iteration stats. windows: rank -> (start, end, prompt
    tokens of the cold request sent alone at the start of the window).
    """
    seen = set()
    for row in iterations:
        rank = row.get("attentionDpRank", None if enable_attention_dp else 0)
        if rank not in windows or "timestamp" not in row:
            continue
        start, end, tokens = windows[rank]
        stats = row.get("inflightBatchingStats") or {}
        if (
            start - margin <= _timestamp(row) <= end + margin
            and stats.get("numContextRequests", 0) >= 1
            and stats.get("numCtxTokens", 0) >= tokens
        ):
            seen.add(rank)
    return [
        f"INCONCLUSIVE: rank {rank} shows no image prefill in its pinned window"
        for rank in sorted(set(windows) - seen)
    ]


@contextmanager
def _openai_client(llm):
    """Borrow the test's engine and own only the server's client and thread pools."""
    from fastapi.testclient import TestClient

    from tensorrt_llm.inputs.media_io import BaseMediaIO
    from tensorrt_llm.serve.openai_server import OpenAIServer

    previous_executor = BaseMediaIO._executor
    server = OpenAIServer(
        llm,
        model=str(llm.args.model),
        tool_parser=None,
        server_role=None,
        metadata_server_cfg=None,
    )
    try:
        # The ASGI lifespan owns engine shutdown; this smoke borrows an engine
        # that the remaining accuracy stages still need.
        with closing(TestClient(server.app)) as client:
            yield server, client
    finally:
        server._input_proc_executor.shutdown(wait=True)
        server._media_load_executor.shutdown(wait=True)
        BaseMediaIO._executor = previous_executor


def check_structured(name, text, failures):
    """Fail unless ``text`` is a JSON object of ``_COLOR_SCHEMA``; a wrong color is inconclusive."""
    try:
        answer = json.loads(text)
    except json.JSONDecodeError:
        failures.append(f"FAIL: {name}: not JSON: {text!r}")
        return
    if (
        not isinstance(answer, dict)
        or set(answer) != {"color"}
        or answer["color"] not in _COLOR_SCHEMA["properties"]["color"]["enum"]
    ):
        failures.append(f"FAIL: {name}: outside the schema: {text!r}")
    elif answer["color"] != "red":
        failures.append(f"INCONCLUSIVE: {name}: {text!r}")


def evaluate(llm, profile, out: Path):
    """Run the panel on every attention-DP rank and return its result with failures labelled."""
    img, img_end = image_tokens(llm)
    red = shape("circle", PAIR_SIZE, "red")
    blue = shape("square", PAIR_SIZE, "blue")
    green = shape("triangle", PAIR_SIZE, "green")
    tall_blue = shape("square", (168, 280), "blue")
    fixtures = Path(__file__).parents[3] / "test_input_files"
    merlion = Image.open(fixtures / "merlion.png").convert("RGB")
    hockey = Image.open(fixtures / "pexels-ron-lach-8975010.jpg").convert("RGB")
    full = profile == "full"
    boundary = (
        [
            ("large", shape("circle", (1540, 1540), "red"), "red"),
            # Not blue: on this bar blue nearly ties with purple, for the reference implementation too.
            ("landscape", shape("square", (1540, 112), "yellow"), "yellow"),
            ("portrait", shape("triangle", (112, 1540), "green"), "green"),
        ]
        if full
        else []
    )
    aggregate = (
        [shape("circle", (1540, 1540), "red") for _ in range(9)]
        + [shape("square", (1540, 1540), "blue")]
        if full
        else []
    )
    named = [("red", red), ("blue", blue), ("green", green), ("tall-blue", tall_blue)]
    named += [("merlion", merlion), ("hockey", hockey)]
    named += [(name, image) for name, image, _ in boundary]
    named += [(f"aggregate-{index}", image) for index, image in enumerate(aggregate)]
    manifest = {
        name: {"size": list(image.size), "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
        for name, image in named
    }
    (out / "mm_images.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def single(image, question="What color is the large shape? Answer one color word."):
        messages = [
            {
                "role": "user",
                "content": [text_part(_PREFIX), image_part(image), text_part(question)],
            }
        ]
        return native_prompt(llm, messages, [image])

    def pair(first, second, question=" Name the colors of the large shapes in order, two words."):
        content = [text_part("First image: "), image_part(first)]
        content += [text_part(" Second image: "), image_part(second), text_part(question)]
        messages = [{"role": "user", "content": content}]
        return messages, native_prompt(llm, messages, [first, second])

    a, b = single(red), single(blue)
    pair_messages, red_blue = pair(red, blue)
    _, red_green = pair(red, green)
    _, blue_red = pair(blue, red)
    # The constructions below rely on token-identical prompts.
    assert a["prompt_token_ids"] == b["prompt_token_ids"]
    assert red_blue["prompt_token_ids"] == red_green["prompt_token_ids"]
    assert red_blue["prompt_token_ids"] == blue_red["prompt_token_ids"]
    (single_span,) = image_spans(a["prompt_token_ids"], img, img_end)
    first_span, second_span = image_spans(red_blue["prompt_token_ids"], img, img_end)
    _, sized = pair(red, tall_blue)
    conversation = pair_messages + [
        {"role": "assistant", "content": "Red blue."},
        {
            "role": "user",
            "content": [
                image_part(blue),
                text_part(
                    "Which previous image has a shape of this color, first or second? "
                    "Answer one word."
                ),
            ],
        },
    ]
    multiturn = native_prompt(llm, conversation, [red, blue, blue])
    text = native_prompt(llm, [{"role": "user", "content": "Reply with Paris only."}], [])
    natural = [
        (
            name,
            pair(first, second, " Which image shows hockey players, first or second? One word.")[1],
            expected,
        )
        for name, first, second, expected in (
            ("natural-merlion-hockey", merlion, hockey, ["second"]),
            ("natural-hockey-merlion", hockey, merlion, ["first"]),
        )
    ]

    attention_dp = llm.args.enable_attention_dp
    ranks = range(llm.args.tensor_parallel_size if attention_dp else 1)
    params = SamplingParams(
        max_tokens=ANSWER_TOKENS,
        temperature=0,
        return_perf_metrics=True,
        skip_special_tokens=False,
    )
    failures, windows, cases = [], {}, 0
    started = time.monotonic()
    with (out / "mm_predictions.jsonl").open("w") as stream:

        def run(batch, rank, phase, salts):
            """batch: (name, prompt, expected answer or None). Returns cached tokens per case."""
            nonlocal cases
            outputs = llm.generate(
                [prompt for _, prompt, _ in batch],
                params,
                cache_salt=salts,
                scheduling_params=(
                    [SchedulingParams(attention_dp_rank=rank, attention_dp_relax=False)]
                    * len(batch)
                    if attention_dp
                    else None
                ),
                use_tqdm=False,
            )
            hits = []
            for (name, prompt, expected), salt, output in zip(batch, salts, outputs, strict=True):
                completion = output.outputs[0]
                if list(output.prompt_token_ids) != prompt["prompt_token_ids"]:
                    failures.append(f"FAIL: rank {rank} {name}: prompt changed or truncated")
                hit = output.cached_tokens if completion.request_perf_metrics else None
                passed = expected is None or answer_matches(final_answer(completion.text), expected)
                if not passed:
                    failures.append(f"INCONCLUSIVE: rank {rank} {name}: {completion.text!r}")
                record = {
                    "name": name,
                    "phase": phase,
                    "attention_dp_rank": rank,
                    "cache_salt": salt,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "cached_tokens": hit,
                    "text": completion.text,
                    "output_token_ids": completion.token_ids,
                    "expected": expected,
                    "passed": passed,
                }
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                hits.append(hit)
                cases += 1
            return hits

        for rank in ranks:
            window_start = time.time()
            salt = f"mm/{rank}"
            # A(s1) -> B(s1) -> A(s1) warm -> A(s2): B differs from A only in its image.
            hits = {}
            for name, prompt, color, namespace in (
                ("red-cold", a, "red", f"{salt}/pair"),
                ("blue-same-text", b, "blue", f"{salt}/pair"),
                ("red-warm", a, "red", f"{salt}/pair"),
                ("red-fresh-salt", a, "red", f"{salt}/fresh"),
            ):
                hits[name] = run([(name, prompt, [color])], rank, "pair", [namespace])[0]
            length = len(a["prompt_token_ids"])
            failures.extend(
                f"rank {rank} {failure}"
                for failure in pair_failures(
                    hits,
                    bounds={
                        "red-cold": 0,
                        "blue-same-text": single_span[0],
                        "red-warm": length - 1,
                        "red-fresh-salt": 0,
                    },
                    witnesses={"red-warm": single_span[0]},
                )
            )
            # Same first image, then a different companion or the swapped order.
            companion, order = {}, f"{salt}/order"
            for name, prompt, expected in (
                ("pair-red-blue", red_blue, ["red", "blue"]),
                ("pair-red-green", red_green, ["red", "green"]),
                ("pair-blue-red", blue_red, ["blue", "red"]),
            ):
                (companion[name],) = run([(name, prompt, expected)], rank, "pair", [order])
            failures.extend(
                f"rank {rank} {failure}"
                for failure in pair_failures(
                    companion,
                    bounds={
                        "pair-red-blue": 0,
                        "pair-red-green": second_span[0],
                        "pair-blue-red": first_span[0],
                    },
                    witnesses={"pair-red-green": first_span[0]},
                )
            )
            # Content and order with sizes, turns and photos, each in its own namespace.
            for name, prompt, expected in [
                ("pair-different-sizes", sized, ["red", "blue"]),
                ("multiturn", multiturn, ["second"]),
                ("text", text, ["paris"]),
                *natural,
            ]:
                run([(name, prompt, expected)], rank, "content", [f"{salt}/{name}"])
            # Mixed batch: warm anchors keep their namespace, every new request gets its own.
            new = [("text", text, ["paris"]), ("pair-different-sizes", sized, ["red", "blue"])]
            new += natural
            batch, salts = [], []
            for index, case in enumerate(new):
                anchor = (
                    ("red-warm", a, ["red"], f"{salt}/pair")
                    if index % 2 == 0
                    else ("pair-red-blue", red_blue, ["red", "blue"], order)
                )
                batch += [anchor[:3], case]
                salts += [anchor[3], f"{salt}/mixed/{index}"]
            mixed = run(batch, rank, "mixed", salts)
            if any(hit is None for hit in mixed):
                failures.append(f"INCONCLUSIVE: rank {rank} mixed: missing cache telemetry")
            elif any(mixed[1::2]):
                failures.append(f"FAIL: rank {rank} mixed: new requests reused cached prefixes")
            elif not any(mixed[::2]):
                failures.append(f"INCONCLUSIVE: rank {rank} mixed: no warm hits")
            if full:
                for name, image, color in boundary:
                    run([(name, single(image), [color])], rank, "boundary", [f"{salt}/{name}"])
                messages = [
                    {
                        "role": "user",
                        "content": [
                            *[image_part(image) for image in aggregate],
                            text_part(
                                "What color is the large shape in the LAST image? "
                                "Answer one color word."
                            ),
                        ],
                    }
                ]
                prompt = native_prompt(llm, messages, aggregate)
                assert len(prompt["prompt_token_ids"]) >= 30000
                run([("aggregate-32k", prompt, ["blue"])], rank, "boundary", [f"{salt}/aggregate"])
                # Sustained batches of 16, each request in its own namespace.
                for iteration in range(3):
                    namespaces = [f"{salt}/sustained/{iteration}/{index}" for index in range(16)]
                    hits = run((new * 4)[:16], rank, f"sustained-{iteration}", namespaces)
                    if any(hit for hit in hits if hit is not None):
                        failures.append(f"FAIL: rank {rank} sustained-{iteration}: reused prefixes")
            windows[rank] = (window_start, time.time(), length)

        if full:
            with _openai_client(llm) as (server, client):
                for name, messages, expected in [
                    (
                        "openai-single",
                        [
                            {
                                "role": "user",
                                "content": [
                                    image_part(red),
                                    text_part("What color is the large shape? One color word."),
                                ],
                            }
                        ],
                        ["red"],
                    ),
                    ("openai-pair", pair_messages, ["red", "blue"]),
                ]:
                    response = client.post(
                        "/v1/chat/completions",
                        json={
                            "model": server.model,
                            "messages": messages,
                            "max_tokens": ANSWER_TOKENS,
                            "temperature": 0,
                            "skip_special_tokens": False,
                            "chat_template_kwargs": {"reasoning_effort": "low"},
                        },
                    )
                    row = {
                        "name": name,
                        "status_code": response.status_code,
                        "body": response.json(),
                    }
                    stream.write(json.dumps(row) + "\n")
                    if response.status_code != 200:
                        failures.append(f"FAIL: {name}: HTTP {response.status_code}")
                        continue
                    answer = final_answer(row["body"]["choices"][0]["message"]["content"])
                    if not answer_matches(answer, expected):
                        failures.append(f"INCONCLUSIVE: {name}: {answer!r}")
                # response_format is stated in the prompt's model settings and enforced.
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": server.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    image_part(red),
                                    text_part("What color is the large shape?"),
                                ],
                            }
                        ],
                        "max_tokens": 64,
                        "temperature": 0,
                        "reasoning_effort": "low",
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "color",
                                "schema": _COLOR_SCHEMA,
                                "strict": True,
                            },
                        },
                    },
                )
                row = {
                    "name": "openai-structured",
                    "status_code": response.status_code,
                    "body": response.json(),
                }
                stream.write(json.dumps(row) + "\n")
                if response.status_code != 200:
                    failures.append(f"FAIL: openai-structured: HTTP {response.status_code}")
                else:
                    check_structured(
                        "openai-structured",
                        row["body"]["choices"][0]["message"]["content"],
                        failures,
                    )

        # Structured outputs are on by default: guided decoding holds the answer to the schema.
        if llm.args.guided_decoding_backend != "xgrammar":
            failures.append(
                f"FAIL: guided decoding backend {llm.args.guided_decoding_backend!r}, not xgrammar"
            )
        else:
            output = llm.generate(
                single(red, question="What color is the large shape? Answer in JSON."),
                SamplingParams(
                    max_tokens=64,
                    temperature=0,
                    guided_decoding=GuidedDecodingParams(json=_COLOR_SCHEMA),
                ),
                use_tqdm=False,
            )
            text = output.outputs[0].text
            stream.write(json.dumps({"name": "structured", "text": text}) + "\n")
            check_structured("structured", text, failures)
            cases += 1

    result = {
        "seconds": time.monotonic() - started,
        "cases": cases,
        "ranks": list(ranks),
        "windows": windows,
        "failures": failures,
    }
    (out / "mm_functional_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
