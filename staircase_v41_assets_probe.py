# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the run-1 artifacts this run reuses, before anything depends on them.

Run 1 left a converted model-parallel-4 checkpoint, a frozen GSM8K request set,
five frozen greedy fixtures and a full-1319 scored record. Reusing them saves
hours; reusing them *unverified* would silently move the anchor. This probe is
the gate: every artifact is checked against the manifest it carries, and against
the accuracy stanza in the repository that cites it.

What it does and does not re-derive, on purpose:

  * The two small tokenizer files and the four HF identity files are re-hashed
    in full -- seconds.
  * The four 130 GB converted rank shards ARE hashed in full against
    ``anchor/ckpt-mp4-sha256.txt``, but not on every invocation: ``--hash-shards``
    recomputes all four and writes a receipt, and every ordinary run then
    re-checks that receipt. The receipt records each shard's size and mtime
    alongside its digest, so it is invalidated by any write to a shard rather
    than trusted indefinitely -- and a missing or stale receipt is a FAILURE
    here, not a skipped check. Measured on this filesystem: 486.8 GiB in
    87.5 s with the four hashed concurrently, against the 1,086 s conversion
    recorded when it did them one after another.
  * Each shard's safetensors header is parsed as well, and its tensor inventory
    compared against the geometry the config declares. That is a different check
    from the digest: the digest says the bytes are the ones conversion wrote,
    the header says those bytes describe the model this run expects.
  * The request set and the fixture set are re-digested from their own contents
    with the same construction the artifacts declare, so an edited, truncated or
    appended-to file stops matching the record that cites it.

Exits 0 only when every artifact validates. It imports no GPU and no
tensorrt_llm; it reads files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

WORK = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/staircase-v41"
)
HF = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/models/DeepSeek-V4.1-Flash"
)
REPO = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/TensorRT-LLM"
)
ACCURACY_YAML = REPO / "tensorrt_llm/_torch/staircase/references/accuracy.yaml"

#: Recorded in the accuracy stanza this run must not move. Restated here so the
#: probe fails when the stanza and the artifacts disagree, rather than when only
#: one of them changes.
ANCHOR_SCORE = 87.1114
ANCHOR_TOL = 5.0
ANCHOR_N = 1319
ANCHOR_FILTER = "exact_match,flexible-extract"
PROMPT_TOKENS_SHA256 = "fe7810168018c866bb07d0d3afd331ab1f673f937519f7d185e0d29f1d657104"
FIXTURES_SHA256 = "6eed4eca2013d24989490463959dc801e93616e1c5dbb83bebd1834d24919c36"
HF_IDENTITY = {
    "config.json": "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
    "model.safetensors.index.json": "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8",
    "tokenizer.json": "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
    "tokenizer_config.json": "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
}
#: The reference-only pins the anchor was measured under. The image ships a
#: newer apache-tvm-ffi that satisfies tilelang's declared range and then dies
#: registering ``ir.DictAttrs``, so the pin is load-bearing rather than tidy.
REFENV_PINS = {"apache-tvm-ffi": "0.1.8.post2", "tilelang": "0.1.8"}

#: Where ``--hash-shards`` records what it measured. Kept beside the manifest it
#: certifies rather than in the repo: it is a statement about these 487 GiB of
#: bytes on this filesystem, not about the source tree.
PAYLOAD_RECEIPT = WORK / "anchor/ckpt-mp4-payload-verified.json"

_checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _checks.append((name, bool(ok), detail))
    print(f"[{'ok  ' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)
    return bool(ok)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def safetensors_header(path: Path) -> dict:
    """Read a safetensors header without mapping the payload.

    The first 8 bytes are a little-endian u64 header length; the header itself
    is JSON naming every tensor with its dtype, shape and byte range. Parsing it
    costs a few MiB of read on a 130 GB file.
    """
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def read_shard_manifest() -> dict[str, str]:
    """Parse ``anchor/ckpt-mp4-sha256.txt`` into ``{filename: sha256}``.

    Conversion wrote a trailing ``(<seconds>)`` column on the shard lines, so
    only the first two fields are read.
    """
    recorded: dict[str, str] = {}
    path = WORK / "anchor/ckpt-mp4-sha256.txt"
    if not path.exists():
        return recorded
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            recorded[parts[1]] = parts[0]
    return recorded


def shard_names(world: int = 4) -> list[str]:
    return [f"model{rank}-mp{world}.safetensors" for rank in range(world)]


def hash_shards(world: int = 4) -> int:
    """Recompute all four converted shards' sha256 and record what was measured.

    Run explicitly (``--hash-shards``) because it reads 487 GiB. The four are
    hashed concurrently: ``hashlib`` releases the GIL around each update, so
    threads overlap the reads that dominate here, and the four files are
    independent. Measured: 83-88 s per shard with all four in flight, i.e. the
    whole set in the time one of them takes. The receipt carries size and mtime
    per shard so an ordinary run can tell "verified" from "verified before
    someone rewrote it".
    """
    manifest = read_shard_manifest()
    ckpt = WORK / "ckpt-mp4"
    missing = [n for n in shard_names(world) if not (ckpt / n).exists()]
    if missing:
        print(f"cannot hash: missing {missing}", flush=True)
        return 1

    def one(name: str) -> dict:
        path = ckpt / name
        stat = path.stat()
        started = time.time()
        digest = sha256_file(path)
        return {
            "name": name,
            "sha256": digest,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "elapsed_s": round(time.time() - started, 1),
            "manifest_sha256": manifest.get(name),
            "matches_manifest": digest == manifest.get(name),
        }

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=world) as pool:
        results = list(pool.map(one, shard_names(world)))
    elapsed = round(time.time() - started, 1)
    total = sum(r["size"] for r in results)
    for r in results:
        print(
            f"[{'ok  ' if r['matches_manifest'] else 'FAIL'}] {r['name']} "
            f"{r['sha256'][:16]}... vs manifest {str(r['manifest_sha256'])[:16]}... "
            f"({r['size'] / 2**30:.1f} GiB in {r['elapsed_s']}s)",
            flush=True,
        )
    payload = {
        "kind": "deepseek-v41-flash-ckpt-mp4-payload/1",
        "host": socket.gethostname(),
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": elapsed,
        "total_bytes": total,
        "shards": {r["name"]: r for r in results},
    }
    PAYLOAD_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    PAYLOAD_RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    bad = [r["name"] for r in results if not r["matches_manifest"]]
    print(
        f"\nhashed {total / 2**30:.1f} GiB in {elapsed}s -> {PAYLOAD_RECEIPT}",
        flush=True,
    )
    if bad:
        print(f"PAYLOAD MISMATCH: {' '.join(bad)}", flush=True)
        return 1
    print("ALL FOUR SHARD DIGESTS MATCH THE MANIFEST", flush=True)
    return 0


def validate_shard_payload(world: int = 4) -> None:
    """Check the recorded full-payload verification, and that it still applies.

    Three ways this fails, all of them loud: no receipt was ever recorded; the
    receipt records a digest that disagrees with the manifest; or a shard has
    been written since it was recorded, which makes the receipt a statement
    about bytes that are no longer there.
    """
    manifest = read_shard_manifest()
    if not check(
        "ckpt-mp4 payload receipt present",
        PAYLOAD_RECEIPT.exists(),
        str(PAYLOAD_RECEIPT) if PAYLOAD_RECEIPT.exists() else "run with --hash-shards",
    ):
        for name in shard_names(world):
            check(f"ckpt-mp4 {name} payload sha256", False, "no recorded verification")
        return
    receipt = json.loads(PAYLOAD_RECEIPT.read_text())
    recorded = receipt.get("shards", {})
    check(
        "ckpt-mp4 payload receipt provenance",
        True,
        f"{receipt.get('host', '?')} at {receipt.get('verified_utc', '?')} "
        f"in {receipt.get('elapsed_s', '?')}s",
    )
    for name in shard_names(world):
        entry = recorded.get(name)
        if entry is None:
            check(f"ckpt-mp4 {name} payload sha256", False, "absent from the receipt")
            continue
        want = manifest.get(name)
        check(
            f"ckpt-mp4 {name} payload sha256",
            bool(want) and entry.get("sha256") == want,
            f"{str(entry.get('sha256'))[:16]}... vs manifest {str(want)[:16]}...",
        )
        path = WORK / "ckpt-mp4" / name
        if not path.exists():
            check(f"ckpt-mp4 {name} unchanged since verification", False, "missing")
            continue
        stat = path.stat()
        same = stat.st_size == entry.get("size") and stat.st_mtime_ns == entry.get("mtime_ns")
        check(
            f"ckpt-mp4 {name} unchanged since verification",
            same,
            f"{stat.st_size} bytes, mtime {stat.st_mtime_ns}"
            if same
            else "size/mtime moved -- rerun --hash-shards",
        )


def validate_hf_identity() -> None:
    for name, want in HF_IDENTITY.items():
        path = HF / name
        if not path.exists():
            check(f"hf identity {name} present", False, "missing")
            continue
        got = sha256_file(path)
        check(f"hf identity {name}", got == want, f"{got[:16]}... vs {want[:16]}...")


def validate_converted_checkpoint() -> None:
    manifest_path = WORK / "anchor/ckpt-mp4-sha256.txt"
    if not check("ckpt-mp4 manifest present", manifest_path.exists(), str(manifest_path)):
        return
    recorded = read_shard_manifest()
    validate_shard_payload()
    ckpt = WORK / "ckpt-mp4"
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path = ckpt / name
        if not path.exists():
            check(f"ckpt-mp4 {name} present", False, "missing")
            continue
        got = sha256_file(path)
        check(f"ckpt-mp4 {name}", got == recorded.get(name), f"{got[:16]}...")
    # The tokenizer the reference leg loads must be the checkpoint's own.
    check(
        "ckpt-mp4 tokenizer is the checkpoint's",
        recorded.get("tokenizer.json") == HF_IDENTITY["tokenizer.json"],
        "manifest agrees with the HF identity",
    )

    cfg = json.loads((HF / "inference/config.json").read_text())
    world = 4
    want_layers = cfg["n_layers"]
    for rank in range(world):
        path = ckpt / f"model{rank}-mp{world}.safetensors"
        if not check(f"ckpt-mp4 rank {rank} present", path.exists(), str(path)):
            continue
        if f"model{rank}-mp{world}.safetensors" not in recorded:
            check(f"ckpt-mp4 rank {rank} in manifest", False, "not listed")
        try:
            header = safetensors_header(path)
        except Exception as exc:  # noqa: BLE001 - an unreadable header is the finding
            check(f"ckpt-mp4 rank {rank} header parses", False, repr(exc))
            continue
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        size = path.stat().st_size
        end = max(v["data_offsets"][1] for v in tensors.values())
        check(
            f"ckpt-mp4 rank {rank} header parses",
            True,
            f"{len(tensors)} tensors, {size / 2**30:.1f} GiB",
        )
        # A truncated or half-written shard shows up here: the header's own
        # last byte offset has to fit inside the file.
        check(
            f"ckpt-mp4 rank {rank} payload complete",
            8 + len(json.dumps(header)) <= size and end <= size,
            f"header end {end} <= file {size}",
        )
        layers = {int(k.split(".")[1]) for k in tensors if k.startswith("layers.")}
        check(
            f"ckpt-mp4 rank {rank} backbone layers",
            layers == set(range(want_layers)) or layers >= set(range(want_layers)),
            f"{len(layers)} layer ids, want {want_layers} backbone",
        )

        # Expert ids have to be split by namespace before they mean anything.
        # ``layers.*`` are the 384 backbone experts, 96 per rank; ``mtp.*`` are
        # DSpark's separate 128-expert stage, 32 per rank, which this task
        # excludes. Measured on rank 1: the union is 128 ids starting at 32,
        # which looks like a wrong window until the two namespaces are
        # separated -- backbone 96..191 and DSpark 32..63.
        def window(prefix: str) -> set[int]:
            return {
                int(k.split(".experts.")[1].split(".")[0])
                for k in tensors
                if ".experts." in k and k.startswith(prefix)
            }

        per_rank = cfg["n_routed_experts"] // world
        backbone = window("layers.")
        check(
            f"ckpt-mp4 rank {rank} owns its backbone expert window",
            len(backbone) == per_rank and min(backbone) == rank * per_rank,
            f"{len(backbone)} experts at {min(backbone)}..{max(backbone)}, want {per_rank} at {rank * per_rank}",
        )
        draft = window("mtp.")
        per_rank_draft = cfg["dspark_n_routed_experts"] // world
        check(
            f"ckpt-mp4 rank {rank} DSpark experts present and out of scope",
            len(draft) == per_rank_draft and min(draft) == rank * per_rank_draft,
            f"{len(draft)} at {min(draft)}..{max(draft)} -- an intentional non-load for this task",
        )


def validate_requests() -> None:
    path = WORK / "anchor/requests.jsonl"
    if not check("requests.jsonl present", path.exists(), str(path)):
        return
    with open(path) as f:
        meta = json.loads(f.readline())["__meta__"]
        records = [json.loads(line) for line in f]
    digest = hashlib.sha256()
    for r in records:
        digest.update(json.dumps(r["prompt_token_ids"]).encode())
    check(
        "requests digest self-consistent",
        digest.hexdigest() == meta["prompt_tokens_sha256"],
        digest.hexdigest()[:16],
    )
    check(
        "requests digest is the recorded one", meta["prompt_tokens_sha256"] == PROMPT_TOKENS_SHA256
    )
    check(
        "requests count",
        len(records) == ANCHOR_N and meta["n_requests"] == ANCHOR_N,
        str(len(records)),
    )
    check("requests doc_ids unique", len({r["doc_id"] for r in records}) == len(records))
    check("requests protocol task", meta["task"] == "gsm8k", meta["task"])
    check("requests 5-shot", meta["num_fewshot"] == 5, str(meta["num_fewshot"]))
    check("requests seed 0", meta["random_seed"] == 0, str(meta["random_seed"]))
    check(
        "requests max_input_length", meta["max_input_length"] == 4096, str(meta["max_input_length"])
    )
    check(
        "requests max_output_length",
        meta["max_output_length"] == 256,
        str(meta["max_output_length"]),
    )
    longest = max(r["n_prompt_tokens"] for r in records)
    check(
        "longest prompt fits max_input_length",
        longest <= meta["max_input_length"],
        f"{longest} tokens",
    )

    # The same set rebuilt under the repo venv the target leg runs in must hash
    # identically; run 1 produced that second artifact and the equality is what
    # makes the two legs comparable rather than assumed comparable.
    repoenv = WORK / "anchor/requests-repoenv.jsonl"
    if repoenv.exists():
        with open(repoenv) as f:
            meta2 = json.loads(f.readline())["__meta__"]
        # That meta predates the ``software`` block anchor.py records now, so
        # the environments are identified by what both artifacts carry.
        ref_env = meta.get("software", {}).get("transformers", meta.get("lm_eval_version", "?"))
        tgt_env = meta2.get("software", {}).get("transformers", meta2.get("lm_eval_version", "?"))
        check(
            "repo-env request rebuild agrees",
            meta2["prompt_tokens_sha256"] == meta["prompt_tokens_sha256"],
            f"identical digest across the two environments ({ref_env} / {tgt_env})",
        )
        check("repo-env request count", meta2["n_requests"] == ANCHOR_N, str(meta2["n_requests"]))
    else:
        check("repo-env request rebuild present", False, "requests-repoenv.jsonl missing")


def validate_fixtures() -> None:
    path = WORK / "anchor/fixtures-frozen-frozen.json"
    if not check("frozen fixtures present", path.exists(), str(path)):
        return
    payload = json.loads(path.read_text())
    digest = hashlib.sha256()
    for e in payload["frozen"]:
        digest.update(json.dumps([e["prompt"], e["generated_token_ids"]]).encode())
    check(
        "fixtures digest self-consistent",
        digest.hexdigest() == payload["fixtures_sha256"],
        digest.hexdigest()[:16],
    )
    check("fixtures digest is the recorded one", payload["fixtures_sha256"] == FIXTURES_SHA256)
    check("fixtures count", len(payload["frozen"]) == 5, str(len(payload["frozen"])))
    check("fixtures greedy", payload["decoding"].startswith("greedy"), payload["decoding"])
    check("fixtures batch size 1", payload["batch_size"] == 1, str(payload["batch_size"]))
    weak = [e["prompt"] for e in payload["frozen"] if e["n_variants"] != 1 or e["n_hosts"] < 4]
    check(
        "every fixture stable across >=4 hosts",
        not weak,
        f"{len(payload['frozen'])} fixtures" if not weak else str(weak),
    )
    for e in payload["frozen"]:
        if len(e["generated_token_ids"]) != e["n_generated_tokens"]:
            check(f"fixture {e['prompt'][:24]!r} token count", False, "declared length disagrees")
    check(
        "fixture token counts",
        True,
        ", ".join(str(e["n_generated_tokens"]) for e in payload["frozen"]),
    )


def validate_reference_record() -> None:
    path = WORK / "anchor/reference-full1319.json"
    if not check("full-1319 record present", path.exists(), str(path)):
        return
    payload = json.loads(path.read_text())
    check("full-1319 sample count", payload["n_samples"] == ANCHOR_N, str(payload["n_samples"]))
    check(
        "full-1319 prompt digest",
        payload["prompt_tokens_sha256"] == PROMPT_TOKENS_SHA256,
        payload["prompt_tokens_sha256"][:16],
    )
    score = payload["scores"][ANCHOR_FILTER]
    check("full-1319 score is the anchor", round(score, 4) == ANCHOR_SCORE, f"{score:.4f}")
    check(
        "full-1319 samples present",
        len(payload["samples"]) == ANCHOR_N,
        str(len(payload["samples"])),
    )
    sample = payload["samples"][0]
    for field in ("doc_id", "prompt", "prompt_token_ids", "generated_token_ids", "target"):
        if field not in sample:
            check(f"full-1319 per-sample {field}", False, "absent")
    check("full-1319 per-sample fields", True, ", ".join(sorted(sample)))


def validate_refenv() -> None:
    path = WORK / "refenv/PINS.txt"
    if not check("refenv pins present", path.exists(), str(path)):
        return
    pins = {}
    for line in path.read_text().splitlines():
        if "==" in line:
            name, version = line.strip().split("==", 1)
            pins[name] = version
    for name, want in REFENV_PINS.items():
        check(f"refenv pin {name}", pins.get(name) == want, f"{pins.get(name)} vs {want}")
    check("refenv venv present", (WORK / "refenv/venv/bin/python3").exists())


def validate_accuracy_stanza() -> None:
    if not check("accuracy.yaml present", ACCURACY_YAML.exists(), str(ACCURACY_YAML)):
        return
    # Parsed without yaml so the probe has no dependency the container might
    # not carry; the stanza is a flat block with a known shape.
    text = ACCURACY_YAML.read_text()
    key = "deepseek-ai/DeepSeek-V4.1-Flash:"
    if not check("V4.1 anchor stanza present", key in text):
        return
    block = text[text.index(key) :]
    fields = {}
    for line in block.splitlines()[1:]:
        if line and not line.startswith(("  ", "\t")):
            break
        stripped = line.strip()
        if ":" in stripped and not stripped.startswith("#"):
            k, _, v = stripped.partition(":")
            fields.setdefault(k.strip(), v.strip())
    check("anchor score", fields.get("score") == f"{ANCHOR_SCORE}", fields.get("score", "<absent>"))
    check("anchor tol", fields.get("tol") == f"{ANCHOR_TOL}", fields.get("tol", "<absent>"))
    check(
        "anchor n_samples",
        fields.get("n_samples") == str(ANCHOR_N),
        fields.get("n_samples", "<absent>"),
    )
    check(
        "anchor target_ckpt",
        fields.get("target_ckpt") == "v41_flash",
        fields.get("target_ckpt", "<absent>"),
    )
    check(
        "anchor source is the reference implementation",
        fields.get("source") == "reference-implementation",
        fields.get("source", "<absent>"),
    )
    check("anchor task", fields.get("task") == "gsm8k", fields.get("task", "<absent>"))
    check(
        "anchor scores_filter",
        fields.get("scores_filter") == ANCHOR_FILTER,
        fields.get("scores_filter", "<absent>"),
    )
    release_bar = ANCHOR_SCORE - ANCHOR_TOL
    check(
        "release bar", abs(release_bar - 82.1114) < 1e-9, f"measured must clear {release_bar:.4f}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="validate the run-1 artifacts this run reuses")
    p.add_argument(
        "--hash-shards",
        action="store_true",
        help="recompute all four 130 GB converted shard sha256s and record the receipt",
    )
    args = p.parse_args()
    if args.hash_shards:
        print(f"work={WORK}\nhashing the four converted shards in full\n", flush=True)
        return hash_shards()

    print(f"work={WORK}\nhf={HF}\nrepo={REPO}\npid={os.getpid()}\n", flush=True)
    validate_hf_identity()
    validate_converted_checkpoint()
    validate_requests()
    validate_fixtures()
    validate_reference_record()
    validate_refenv()
    validate_accuracy_stanza()
    failed = [name for name, ok, _ in _checks if not ok]
    print(f"\n{len(_checks) - len(failed)} of {len(_checks)} asset checks passed", flush=True)
    if failed:
        print("FAILED: " + "; ".join(failed), flush=True)
        return 1
    print("ASSETS VALIDATE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
