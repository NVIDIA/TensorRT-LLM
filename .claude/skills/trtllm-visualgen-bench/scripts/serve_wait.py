#!/usr/bin/env python3
"""Wait for a trtllm-serve VisualGen server, then verify it came up as configured.

  serve_wait.py --log server.log --config server.yaml [--port 8000] [--stall 300]

Exits 0 only when /health returns 200 AND the log agrees with the config. Exits
non-zero on a stall (no new log line for --stall seconds), on server death, or on
a mismatch -- a server that loads happily at the wrong precision, the wrong
attention backend or an unwarmed shape is the failure this exists to catch, and
none of those raise on their own.
"""

import argparse
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml


def health(port, host):
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=5) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


# Fatal signatures that never produce a Python traceback. Deliberately narrow --
# a false positive aborts a healthy 20-minute startup.
FATAL = re.compile(
    r"^Killed$|Segmentation fault|core dumped|Fatal Python error|"
    r"terminate called after throwing|CUDA out of memory|torch\.OutOfMemoryError|"
    r"srun: error:|Exited with exit code|ncclInternalError|ncclUnhandledCudaError|"
    r"CUDA error: out of memory|what\(\):|"
    # C++/NCCL/CUDA fatals, which abort without unwinding into a Python traceback.
    r"\[FATAL\]|Fatal error|NCCL WARN.*(?:unhandled|Fatal)|CUDA_ERROR_|"
    r"cudaError|Assertion .* failed",
    re.MULTILINE,
)


def port_open(host, port):
    with socket.socket() as s:
        s.settimeout(3)
        return s.connect_ex((host, port)) == 0


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def check(log, cfg):
    """[(level, message)] comparing what the config asked for to what the log reports."""
    out = []
    want_pc = cfg.get("parallel_config") or {}
    a2 = want_pc.get("attn2d_size") or [1, 1]
    cp = max(a2[0] * a2[1], want_pc.get("ring_size", 1), 1)
    want_workers = (
        want_pc.get("cfg_size", 1) * cp * want_pc.get("ulysses_size", 1) * want_pc.get("tp_size", 1)
    )

    m = re.search(r"World size:\s*(\d+)", log)
    if not m:
        out.append(("FAIL", "no 'World size:' line -- the server never finished loading"))
    elif int(m.group(1)) != want_workers:
        out.append(("FAIL", f"world size {m.group(1)} but parallel_config implies {want_workers}"))
    else:
        out.append(("ok", f"world size {want_workers}"))

    # `Quantization:` is logged only once a quant_algo resolved -- from the YAML, or
    # from the checkpoint's own metadata when the YAML says nothing. Its ABSENCE is
    # therefore the BF16 signal, but absence has two very different causes, and the
    # skipping warning is what separates "you configured it wrong" from "the loader
    # does not understand this checkpoint's format".
    want_q = (cfg.get("quant_config") or {}).get("quant_algo")
    got_q = re.search(r"Quantization:\s*(\S+)", log)
    got_dyn = re.search(r"Dynamic weight quant:\s*(\S+)", log)
    skipped = re.search(r"_quantization_metadata format '([^']+)' is not supported", log)
    mixed = "_quantization_metadata has mixed formats" in log
    if got_q:
        got = got_q.group(1)
        dyn = got_dyn.group(1) if got_dyn else "?"
        if want_q and got.upper() != str(want_q).upper():
            out.append(("FAIL", f"quantization {got} but config asked for {want_q}"))
        else:
            out.append(
                (
                    "ok",
                    f"quantization {got} (dynamic={dyn})"
                    + ("" if want_q else ", resolved from the checkpoint"),
                )
            )
    elif skipped:
        out.append(
            (
                "FAIL",
                f"the checkpoint declares format '{skipped.group(1)}' which "
                f"this build's loader does not map to a quant algo -- the "
                f"quantized weights were IGNORED and it is running BF16",
            )
        )
    elif mixed:
        out.append(
            (
                "FAIL",
                "checkpoint quantization metadata has mixed formats; it was "
                "skipped and this is running BF16",
            )
        )
    elif want_q:
        out.append(
            ("FAIL", f"config asked for {want_q} but no 'Quantization:' line -- it loaded BF16")
        )
    else:
        out.append(
            (
                "warn",
                "no quantization -- BF16. Correct only if you meant to serve a BF16 checkpoint",
            )
        )

    want_a = (cfg.get("attention_config") or {}).get("backend", "VANILLA")
    got_a = re.search(r"Attention backend:\s*(\S+)", log)
    if not got_a:
        out.append(("warn", "no 'Attention backend:' line"))
    elif got_a.group(1).rstrip("(,").upper() != str(want_a).upper():
        out.append(("FAIL", f"attention backend {got_a.group(1)} but config says {want_a}"))
    else:
        out.append(("ok", f"attention backend {got_a.group(1)}"))

    cc = cfg.get("compilation_config") or {}
    if "Skipping invalid warmup shape" in log:
        bad = re.findall(r"Skipping invalid warmup shape \(([^)]*)\)", log)
        out.append(
            (
                "FAIL",
                f"warmup shape(s) rejected and dropped: {bad} -- those "
                f"requests will compile inside the measured latency",
            )
        )
    elif "Warmup disabled (no warmup shapes)" in log:
        out.append(
            (
                "FAIL" if cc.get("resolutions") else "warn",
                "warmup disabled -- the first request of every shape pays compile",
            )
        )
    elif re.search(r"Warmup completed in", log):
        # The server prints the shapes it actually warmed as HxWxF; compare them
        # against the Cartesian product the config asked for, so a server that
        # warmed a different shape is not reported as covered.
        want = {
            (h, w, f)
            for h, w in (cc.get("resolutions") or [])
            for f in (cc.get("num_frames") or [1])
        }
        ran = set()
        for block in re.findall(r"Running warmup for \S+: \d+ shape\(s\) \[([^\]]*)\]", log):
            for shape in block.split(","):
                parts = shape.strip().split("x")
                if len(parts) == 3 and all(p.isdigit() for p in parts):
                    ran.add(tuple(int(p) for p in parts))
        missing = sorted(want - ran)
        if missing:
            out.append(
                (
                    "FAIL",
                    f"config asked to warm {missing} but the server warmed "
                    f"{sorted(ran)} -- those shapes compile inside the "
                    f"measured latency",
                )
            )
        else:
            done = re.search(r"Warmup completed in [\d.]+s", log).group(0).lower()
            out.append(("ok", f"{done}, covering {sorted(ran)}" if ran else done))
    elif not cc.get("skip_warmup"):
        out.append(("warn", "no 'Warmup completed' line"))

    if want_pc.get("parallel_vae_size", 1) > 1 and "Parallel VAE not supported" in log:
        out.append(
            (
                "FAIL",
                "parallel_vae_size set but the VAE type is unsupported -- running unparallelized",
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--config", type=Path, help="the --visual_gen_args yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--stall",
        type=int,
        default=300,
        help="fail if the log is silent this many seconds (default 300)",
    )
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--pid", type=int, help="server pid; exit the moment it disappears")
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text()) if a.config and a.config.is_file() else {}
    t0 = last_change = time.time()
    size, bound = -1, False
    while True:
        now = time.time()
        cur = a.log.stat().st_size if a.log.is_file() else -1
        if cur != size:
            size, last_change = cur, now

        if health(a.port, a.host):
            break

        text = a.log.read_text(errors="replace") if a.log.is_file() else ""
        if "Traceback (most recent call last)" in text:
            # Report the FIRST traceback: a worker dies, then the parent raises its
            # own "Worker died during initialization" and the shutdown chatter that
            # follows would otherwise be all a tail shows.
            lines = text.splitlines()
            start = next(
                i for i, ln in enumerate(lines) if "Traceback (most recent call last)" in ln
            )
            block = lines[start : start + 40]
            end = next(
                (
                    i
                    for i, ln in enumerate(block)
                    if i and re.match(r"^\w[\w.]*(Error|Exception)\b", ln)
                ),
                len(block) - 1,
            )
            print("\n".join(block[: end + 1]), file=sys.stderr)
            sys.exit("\nserver died during startup (first traceback above)")

        # Deaths that produce no traceback at all. Checked before the stall window so
        # a kill is reported as a kill in one poll, not as a stall five minutes later.
        m = FATAL.search(text)
        if m:
            print("\n".join(text.rstrip().splitlines()[-20:]), file=sys.stderr)
            sys.exit(f"\nserver died: matched {m.group(0)!r} (no traceback). Last lines above.")
        if a.pid and not pid_alive(a.pid):
            print("\n".join(text.rstrip().splitlines()[-20:]), file=sys.stderr)
            sys.exit(
                f"\npid {a.pid} is gone and /health never came up -- killed, "
                f"no traceback. Last lines above."
            )
        if port_open(a.host, a.port):
            bound = True
        elif bound:
            print("\n".join(text.rstrip().splitlines()[-20:]), file=sys.stderr)
            sys.exit(
                "\nthe listening socket disappeared -- the server was killed. Last lines above."
            )

        if now - last_change > a.stall:
            tail = text.rstrip().splitlines()[-15:]
            print("\n".join(tail), file=sys.stderr)
            sys.exit(
                f"\nSTALLED: no new log output for {int(now - last_change)}s "
                f"(limit {a.stall}). Last lines above."
            )
        if now - t0 > a.timeout:
            sys.exit(f"timed out after {a.timeout}s without /health returning 200")
        print(
            f"  [{int(now - t0):5d}s] loading, log {cur}B, quiet {int(now - last_change)}s",
            flush=True,
        )
        time.sleep(10)

    print(f"\nREADY after {int(time.time() - t0)}s")
    results = check(a.log.read_text(errors="replace"), cfg)
    for lvl, msg in results:
        print(f"  [{lvl:4s}] {msg}")
    fails = sum(1 for lvl, _ in results if lvl == "FAIL")
    print(f"\n{fails} mismatch(es)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
