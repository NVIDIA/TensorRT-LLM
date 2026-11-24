#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Put the proxy over one disaggregated deployment's servers and score a benchmark through it.

Usage (under the tree's interpreter, the client of ``infra/compute/steps.servers_script``)::

    disagg_driver.py --directory D --config C --model M --servers ctx0,gen0 <benchmark> [...]

It waits for each server's ``address.<name>`` in ``D`` (``exit.<name>`` says one ended), starts
``trtllm-serve disaggregated`` over them with the deployment's own config ``C``, and runs the
tree's ``trtllm-eval`` command of ``<benchmark>`` with an ``LLM`` that sends each prompt to
the proxy as ``tests/integration/defs/accuracy/test_disaggregated_serving.py`` does. One JSON
line behind ``MV2FLOW_SESSION`` says what happened. Exits 0 when the benchmark ran to its end.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

SESSION_MARKER = "MV2FLOW_SESSION"
DEFAULT_TIMEOUT_S = 7200
STOP_GRACE_S = 90
#: What the proxy must not read of the client's step.
_BORROWED = ("PMI", "PMIX", "OMPI", "MPI", "SLURM")
try:
    from . import serve_driver
except ImportError:  # run by path: the sibling module is on sys.path[0]
    import serve_driver  # type: ignore[no-redef]

runtime, wio, entry_point = serve_driver.runtime, serve_driver.wio, serve_driver.entry_point


def proxy_config(stated: dict[str, Any], model: str, ctx: list[str], gen: list[str]) -> dict:
    """The deployment's own config, with where the proxy listens, the model and the servers."""
    config = {**stated, "hostname": "0.0.0.0", "port": 0, "model": str(model)}  # nosec B104
    for side, urls in (("context_servers", ctx), ("generation_servers", gen)):
        config[side] = {**(stated.get(side) or {}), "urls": list(urls)}
    return config


def request_body(model: str, prompt: str, sampling: Any) -> dict[str, Any]:
    """One completion request, sampled as the engine samples the same parameters."""
    body: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if sampling is not None:
        knobs = (sampling.temperature, sampling.top_p, getattr(sampling, "top_k", None))
        body.update(
            max_tokens=sampling.max_tokens,
            n=sampling.n,
            best_of=sampling.best_of,
            # With none of them set the `LLM` is greedy, and the API would sample.
            temperature=0 if all(knob is None for knob in knobs) else sampling.temperature,
            top_p=sampling.top_p,
            stop=sampling.stop,
            seed=sampling.seed,
        )
        if knobs[2] is not None:
            body["top_k"] = knobs[2]
    return body


class EndpointLLM:
    """What an evaluator needs of an ``LLM``, answered by the endpoint at ``url``."""

    def __init__(self, url: str, model: str, tokenizer: Any) -> None:
        self.url, self.model, self.tokenizer, self.args = url, model, tokenizer, None
        self._pool = ThreadPoolExecutor(max_workers=16)

    def _complete(self, prompt: str, sampling: Any) -> Any:
        from tensorrt_llm.executor.result import GenerationResultBase
        from tensorrt_llm.llmapi import CompletionOutput, RequestOutput

        class Result(GenerationResultBase):
            """A finished generation built from an endpoint's answer."""

            def __init__(self, outputs: list) -> None:
                super().__init__(0, sampling)
                self._outputs = outputs
                self._streaming = False

            @property
            def outputs(self) -> list:
                return self._outputs

            def result(self):
                return self

        request = urllib.request.Request(
            f"{self.url}/v1/completions",
            data=json.dumps(request_body(self.model, prompt, sampling)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with serve_driver._DIRECT.open(request, timeout=3600) as reply:
            payload = json.loads(reply.read().decode("utf-8"))
        result = Result(
            [
                CompletionOutput(text=choice["text"], index=index)
                for index, choice in enumerate(payload["choices"])
            ]
        )
        output = RequestOutput._from_generation_result(result, prompt=prompt)
        output.result = result.result
        return output

    def generate_async(self, prompt: str, sampling_params: Any = None, streaming: bool = False):
        return self._pool.submit(self._complete, prompt, sampling_params)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


def _late(signum, frame):  # noqa: ARG001
    raise SystemExit("the timeout was up before the benchmark ended")


def score(url: str, model: str, name: str, arguments: list[str]) -> None:
    """``trtllm-eval``'s command of benchmark ``name``, its ``LLM`` the endpoint at ``url``."""
    from tensorrt_llm.commands.eval import main as trtllm_eval
    from tensorrt_llm.llmapi.tokenizer import load_hf_tokenizer
    from tensorrt_llm.logger import logger

    logger.set_level("info")  # as `trtllm-eval` sets it: the scores are logged
    llm = EndpointLLM(url, model, load_hf_tokenizer(model))
    trtllm_eval.commands[name].main(args=arguments, prog_name=name, obj=llm, standalone_mode=False)


def run(options: argparse.Namespace) -> int:
    """The whole of it; 0 when the benchmark ran to its end."""
    import yaml

    directory: Path = options.directory
    deadline = time.monotonic() + options.timeout
    names = options.servers.split(",")
    addresses: dict[str, str] = {}
    report = {"host": socket.gethostname(), "started": False, "error": "", "addresses": addresses}
    proxy: subprocess.Popen | None = None

    def left() -> float:
        """What remains of the time, a second at least."""
        return max(1.0, deadline - time.monotonic())

    def ended() -> str:
        """Which server (or the proxy) ended and what its log says; ``""`` while none did."""
        over = [(name, directory / f"exit.{name}") for name in names]
        over = [(name, code.read_text().strip() or "?") for name, code in over if code.is_file()]
        if proxy is not None and proxy.poll() is not None:
            over.append(("proxy", str(proxy.returncode)))
        for name, code in over:
            log = directory / f"{name}.log"
            said = serve_driver.last_error(runtime.tail(log, 200))
            return f"{name} ended with {code} before every server was healthy; its log ({log}) says: {said}"
        return ""

    def healthy(name: str) -> str:
        """The address ``name`` reported and answers at."""
        return serve_driver.wait_reported(directory / f"address.{name}", left(), ended)[0]

    try:
        for name in names:
            addresses[name] = healthy(name)
        ctx, gen = ([addresses[n] for n in names if n.startswith(role)] for role in ("ctx", "gen"))
        config = directory / "proxy.yaml"
        stated = yaml.safe_load(options.config.read_text(encoding="utf-8")) or {}
        config.write_text(
            yaml.safe_dump(proxy_config(stated, options.model, ctx, gen)), encoding="utf-8"
        )
        argv = [entry_point("trtllm-serve"), "disaggregated", "-c", str(config)]
        argv += ["--server_start_timeout", str(int(left())), "-r", "360000"]
        argv += ["--report_addr", str(directory / "address.proxy")]
        (directory / "proxy.launch").write_text(" ".join(argv) + "\n", encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not k.startswith(_BORROWED)}
        proxy = runtime.start_server(argv, directory / "proxy.log", env)
        report.update(started=True, url="http://" + healthy("proxy"))
        signal.alarm(int(left()))
        score(report["url"], options.model, *options.benchmark[:1], options.benchmark[1:])
    except SystemExit as exc:
        if not isinstance(exc.code, str):
            raise  # a signal, not a server: see runtime._terminate
        report["error"] = exc.code
        return 1
    finally:
        signal.alarm(0)
        wio.emit(SESSION_MARKER, report)
        if proxy is not None:
            runtime.stop_server(proxy, 1, set(), grace_s=STOP_GRACE_S, holders=lambda _: set())
    return 0


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGTERM, runtime._terminate)
    signal.signal(signal.SIGALRM, _late)
    # A client holds no device: the servers do.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, help="the checkpoint directory")
    parser.add_argument("--servers", required=True, help="their names, context first")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds in all")
    parser.add_argument("benchmark", nargs=argparse.REMAINDER, help="of trtllm-eval, and its own")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
