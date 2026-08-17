# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VideoMME accuracy over llmapi encode / prefill-decode (E/PD) disaggregation.

Separated from test_disaggregated_serving.py: the EPD-multimodal path uses an
in-process MultimodalEncoder plus a combined prefill/decode LLM, which is a
different mechanism from the trtllm-serve subprocess disaggregation exercised by
the other tests in that file.
"""

# NOTE:
# The encoder and PD are resident on the same physical GPU in the current test
# harness. Placing them on different physical GPUs silently corrupts the
# embeddings (garbage output, no error raised) in TRT-LLM's current state because
# the consumer (PD worker) rebuilds the encoder's embedding from a CUDA-IPC handle
# that currently never copies the tensor onto the PD's own compute device.
# Real cross-GPU E/PD therefore requires a real cross-device transfer
# (CPU staging or NIXL/RDMA) that is currently not natively supported in TRT-LLM.

import contextlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Protocol
from unittest import mock

import pytest

from tensorrt_llm import LLM, MultimodalEncoder
from tensorrt_llm.llmapi import KvCacheConfig, RequestOutput, SamplingParams
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import llm_models_root, skip_pre_blackwell, skip_pre_hopper
from .accuracy_core import LlmapiAccuracyTestHarness, VideoMME
from .test_disaggregated_serving import DEFAULT_TEST_TIMEOUT, MyThreadPoolExecutor


class VideoMMECompatibleLLM(Protocol):
    """LLM surface consumed by the VideoMME evaluator."""

    args: Any
    model: str
    _hf_model_dir: str
    tokenizer: Any
    input_processor: Any

    def generate_async(
        self,
        inputs: Dict[str, Any],
        sampling_params: Optional[SamplingParams] = None,
        streaming: bool = False,
    ) -> Any: ...


class _MultimodalEncoderPDAdapter:
    """Adapter that runs VideoMME dict inputs through llmapi E/PD."""

    def __init__(
        self, encoder: MultimodalEncoder, pd_llm: LLM, thread_pool: MyThreadPoolExecutor
    ) -> None:
        self._encoder = encoder
        self._pd_llm = pd_llm
        self._thread_pool = thread_pool
        self.args = pd_llm.args
        self.model = pd_llm._hf_model_dir
        self._hf_model_dir = pd_llm._hf_model_dir
        self.tokenizer = pd_llm.tokenizer
        self.input_processor = pd_llm.input_processor

    def _generate(
        self, inputs: Dict[str, Any], sampling_params: Optional[SamplingParams], streaming: bool
    ) -> RequestOutput:
        if not isinstance(inputs, dict):
            raise TypeError(f"Unsupported E/PD request input type: {type(inputs)}")

        encoder_output = self._encoder.generate_async(inputs).result()
        disaggregated_params = encoder_output.disaggregated_params
        if disaggregated_params is None:
            raise RuntimeError("Multimodal encoder did not return disaggregated params.")
        if disaggregated_params.multimodal_embedding_handles is None:
            raise RuntimeError("Multimodal encoder did not return embedding handles.")

        disaggregated_params.request_type = "context_and_generation"
        return self._pd_llm.generate_async(
            inputs,
            sampling_params=sampling_params,
            streaming=streaming,
            disaggregated_params=disaggregated_params,
        ).result()

    def generate_async(
        self,
        inputs: Dict[str, Any],
        sampling_params: Optional[SamplingParams] = None,
        streaming: bool = False,
    ):
        future = self._thread_pool.submit(self._generate, inputs, sampling_params, streaming)
        self._thread_pool.futures.append(future)
        return future


@contextlib.contextmanager
def launch_multimodal_encoder_pd_llm(
    encoder_llm_config: Dict[str, Any],
    pd_llm_config: Dict[str, Any],
    model_name: str,
    max_workers: int = 16,
) -> Iterator[VideoMMECompatibleLLM]:
    """Launch separate encoder and combined prefill/decode llmapi instances."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, {"TLLM_MULTIMODAL_DISAGGREGATED": "1"}))
        thread_pool = stack.enter_context(MyThreadPoolExecutor(max_workers=max_workers))
        encoder = MultimodalEncoder(model=model_name, **encoder_llm_config)
        pd_llm = LLM(model=model_name, **pd_llm_config)
        with encoder, pd_llm:
            yield _MultimodalEncoderPDAdapter(encoder, pd_llm, thread_pool)


@dataclass(frozen=True)
class EPDVariant:
    """Immutable per-variant config for a VideoMME E/PD run."""

    model_name: str
    model_path: str
    encoder_config: Mapping[str, Any]
    pd_config: Mapping[str, Any]
    expected_quant_algo: Optional[QuantAlgo]
    max_workers: int

    @classmethod
    def _build(
        cls,
        *,
        model_name: str,
        model_path: str,
        kv_cache_config: KvCacheConfig,
        max_batch_size: int,
        expected_quant_algo: Optional[QuantAlgo],
        max_num_tokens: int = 512,
        attn_backend: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> "EPDVariant":
        """Fill shared encoder/PD defaults for one variant.

        Optional overrides are applied before construction so the frozen
        instance never needs post-hoc mutation.
        """
        # Optional attn_backend override, applied to both configs via a spread
        # so the frozen instance never needs post-hoc mutation.
        attn_override = {"attn_backend": attn_backend} if attn_backend is not None else {}
        encoder_config = {
            "trust_remote_code": True,
            "max_batch_size": max_batch_size,
            "cuda_graph_config": None,
            **attn_override,
        }
        pd_config = {
            "backend": "pytorch",
            "disable_overlap_scheduler": True,
            "trust_remote_code": True,
            "kv_cache_config": kv_cache_config,
            "enable_chunked_prefill": True,
            "max_num_tokens": max_num_tokens,
            "max_batch_size": max_batch_size,
            "cuda_graph_config": None,
            **attn_override,
        }

        return cls(
            model_name=model_name,
            model_path=model_path,
            encoder_config=encoder_config,
            pd_config=pd_config,
            expected_quant_algo=expected_quant_algo,
            max_workers=max_workers if max_workers is not None else VideoMME.MAX_BATCH_SIZE,
        )

    @classmethod
    def qwen3vl_2b(cls) -> "EPDVariant":
        return cls._build(
            model_name="Qwen/Qwen3-VL-2B-Instruct",
            model_path=f"{llm_models_root()}/Qwen3/Qwen3-VL-2B-Instruct",
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=0.8,
                enable_block_reuse=False,
                dtype="auto",
            ),
            max_batch_size=16,
            expected_quant_algo=None,
            max_workers=16,
            attn_backend="VANILLA",
            # Qwen3-VL VideoMME prompts can exceed 1024 tokens after visual
            # expansion; avoid splitting a single context across vanilla
            # SDPA chunks in the E/P handoff path.
            max_num_tokens=2048,
        )

    @classmethod
    def nano_omni_fp8(cls) -> "EPDVariant":
        return cls._build(
            model_name="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
            model_path=f"{llm_models_root()}/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=0.8,
                mamba_ssm_cache_dtype="float32",
                enable_block_reuse=False,
                dtype="fp8",
            ),
            max_batch_size=64,
            expected_quant_algo=QuantAlgo.FP8,
        )

    @classmethod
    def nano_omni_nvfp4(cls) -> "EPDVariant":
        return cls._build(
            model_name="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
            model_path=f"{llm_models_root()}/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=0.8,
                mamba_ssm_cache_dtype="float32",
                enable_block_reuse=False,
                dtype="fp8",
            ),
            max_batch_size=128,
            expected_quant_algo=QuantAlgo.MIXED_PRECISION,
        )


class TestVideoMMEEPD(LlmapiAccuracyTestHarness):
    """VideoMME accuracy over llmapi encode / prefill-decode (E/PD) disaggregation."""

    SAMPLING_PARAMS = SamplingParams(
        max_tokens=VideoMME.MAX_OUTPUT_LEN,
        truncate_prompt_tokens=VideoMME.MAX_INPUT_LEN,
        temperature=0.0,
        top_k=1,
    )

    # Identical across all variants today; lifted to a class constant to mirror
    # agg no_thinking_evaluator_kwargs.
    NO_THINKING_EVALUATOR_KWARGS = {
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }

    def _launch_epd(self, variant: EPDVariant):
        """Context manager: encoder + combined PD llmapi."""
        return launch_multimodal_encoder_pd_llm(
            variant.encoder_config,
            variant.pd_config,
            variant.model_path,
            max_workers=variant.max_workers,
        )

    def _run_videomme(self, llm, variant: EPDVariant) -> None:
        actual_quant_algo = (
            llm.args.quant_config.quant_algo if llm.args.quant_config is not None else None
        )
        assert actual_quant_algo == variant.expected_quant_algo
        VideoMME(variant.model_name).evaluate(
            llm,
            sampling_params=self.SAMPLING_PARAMS,
            extra_evaluator_kwargs=self.NO_THINKING_EVALUATOR_KWARGS,
        )

    @pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
    @skip_pre_hopper
    @pytest.mark.skip_less_device_memory(80000)
    @pytest.mark.parametrize("_repeat", range(560, 1060), ids=lambda i: f"rep{i:02d}")
    @pytest.mark.parametrize(
        "variant",
        [
            pytest.param(
                EPDVariant.qwen3vl_2b(), marks=skip_pre_blackwell, id="qwen3vl_2b_instruct"
            ),
            pytest.param(
                EPDVariant.nano_omni_fp8(), marks=skip_pre_hopper, id="nemotron_nano_v3_omni_fp8"
            ),
            pytest.param(
                EPDVariant.nano_omni_nvfp4(),
                marks=skip_pre_blackwell,
                id="nemotron_nano_v3_omni_nvfp4",
            ),
        ],
    )
    # `torch.compile` uses a thread pool to compile and it's used in audio pre-processing.
    @pytest.mark.threadleak(enabled=False)
    def test_disaggregated_videomme(
        self, variant: EPDVariant, _repeat: int, request: pytest.FixtureRequest
    ) -> None:
        """Run VideoMME shard through a model-specific llmapi E/PD config."""
        import faulthandler
        import subprocess
        import sys
        import tempfile
        import textwrap
        from pathlib import Path

        # Pass an OS-level file (not sys.stderr) to faulthandler so the
        # SIGSEGV / SIGABRT thread dump survives pytest's --capture=fd
        # buffering. Writing to sys.stderr goes into the capture pipe;
        # if pytest is killed by the signal before flushing, the dump is
        # lost. Writing to a real file gets a direct write() from the
        # signal handler before the process dies.
        output_dir_opt = request.config.getoption("--output-dir", default=None)
        crash_dir = Path(output_dir_opt) if output_dir_opt else Path(tempfile.gettempdir())
        crash_dir.mkdir(parents=True, exist_ok=True)
        variant_slug = variant.model_name.rsplit("/", 1)[-1]
        crash_log_path = crash_dir / f"faulthandler_{variant_slug}_rep{_repeat:03d}.log"
        crash_log_file = open(crash_log_path, "w", buffering=1)
        # Route the path to sys.__stderr__ (pre-capture) so the CI console
        # can locate the file even if pytest capture ate the write above.
        print(
            f"[NVBUG-6327718] faulthandler log: {crash_log_path}",
            file=sys.__stderr__,
            flush=True,
        )

        # ==== NVBUG-6327718 full-coverage hang instrumentation ====
        # Every knob below is an existing in-tree mechanism, so this only
        # wires them together — nothing new to install in the CI container.
        # Artifacts land under --output-dir, which pytest packs into the
        # results tarball that CI uploads to urm.
        dump_dir = crash_dir / f"hang_dumps_rep{_repeat:03d}"
        dump_dir.mkdir(parents=True, exist_ok=True)

        # A sitecustomize.py runs at the top of every Python child process
        # that has this directory on PYTHONPATH — including the
        # mpi4py.futures.server workers spawned by MpiPoolSession. Prior
        # attempts had faulthandler + persistent stderr only, but every one
        # of ~85 rep632-style H100 fp8 crashes in PR_Github #66364 left the
        # per-worker faulthandler at 0 bytes — because the fault was inside
        # CUDA driver code, so control never returned to Python and no
        # Python-level SIGSEGV handler ever fired. This attempt widens
        # capture to four independent channels:
        #
        #   1. libSegFault.so (LD_PRELOAD, set below in debug_env) —
        #      glibc's fault handler, runs regardless of Python state,
        #      writes addr2line-resolvable backtrace + registers.
        #   2. resource.RLIMIT_CORE = unlimited — lets the kernel emit a
        #      core file, retrievable via `docker cp` or SLURM staging.
        #   3. py-spy watchdog + on-crash py-spy dump — walks the process
        #      externally, so native frames survive even when Python does
        #      not.
        #   4. Faulthandler as before, for hangs the kernel doesn't kill.
        sitecustomize_dir = crash_dir / f"sitecustomize_rep{_repeat:03d}"
        sitecustomize_dir.mkdir(parents=True, exist_ok=True)
        (sitecustomize_dir / "sitecustomize.py").write_text(
            textwrap.dedent("""\
            import os, sys, faulthandler, signal, pathlib, resource, \
                   subprocess, threading, time, traceback
            _dd = pathlib.Path(os.environ.get(
                "TLLM_HANG_DUMP_DIR", "/tmp/tllm_hang_dumps"))
            _dd.mkdir(parents=True, exist_ok=True)
            _pid = os.getpid()
            _exe = pathlib.Path(sys.argv[0] if sys.argv else "python").name
            _slot = _dd / f"{_pid}_{_exe}"
            _slot.mkdir(parents=True, exist_ok=True)

            def _note(m):
                try:
                    with (_slot / "instrument.log").open("a") as _f:
                        _f.write(f"[{time.strftime('%H:%M:%S')}] {m}\\n")
                except Exception:
                    pass

            # 1) Faulthandler + stderr redirect, as before.
            _fh = open(_slot / "faulthandler.log", "a", buffering=1)
            faulthandler.enable(file=_fh, all_threads=True)
            faulthandler.dump_traceback_later(30, repeat=True, file=_fh)
            try:
                faulthandler.register(signal.SIGUSR1, file=_fh,
                                      all_threads=True, chain=False)
            except Exception:
                pass
            _err = open(_slot / "stderr.log", "a", buffering=1)
            os.dup2(_err.fileno(), 2)

            # 2) Allow the kernel to write a core file. On some CI hosts
            #    core_pattern lives outside our reach; the ulimit is enough
            #    for the docker/slurm sandbox to keep the file if it lands
            #    in cwd or the configured pattern dir.
            try:
                resource.setrlimit(resource.RLIMIT_CORE,
                    (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
                _note("RLIMIT_CORE=unlimited")
            except Exception as _e:
                _note(f"setrlimit failed: {_e}")

            # 3) py-spy: (a) periodic watchdog every 30s, (b) on-crash dump
            #    from the signal handler. py-spy needs ptrace on the same
            #    pid; SYS_PTRACE is granted in the CI container so this
            #    works. Failure is silent — we still have (1) + (2).
            _pyspy_done = [False]
            def _pyspy(reason):
                if _pyspy_done[0]:
                    return
                _pyspy_done[0] = True
                out = _slot / f"pyspy_{reason}.log"
                try:
                    subprocess.run(
                        ["py-spy", "dump", "--pid", str(_pid),
                         "--native", "--nonblocking"],
                        stdout=out.open("wb"),
                        stderr=subprocess.STDOUT,
                        timeout=30, check=False,
                    )
                    _note(f"pyspy_{reason}.log written")
                except FileNotFoundError:
                    _note("py-spy not installed")
                except Exception as _e:
                    _note(f"pyspy {reason} failed: {_e}")

            def _watchdog():
                n = 0
                while True:
                    time.sleep(30)
                    n += 1
                    _pyspy_done[0] = False
                    out = _slot / f"pyspy_watchdog_{n:04d}.log"
                    try:
                        subprocess.run(
                            ["py-spy", "dump", "--pid", str(_pid),
                             "--native", "--nonblocking"],
                            stdout=out.open("wb"),
                            stderr=subprocess.STDOUT,
                            timeout=15, check=False,
                        )
                    except Exception:
                        return
            threading.Thread(target=_watchdog, name="pyspy_watchdog",
                             daemon=True).start()

            # 4) Signal forensics. Faulthandler.enable already catches
            #    SIGSEGV/SIGABRT/SIGBUS/SIGFPE/SIGILL but sequences its
            #    Python-frame dump BEFORE any C-level info. We chain a
            #    py-spy invocation on top so we get native frames next
            #    time a driver-level fault fires. Then re-raise via
            #    default handler so the core file still gets written.
            def _forensics(signum, frame):
                try:
                    with (_slot / f"crash_{signum}.txt").open("w") as _f:
                        _f.write(f"pid={_pid} signum={signum} "
                                 f"time={time.time()}\\n")
                        _f.write(f"signal_name="
                                 f"{signal.strsignal(signum)}\\n")
                        _f.write("--- python stack ---\\n")
                        traceback.print_stack(frame, file=_f)
                    _pyspy(f"sig{signum}")
                except Exception:
                    pass
                signal.signal(signum, signal.SIG_DFL)
                os.kill(_pid, signum)
            for _sig in (signal.SIGSEGV, signal.SIGABRT,
                         signal.SIGBUS, signal.SIGFPE, signal.SIGILL):
                try:
                    signal.signal(_sig, _forensics)
                except (ValueError, OSError):
                    pass
            _note(f"instrumentation live in {_slot}")
        """)
        )

        # In-tree debug knobs, propagated to MPI workers via LLM.env_overrides.
        # ZMQ + executor-loop trace knobs were removed after the first attempt
        # because they inflate the Jenkins console 10-100x per rep — for
        # ~500 reps that would truncate the tail where the crash lives.
        # The framework's --periodic-hang-traceback already dumps main-thread
        # stacks on hang, so the added value here is:
        #   * worker-side faulthandler (catches SIGSEGV / SIGABRT in the MPI
        #     worker itself; framework only introspects pytest)
        #   * persistent stderr (survives --capture=fd when worker dies)
        #   * TORCH_NCCL_DUMP_ON_TIMEOUT (only fires on collective timeout,
        #     so quiet on success)
        debug_env = {
            "TLLM_HANG_DUMP_DIR": str(dump_dir),
            "PYTHONPATH": str(sitecustomize_dir) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "TRTLLM_WORKER_PRINT_STACKS_PERIOD": "60",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "20000",
            "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": str(dump_dir / "nccl_dump"),
            "NCCL_DEBUG_SUBSYS": "INIT,COLL,GRAPH",
            "PYTHONFAULTHANDLER": "1",
        }
        os.environ.update(debug_env)

        # Best-effort py-spy install for native stack capture. The CI container
        # has SYS_ADMIN + seccomp=unconfined (jenkins/L0_Test.groovy:1280) so
        # ptrace works if py-spy is on PATH. Silent failure is fine — the
        # sitecustomize bootstrap above covers the Python-side signal already.
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--user", "--quiet", "py-spy"],
                timeout=60,
                check=False,
            )
        except Exception:
            pass

        # variant.encoder_config / pd_config are frozen dataclass mappings;
        # overlay env_overrides here so both LLM instances widen the debug
        # env into their MPI workers (worker.py:198 applies env_overrides
        # after MPI_Init, which caches the OS env at import time).
        encoder_config = {**variant.encoder_config, "env_overrides": debug_env}
        pd_config = {**variant.pd_config, "env_overrides": debug_env}
        # ==== end NVBUG-6327718 instrumentation ====

        faulthandler.enable(file=crash_log_file, all_threads=True)
        faulthandler.dump_traceback_later(60, repeat=True, file=crash_log_file)
        try:
            with launch_multimodal_encoder_pd_llm(
                encoder_config,
                pd_config,
                variant.model_path,
                max_workers=variant.max_workers,
            ) as llm:
                self._run_videomme(llm, variant)
        finally:
            faulthandler.cancel_dump_traceback_later()
            crash_log_file.flush()
            crash_log_file.close()
