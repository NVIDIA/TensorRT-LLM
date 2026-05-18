#!/usr/bin/env python3
"""Unified RDMA draft inference server.

Speaks the 96-byte WRITE_WITH_IMM Draft API protocol that Phase 1
defined, so it can interoperate with both:

- our ``IbverbsDraftOffloadLayer`` target,
- izzy's ``standalone/mock_client.py`` (smoke-only, also 96-byte).

Modes
-----

``--backend mock``
    Returns deterministic dummy tokens (``[1, 2, ..., max_draft_len]``).
    Useful for end-to-end RDMA wiring tests without loading a model.

``--backend transformers``
    Uses ``transformers.AutoModelForCausalLM`` to do real prefill +
    decode.  Each request maintains a stateful KV cache across rounds.

Run
---

::

    # Mock mode (no model needed)
    python3 draft_rdma_server.py --backend mock --control-port 47001

    # Real-model mode (requires container)
    python3 draft_rdma_server.py --control-port 47001
"""

import argparse
import importlib.util
import json
import os
import signal
import socket
import sys
import threading
import time
import types

# --- import the in-tree TRT-LLM modules ----------------------------------
#
# Two import strategies depending on the environment:
# - In the dev container, ``tensorrt_llm`` is pip-installed editable and
#   imports fully (incl. ``LLM``).  We just do regular imports.
# - On the host (no GPU torch / no CUDA libs), the full ``tensorrt_llm``
#   package can't import.  But the four speculative-decoding modules we
#   actually need are pure Python with no torch dependency at import time
#   (except ``ibverbs_draft_offload`` — but we don't use that here).
#   In that case we stub the parent packages and importlib-load the
#   submodules directly so mock-backend smoke tests can run on host.

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_SPEC_ROOT = os.path.join(_REPO_ROOT, "tensorrt_llm", "_torch", "speculative")
sys.path.insert(0, _REPO_ROOT)

try:
    # Try the natural import first.  Works in the dev container; needed
    # if ``--backend trtllm`` will instantiate ``tensorrt_llm.LLM`` later.
    from tensorrt_llm._torch.speculative.draft_api_protocol import (
        DraftApiProtocol,  # noqa: F401,E402
    )
    from tensorrt_llm._torch.speculative.ibverbs_endpoint import (  # noqa: F401,E402
        IbverbsEndpointConfig,
        _IbverbsRdmaBackend,
    )
    from tensorrt_llm._torch.speculative.pearl_trace import log as _pearl_log  # noqa: E402
    from tensorrt_llm._torch.speculative.spec_decode_channel import (  # noqa: F401,E402
        SpecDecodeChannel,
    )
    from tensorrt_llm._torch.speculative.tcp_endpoint import (  # noqa: F401,E402
        TcpEndpointConfig,
        _TcpRdmaBackend,
    )
except Exception:
    # Host fallback — stub the parent packages and load the modules by
    # path.  Only mock/standalone use cases hit this branch.

    def _stub_pkg(name):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m

    for pkg in (
        "tensorrt_llm",
        "tensorrt_llm._torch",
        "tensorrt_llm._torch.speculative",
    ):
        _stub_pkg(pkg)

    def _import_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _import_module(
        "tensorrt_llm._torch.speculative.draft_api_protocol",
        os.path.join(_SPEC_ROOT, "draft_api_protocol.py"),
    )
    _import_module(
        "tensorrt_llm._torch.speculative.spec_decode_channel",
        os.path.join(_SPEC_ROOT, "spec_decode_channel.py"),
    )
    _import_module(
        "tensorrt_llm._torch.speculative.ibverbs_endpoint",
        os.path.join(_SPEC_ROOT, "ibverbs_endpoint.py"),
    )
    _import_module(
        "tensorrt_llm._torch.speculative.tcp_endpoint",
        os.path.join(_SPEC_ROOT, "tcp_endpoint.py"),
    )
    _import_module(
        "tensorrt_llm._torch.speculative.pearl_trace",
        os.path.join(_SPEC_ROOT, "pearl_trace.py"),
    )

    from tensorrt_llm._torch.speculative.draft_api_protocol import (
        DraftApiProtocol,  # noqa: F401,E402
    )
    from tensorrt_llm._torch.speculative.ibverbs_endpoint import (  # noqa: F401,E402
        IbverbsEndpointConfig,
        _IbverbsRdmaBackend,
    )
    from tensorrt_llm._torch.speculative.pearl_trace import log as _pearl_log  # noqa: E402
    from tensorrt_llm._torch.speculative.spec_decode_channel import (  # noqa: F401,E402
        SpecDecodeChannel,
    )
    from tensorrt_llm._torch.speculative.tcp_endpoint import (  # noqa: F401,E402
        TcpEndpointConfig,
        _TcpRdmaBackend,
    )

# Local module — sits next to this file.
sys.path.insert(0, _HERE)
import draft_session_protocol  # noqa: E402

# --- backends --------------------------------------------------------------


class _MockBackend:
    """Deterministic dummy tokens, no model required."""

    def __init__(self, max_draft_len):
        self.max_draft_len = max_draft_len
        self._sessions = {}
        self.vocab_size = 0
        self.eos_token_id = 0

    def reset_sessions(self):
        self._sessions.clear()

    def matches_model_init(self, msg):
        return int(self.max_draft_len) == int(msg.max_draft_len)

    def init_session(self, request_id, prompt_tokens):
        # PEARL pre-verify shape:
        #   prompt: a b c d e
        #   draft prefill produces d_f before the target asks for drafts.
        # The mock backend does not own real KV, so it only remembers the
        # prompt length; real backends below materialize d_f immediately.
        self._sessions.pop(int(request_id), None)
        self._sessions[request_id] = len(prompt_tokens)
        return len(prompt_tokens) - 1  # last_token_position

    def step(self, request_id, last_token, position, round_seq, num_tokens=None):
        # Encode the round so we can verify ordering on the target side.
        n = int(num_tokens or self.max_draft_len)
        return [((round_seq * 1000 + i) % (1 << 31)) for i in range(1, n + 1)]


class DraftExecutorRunner:
    """PyExecutor-backed draft runner shell.

    This object makes the draft-server backend explicitly own the same
    components that TRT-LLM's native PyTorch path uses: PyExecutor,
    PyTorchModelEngine, ResourceManager/KVCacheManager, Sampler, and optional
    seq-slot managers.  The current step implementation still uses the
    executor's public streaming request API for token production; the
    important structural change is that all internal handles are now surfaced
    in one place so the next iteration can replace ``pull_stream_token`` with
    a direct ``model_engine.forward`` + ``sampler`` step.
    """

    def __init__(self, llm, role="draft"):
        self.llm = llm
        self.role = str(role)
        self.generation_executor = getattr(llm, "_executor", None)
        self.py_executor = self._extract_py_executor(self.generation_executor)
        self.draft_model_engine = getattr(self.py_executor, "model_engine", None)
        self.resource_manager = getattr(self.py_executor, "resource_manager", None)
        self.sampler = getattr(self.py_executor, "sampler", None)
        self.drafter = getattr(self.py_executor, "drafter", None)
        self.seq_slot_manager = getattr(self.py_executor, "seq_slot_manager", None)
        if self.seq_slot_manager is None and self.drafter is not None:
            self.seq_slot_manager = getattr(self.drafter, "draft_seq_slot_manager", None)
        self.kv_cache_manager = None
        self.draft_kv_cache_manager = None
        if self.resource_manager is not None:
            try:
                from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType

                managers = getattr(self.resource_manager, "resource_managers", {})
                self.kv_cache_manager = managers.get(ResourceManagerType.KV_CACHE_MANAGER)
                self.draft_kv_cache_manager = managers.get(
                    ResourceManagerType.DRAFT_KV_CACHE_MANAGER
                )
            except Exception:
                pass

    @staticmethod
    def _extract_py_executor(generation_executor):
        if generation_executor is None:
            return None
        engine = getattr(generation_executor, "engine", None)
        if engine is not None and hasattr(engine, "model_engine"):
            return engine
        worker = getattr(generation_executor, "worker", None)
        engine = getattr(worker, "engine", None)
        if engine is not None and hasattr(engine, "model_engine"):
            return engine
        return generation_executor if hasattr(generation_executor, "model_engine") else None

    @property
    def available(self):
        return self.py_executor is not None and self.draft_model_engine is not None

    def describe(self):
        return {
            "role": self.role,
            "available": bool(self.available),
            "generation_executor": type(self.generation_executor).__name__
            if self.generation_executor is not None
            else None,
            "py_executor": type(self.py_executor).__name__
            if self.py_executor is not None
            else None,
            "draft_model_engine": type(self.draft_model_engine).__name__
            if self.draft_model_engine is not None
            else None,
            "resource_manager": type(self.resource_manager).__name__
            if self.resource_manager is not None
            else None,
            "kv_cache_manager": type(self.kv_cache_manager).__name__
            if self.kv_cache_manager is not None
            else None,
            "draft_kv_cache_manager": type(self.draft_kv_cache_manager).__name__
            if self.draft_kv_cache_manager is not None
            else None,
            "sampler": type(self.sampler).__name__ if self.sampler is not None else None,
            "seq_slot_manager": type(self.seq_slot_manager).__name__
            if self.seq_slot_manager is not None
            else None,
        }


class _TrtLlmBackend:
    """Real per-request inference using a resident TRT-LLM decode stream.

    This backend keeps one long-lived ``generate_async(streaming=True)``
    request per draft request.  Normal ``step()`` calls only consume the
    next token(s) from that stream, so the TRT-LLM executor keeps the decode
    request and its KV state resident across draft rounds.  If the target
    reports a rollback or a different token than the stream predicted, we
    abort the stale stream and restart it from the corrected target prefix.
    """

    def __init__(
        self,
        model_path,
        prompt,
        max_draft_len,
        device="cuda:0",
        stream_max_tokens=2048,
    ):
        from transformers import AutoTokenizer

        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig

        self._SamplingParams = SamplingParams
        self.model_path = str(model_path)
        self.max_draft_len = max_draft_len
        self.stream_max_tokens = int(stream_max_tokens)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.vocab_size = int(self.tokenizer.vocab_size)
        self.eos_token_id = int(self.tokenizer.eos_token_id or 0)
        self.llm = LLM(
            model=model_path,
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True,
                free_gpu_memory_fraction=0.4,
            ),
            cuda_graph_config=None,
            max_batch_size=8,
        )
        self.executor_runner = DraftExecutorRunner(self.llm)
        # Greedy sampling so the draft tokens are deterministic.
        self.sampling_params = self._make_sampling_params()
        self.fixed_prompt_ids = None
        if prompt:
            self.fixed_prompt_ids = self.tokenizer.encode(prompt)
        # request_id -> session dict.  ``tokens`` is the target-corrected
        # committed prefix plus draft tokens that we have already consumed
        # from the active stream.
        self._sessions = {}
        self._next_branch_id = 1
        _pearl_log(
            "draft",
            "draft_executor_runner_init",
            backend="trtllm",
            runner=self.executor_runner.describe(),
        )

    def _make_sampling_params(self):
        return self._SamplingParams(
            max_tokens=max(1, self.stream_max_tokens),
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            detokenize=False,
        )

    def reset_sessions(self):
        for sess in list(self._sessions.values()):
            self._abort_stream(sess)
        self._sessions.clear()

    def matches_model_init(self, msg):
        return self.model_path == str(msg.model_path) and int(self.max_draft_len) == int(
            msg.max_draft_len
        )

    def init_session(self, request_id, prompt_tokens):
        self._sessions.pop(int(request_id), None)
        sess = {
            "tokens": list(prompt_tokens),
            "stream": None,
            "stream_iter": None,
            "stream_start_len": 0,
            "stream_consumed": 0,
            "stream_generation": 0,
            "stream_buffer": [],
        }
        self._sessions[int(request_id)] = sess

        # PEARL pre-verify timeline used by this server:
        #
        #   prompt: a b c d e
        #   draft prefill -> d_f
        #   target prefill -> t_f
        #
        # We generate d_f immediately after prompt prefill and keep the
        # streaming branch alive.  When the target's first data-plane packet
        # arrives with t_f, _sync_to_target compares that token against this
        # already-generated d_f:
        #
        #   if d_f == t_f:
        #       keep the stream branch and continue generating d_g, d_h, ...
        #   else:
        #       rollback to the prompt, append t_f, and regenerate from t_f
        #
        # The current TRT-LLM target path is synchronous, so d_f is not sent
        # as an unsolicited packet during prompt_init.  It is nevertheless
        # generated before target->draft verification, which gives the draft
        # side the same keep-or-rollback decision point as the PEARL diagram.
        preverify_token = self._pull_stream_token(sess)
        _pearl_log(
            "draft",
            "prompt_session_init",
            backend="trtllm",
            runner="stream",
            request_id=int(request_id),
            prompt_tokens=[int(t) for t in prompt_tokens],
            prompt_token_count=len(prompt_tokens),
            preverify_token=int(preverify_token),
            last_token_position=len(prompt_tokens) - 1,
            session_tokens_after=list(sess["tokens"]),
        )
        return len(prompt_tokens) - 1

    def _ensure_session(self, request_id):
        rid = int(request_id)
        sess = self._sessions.get(rid)
        if sess is not None:
            return sess
        if self.fixed_prompt_ids is None:
            raise RuntimeError(
                "no session for request %d and no fixed prompt set; "
                "send a TcpPromptInit first" % rid
            )
        sess = {
            "tokens": list(self.fixed_prompt_ids),
            "stream": None,
            "stream_iter": None,
            "stream_start_len": 0,
            "stream_consumed": 0,
            "stream_generation": 0,
            "stream_buffer": [],
        }
        self._sessions[rid] = sess
        return sess

    def _abort_stream(self, sess):
        stream = sess.get("stream")
        if stream is not None and not getattr(stream, "finished", True):
            try:
                stream.abort()
            except Exception:
                pass
        sess["stream"] = None
        sess["stream_iter"] = None
        sess["stream_start_len"] = 0
        sess["stream_consumed"] = 0
        sess["stream_buffer"] = []

    def _new_session_from_tokens(self, tokens):
        return {
            "tokens": list(tokens),
            "stream": None,
            "stream_iter": None,
            "stream_start_len": 0,
            "stream_consumed": 0,
            "stream_generation": 0,
            "stream_buffer": [],
        }

    def _start_stream(self, sess):
        self._abort_stream(sess)
        sess["stream_generation"] = int(sess.get("stream_generation", 0)) + 1
        sess["stream_start_len"] = len(sess["tokens"])
        sess["stream_consumed"] = 0
        sess["stream_buffer"] = []
        stream = self.llm.generate_async(
            {"prompt_token_ids": list(sess["tokens"])},
            sampling_params=self._make_sampling_params(),
            streaming=True,
        )
        sess["stream"] = stream
        sess["stream_iter"] = iter(stream)
        return stream

    def create_speculative_branch(self, request_id):
        """Fork a lightweight speculative decode branch from current tokens.

        TRT-LLM's public LLM API does not expose in-place KV rollback for an
        active streaming request.  PEARL prefetch therefore runs on a separate
        streaming request.  On cache hit we adopt it; on miss we abort it and
        the main stream remains at the last verified path.
        """
        sess = self._ensure_session(request_id)
        branch = self._new_session_from_tokens(sess["tokens"])
        branch["branch_id"] = self._next_branch_id
        self._next_branch_id += 1
        branch["request_id"] = int(request_id)
        return branch

    def discard_speculative_branch(self, branch):
        if branch is not None:
            self._abort_stream(branch)

    def commit_speculative_branch(self, request_id, branch):
        if branch is None:
            return
        old = self._sessions.get(int(request_id))
        if old is not None and old is not branch:
            self._abort_stream(old)
        self._sessions[int(request_id)] = branch

    def step_branch(
        self,
        branch,
        request_id,
        last_token,
        position,
        round_seq,
        num_tokens=None,
        on_token=None,
        cancel_event=None,
    ):
        before_tokens = list(branch["tokens"])
        sync_info = self._sync_to_target(branch, last_token, position)
        compute_prefix = list(branch["tokens"])
        n = int(num_tokens or self.max_draft_len)
        draft_tokens = []
        cancelled = False
        for idx in range(n):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            token = self._pull_stream_token(branch)
            draft_tokens.append(token)
            if on_token is not None:
                keep_going = on_token(int(token), idx, branch)
                if keep_going is False:
                    cancelled = True
                    break
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
        _pearl_log(
            "draft",
            "backend_step",
            backend="trtllm",
            runner="stream_branch",
            request_id=int(request_id),
            branch_id=int(branch.get("branch_id", 0)),
            round_seq=int(round_seq),
            received_last_token=int(last_token),
            received_position=int(position),
            requested_num_tokens=n,
            stream_generation=int(branch.get("stream_generation", 0)),
            stream_consumed=int(branch.get("stream_consumed", 0)),
            stream_restart_reason=sync_info["restart_reason"],
            pearl_preverify_match=bool(sync_info["pearl_preverify_match"]),
            sync_prefix_len_before_last=int(sync_info["prefix_len_before_last"]),
            sync_desired_len=int(sync_info["desired_len"]),
            cancelled=cancelled,
            session_tokens_before=before_tokens,
            compute_prefix_tokens=compute_prefix,
            generated_draft_tokens=[int(t) for t in draft_tokens],
            session_tokens_after=list(branch["tokens"]),
        )
        return draft_tokens

    def _pull_stream_token(self, sess):
        if sess.get("stream_buffer"):
            token = int(sess["stream_buffer"].pop(0))
            sess["stream_consumed"] = int(sess.get("stream_consumed", 0)) + 1
            sess["tokens"].append(token)
            return token

        if sess.get("stream_iter") is None or getattr(sess.get("stream"), "finished", False):
            self._start_stream(sess)

        try:
            chunk = next(sess["stream_iter"])
        except StopIteration:
            self._start_stream(sess)
            chunk = next(sess["stream_iter"])

        token_diff = list(chunk.outputs[0].token_ids_diff)
        if not token_diff:
            # A final/metadata-only streaming response can happen at request
            # boundaries.  Restart once from the current prefix and pull again.
            self._start_stream(sess)
            chunk = next(sess["stream_iter"])
            token_diff = list(chunk.outputs[0].token_ids_diff)
        if not token_diff:
            raise RuntimeError("TRT-LLM streaming step produced no token")

        sess["stream_buffer"].extend(int(t) for t in token_diff)
        token = int(sess["stream_buffer"].pop(0))
        sess["stream_consumed"] = int(sess.get("stream_consumed", 0)) + 1
        sess["tokens"].append(token)
        return token

    def _sync_to_target(self, sess, last_token, position):
        restart_reason = None
        before_len = len(sess["tokens"])
        pos = int(position)
        token = int(last_token)
        tokens = sess["tokens"]

        # PEARL pre/post-verify alignment.
        #
        # Suppose prompt is "a b c d e".  Draft has already produced d_f
        # during prompt_init while target independently produced t_f.
        #
        # First verification:
        #   target -> draft: t_f, position=e
        #   if existing token at position+1 is d_f == t_f:
        #       keep the branch and the next pulls become d_g, d_h, ...
        #   else:
        #       trim back to the prompt, append t_f, restart generation.
        #
        # Later verification is the same shape:
        #   target -> draft: t_g, position=f
        #   if buffered/streamed d_g == t_g, keep going;
        #   otherwise roll back from d_g and regenerate after t_g.
        # Target sends the position immediately before ``last_token``.  For
        # example, after prompt prefill the first generated token arrives with
        # position == prompt_last_position.  Keep the prefix through that
        # position, then align/append ``last_token`` at position + 1.
        prefix_len_before_last = max(0, pos + 1)
        already_has_last_at_next = (
            0 <= prefix_len_before_last < len(tokens)
            and int(tokens[prefix_len_before_last]) == token
        )
        # Be permissive for older traces/callers that may have sent the
        # position of last_token itself.
        already_has_last_at_pos = 0 <= pos < len(tokens) and int(tokens[pos]) == token
        if already_has_last_at_next:
            desired_len = prefix_len_before_last + 1
        elif already_has_last_at_pos:
            desired_len = pos + 1
        else:
            desired_len = prefix_len_before_last
        pearl_preverify_match = bool(already_has_last_at_next)

        if desired_len < len(sess["tokens"]):
            sess["tokens"] = sess["tokens"][:desired_len]
            self._abort_stream(sess)
            restart_reason = "rollback"

        if not sess["tokens"] or sess["tokens"][-1] != token:
            aligned_from_stream = False
            if restart_reason is None and sess.get("stream_iter") is not None:
                try:
                    predicted = self._pull_stream_token(sess)
                    aligned_from_stream = predicted == token
                    if not aligned_from_stream:
                        sess["tokens"].pop()
                except Exception:
                    aligned_from_stream = False

            if not aligned_from_stream:
                self._abort_stream(sess)
                sess["tokens"].append(token)
                restart_reason = restart_reason or "target_token_mismatch"

        return {
            "before_len": before_len,
            "after_len": len(sess["tokens"]),
            "prefix_len_before_last": prefix_len_before_last,
            "desired_len": desired_len,
            "restart_reason": restart_reason,
            "pearl_preverify_match": pearl_preverify_match,
        }

    def step(self, request_id, last_token, position, round_seq, num_tokens=None):
        sess = self._ensure_session(request_id)
        before_tokens = list(sess["tokens"])
        sync_info = self._sync_to_target(sess, last_token, position)
        compute_prefix = list(sess["tokens"])
        n = int(num_tokens or self.max_draft_len)
        draft_tokens = [self._pull_stream_token(sess) for _ in range(n)]
        _pearl_log(
            "draft",
            "backend_step",
            backend="trtllm",
            runner="stream",
            request_id=int(request_id),
            round_seq=int(round_seq),
            received_last_token=int(last_token),
            received_position=int(position),
            requested_num_tokens=n,
            stream_generation=int(sess.get("stream_generation", 0)),
            stream_consumed=int(sess.get("stream_consumed", 0)),
            stream_restart_reason=sync_info["restart_reason"],
            pearl_preverify_match=bool(sync_info["pearl_preverify_match"]),
            sync_prefix_len_before_last=int(sync_info["prefix_len_before_last"]),
            sync_desired_len=int(sync_info["desired_len"]),
            session_tokens_before=before_tokens,
            compute_prefix_tokens=compute_prefix,
            generated_draft_tokens=[int(t) for t in draft_tokens],
            session_tokens_after=list(sess["tokens"]),
        )
        return draft_tokens


class _TrtLlmExecutorBackend(_TrtLlmBackend):
    """Direct PyExecutor draft backend with explicit request/KV ownership.

    Unlike ``_TrtLlmBackend``, this path does not ask the public LLM API to
    start a fresh streaming generation when the target diverges.  It creates a
    resident ``LlmRequest`` for every draft request, drives the local
    ``PyExecutor`` manually, and rewinds the KV cache by setting
    ``request.py_rewind_len`` before calling ``ResourceManager.update_resources``.

    The LLM worker thread is still created by ``LLM(...)``.  We use
    ``PyExecutor.control_action()`` around each manual forward so the event loop
    is paused while the draft server touches model/KV/sampler internals.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.executor_runner.available:
            raise RuntimeError(
                "failed to initialize PyExecutor-backed draft runner; "
                "LLM._executor did not expose a local PyExecutor/model_engine"
            )
        try:
            from tensorrt_llm._torch.pyexecutor.llm_request import (
                LlmRequest,
                LlmRequestState,
                SamplingConfig,
            )
            from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
        except Exception as exc:
            raise RuntimeError("failed to import PyExecutor request/scheduler internals") from exc

        self._LlmRequest = LlmRequest
        self._LlmRequestState = LlmRequestState
        self._SamplingConfig = SamplingConfig
        self._ScheduledRequests = ScheduledRequests
        self._manual_lock = threading.RLock()
        _pearl_log(
            "draft",
            "draft_executor_runner_ready",
            backend="trtllm-executor",
            runner=self.executor_runner.describe(),
        )

    def reset_sessions(self):
        with self._manual_lock:
            for sess in list(self._sessions.values()):
                self._free_manual_request(sess)
            self._sessions.clear()

    def init_session(self, request_id, prompt_tokens):
        rid = int(request_id)
        with self._manual_lock:
            old = self._sessions.pop(rid, None)
            if old is not None:
                self._free_manual_request(old)
            req = self._make_manual_request(rid, list(prompt_tokens))
            sess = {
                "request": req,
                "tokens": list(prompt_tokens),
                # KV contains exactly the prompt after context prefill.  Target
                # will send the first generated token before we decode further.
                "kv_token_len": 0,
                "forward_count": 0,
            }
            self._sessions[rid] = sess
            self._manual_prefill_context(sess)
            # PEARL pre-verify shape:
            #   draft prefill produces d_f before the first data-plane round.
            # If target later sends the same t_f, _sync_to_target keeps this
            # resident KV path and the next manual forwards produce d_g... .
            # If t_f differs, _sync_to_target rewinds to the prompt and uses
            # t_f as the corrected input before regenerating.
            preverify_token = self._manual_forward_one(sess, is_context=False)
            _pearl_log(
                "draft",
                "prompt_session_init",
                backend="trtllm-executor",
                runner="manual_pyexecutor",
                request_id=rid,
                prompt_tokens=[int(t) for t in prompt_tokens],
                prompt_token_count=len(prompt_tokens),
                preverify_token=int(preverify_token),
                last_token_position=len(prompt_tokens) - 1,
                kv_token_len=int(sess["kv_token_len"]),
                session_tokens_after=list(sess["tokens"]),
            )
            return len(prompt_tokens) - 1

    def _ensure_session(self, request_id):
        rid = int(request_id)
        sess = self._sessions.get(rid)
        if sess is not None:
            return sess
        if self.fixed_prompt_ids is None:
            raise RuntimeError(
                "no session for request %d and no fixed prompt set; "
                "send a TcpPromptInit first" % rid
            )
        self.init_session(rid, self.fixed_prompt_ids)
        return self._sessions[rid]

    def _make_manual_request(self, request_id, prompt_tokens):
        sampling = self._make_sampling_params()
        req = self._LlmRequest(
            request_id=int(request_id),
            max_new_tokens=max(1, int(self.stream_max_tokens)),
            input_tokens=[int(t) for t in prompt_tokens],
            sampling_config=self._SamplingConfig(sampling._get_sampling_config()),
            is_streaming=False,
            end_id=int(self.eos_token_id),
            pad_id=int(self.eos_token_id),
            return_context_logits=False,
            return_generation_logits=False,
            exclude_last_generation_logits=True,
        )
        return req

    def _free_manual_request(self, sess):
        req = sess.get("request") if sess is not None else None
        if req is None:
            return
        try:
            self.executor_runner.resource_manager.free_resources(req)
        except Exception:
            pass

    def _request_tokens(self, req):
        return [int(t) for t in req.get_tokens(0)]

    def _set_request_tokens(self, req, full_tokens):
        prompt_len = int(req.py_prompt_len)
        generated = [int(t) for t in full_tokens[prompt_len:]]
        req.set_generated_tokens([generated])

    def _append_request_token(self, req, token):
        req.add_new_token(int(token), 0)

    def _make_batch(self, req, is_context=False):
        batch = self._ScheduledRequests()
        if is_context or req.state != self._LlmRequestState.GENERATION_IN_PROGRESS:
            batch.append_context_request(req)
        else:
            batch.append_generation_request(req)
        return batch

    def _update_kv_resources(self, batch):
        model_engine = self.executor_runner.draft_model_engine
        attn_metadata = getattr(model_engine, "attn_metadata", None)
        kv_cache_dtype_byte_size = getattr(model_engine, "kv_cache_dtype_byte_size", None)
        self.executor_runner.resource_manager.update_resources(
            batch, attn_metadata, kv_cache_dtype_byte_size
        )

    def _manual_prefill_context(self, sess):
        import torch

        req = sess["request"]
        batch = self._make_batch(req, is_context=True)
        with self.executor_runner.py_executor.control_action():
            self.executor_runner.resource_manager.prepare_resources(batch)
            py_executor = self.executor_runner.py_executor
            gather_context_logits = any(
                request.py_return_context_logits for request in batch.context_requests
            )
            cache_indirection_buffer = self.executor_runner.sampler.get_cache_indirection()
            py_executor.execution_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(py_executor.execution_stream):
                self.executor_runner.draft_model_engine.forward(
                    batch,
                    self.executor_runner.resource_manager,
                    new_tensors_device=None,
                    gather_context_logits=gather_context_logits,
                    cache_indirection_buffer=cache_indirection_buffer,
                    num_accepted_tokens_device=None,
                )
            torch.cuda.current_stream().wait_stream(py_executor.execution_stream)
            py_executor._update_request_states(batch)
            self._update_kv_resources(batch)
            py_executor.iter_counter += 1
        sess["kv_token_len"] = len(self._request_tokens(req))
        sess["tokens"] = self._request_tokens(req)
        sess["forward_count"] = int(sess.get("forward_count", 0)) + 1

    def _manual_forward_one(self, sess, is_context=False):
        import torch

        req = sess["request"]
        tokens_before = self._request_tokens(req)
        batch = self._make_batch(req, is_context=is_context)
        with self.executor_runner.py_executor.control_action():
            self.executor_runner.resource_manager.prepare_resources(batch)
            gather_context_logits = any(
                request.py_return_context_logits for request in batch.context_requests
            )
            cache_indirection_buffer = self.executor_runner.sampler.get_cache_indirection()
            py_executor = self.executor_runner.py_executor
            py_executor.execution_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(py_executor.execution_stream):
                outputs = self.executor_runner.draft_model_engine.forward(
                    batch,
                    self.executor_runner.resource_manager,
                    new_tensors_device=None,
                    gather_context_logits=gather_context_logits,
                    cache_indirection_buffer=cache_indirection_buffer,
                    num_accepted_tokens_device=None,
                )
            torch.cuda.current_stream().wait_stream(py_executor.execution_stream)
            sample_state = self.executor_runner.py_executor._sample_async(batch, outputs)
            if sample_state is None:
                raise RuntimeError("manual PyExecutor sampling failed")
            if sample_state.sampler_event is not None:
                sample_state.sampler_event.synchronize()
            self.executor_runner.py_executor._update_requests(
                sample_state, self.executor_runner.resource_manager
            )
            self.executor_runner.py_executor._update_request_states(batch)
            self._update_kv_resources(batch)
            self.executor_runner.py_executor.iter_counter += 1

        tokens_after = self._request_tokens(req)
        if len(tokens_after) <= len(tokens_before):
            raise RuntimeError("manual PyExecutor forward did not append a sampled token")
        # One forward computes KV for the tokens that existed before sampling;
        # the newly sampled token is the unprocessed tail.
        sess["kv_token_len"] = max(int(sess.get("kv_token_len", 0)), len(tokens_before))
        sess["tokens"] = tokens_after
        sess["forward_count"] = int(sess.get("forward_count", 0)) + 1
        return int(tokens_after[-1])

    def _apply_manual_rollback(self, sess, desired_len):
        req = sess["request"]
        current_tokens = self._request_tokens(req)
        desired_len = max(0, min(int(desired_len), len(current_tokens)))
        kv_token_len = int(sess.get("kv_token_len", 0))
        rewind_len = max(0, kv_token_len - desired_len)
        if rewind_len > 0:
            req.py_rewind_len = int(rewind_len)
            batch = self._ScheduledRequests()
            batch.append_generation_request(req)
            with self.executor_runner.py_executor.control_action():
                self._update_kv_resources(batch)
            req.py_rewind_len = 0
            sess["kv_token_len"] = kv_token_len - rewind_len
        trimmed = current_tokens[:desired_len]
        self._set_request_tokens(req, trimmed)
        sess["tokens"] = self._request_tokens(req)
        return rewind_len

    def _sync_to_target(self, sess, last_token, position):
        req = sess["request"]
        before_tokens = self._request_tokens(req)
        restart_reason = None
        pos = int(position)
        token = int(last_token)

        # This is the manual PyExecutor version of the PEARL state machine:
        # target sends the verified token for the next position; if the token
        # already sitting at position+1 is the same draft token, keep the
        # resident KV branch. Otherwise rewind to before that token, append
        # target's correction, and regenerate subsequent draft tokens.
        prefix_len_before_last = max(0, pos + 1)
        already_has_last_at_next = (
            0 <= prefix_len_before_last < len(before_tokens)
            and int(before_tokens[prefix_len_before_last]) == token
        )
        already_has_last_at_pos = 0 <= pos < len(before_tokens) and int(before_tokens[pos]) == token
        if already_has_last_at_next:
            desired_len = prefix_len_before_last + 1
        elif already_has_last_at_pos:
            desired_len = pos + 1
        else:
            desired_len = prefix_len_before_last
        pearl_preverify_match = bool(already_has_last_at_next)

        rewind_len = 0
        if desired_len < len(before_tokens):
            rewind_len = self._apply_manual_rollback(sess, desired_len)
            restart_reason = "rollback" if rewind_len > 0 else "token_tail_trim"

        tokens = self._request_tokens(req)
        if not tokens or int(tokens[-1]) != token:
            tokens = tokens[:prefix_len_before_last]
            if len(tokens) < prefix_len_before_last:
                tokens = tokens + before_tokens[len(tokens) : prefix_len_before_last]
            self._set_request_tokens(req, tokens)
            self._append_request_token(req, token)
            sess["tokens"] = self._request_tokens(req)
            restart_reason = restart_reason or "target_token_mismatch"

        return {
            "before_len": len(before_tokens),
            "after_len": len(sess["tokens"]),
            "prefix_len_before_last": prefix_len_before_last,
            "desired_len": desired_len,
            "restart_reason": restart_reason,
            "pearl_preverify_match": pearl_preverify_match,
            "rewind_len": rewind_len,
            "kv_token_len": int(sess.get("kv_token_len", 0)),
        }

    def step(self, request_id, last_token, position, round_seq, num_tokens=None):
        with self._manual_lock:
            sess = self._ensure_session(request_id)
            before_tokens = self._request_tokens(sess["request"])
            sync_info = self._sync_to_target(sess, last_token, position)
            compute_prefix = self._request_tokens(sess["request"])
            n = int(num_tokens or self.max_draft_len)
            draft_tokens = [self._manual_forward_one(sess, is_context=False) for _ in range(n)]
            _pearl_log(
                "draft",
                "backend_step",
                backend="trtllm-executor",
                runner="manual_pyexecutor",
                request_id=int(request_id),
                round_seq=int(round_seq),
                received_last_token=int(last_token),
                received_position=int(position),
                requested_num_tokens=n,
                stream_restart_reason=sync_info["restart_reason"],
                pearl_preverify_match=bool(sync_info["pearl_preverify_match"]),
                manual_rewind_len=int(sync_info["rewind_len"]),
                manual_kv_token_len=int(sess.get("kv_token_len", 0)),
                sync_prefix_len_before_last=int(sync_info["prefix_len_before_last"]),
                sync_desired_len=int(sync_info["desired_len"]),
                forward_count=int(sess.get("forward_count", 0)),
                session_tokens_before=before_tokens,
                compute_prefix_tokens=compute_prefix,
                generated_draft_tokens=[int(t) for t in draft_tokens],
                session_tokens_after=self._request_tokens(sess["request"]),
            )
            return draft_tokens

    def create_speculative_branch(self, request_id):
        with self._manual_lock:
            sess = self._ensure_session(request_id)
            branch = {
                "branch_id": self._next_branch_id,
                "request_id": int(request_id),
                "snapshot_tokens": self._request_tokens(sess["request"]),
                "snapshot_kv_token_len": int(sess.get("kv_token_len", 0)),
                "snapshot_forward_count": int(sess.get("forward_count", 0)),
            }
            self._next_branch_id += 1
            return branch

    def discard_speculative_branch(self, branch):
        if branch is None:
            return
        with self._manual_lock:
            sess = self._sessions.get(int(branch["request_id"]))
            if sess is None:
                return
            self._apply_manual_rollback(sess, len(branch["snapshot_tokens"]))
            self._set_request_tokens(sess["request"], branch["snapshot_tokens"])
            sess["tokens"] = self._request_tokens(sess["request"])
            sess["kv_token_len"] = min(int(branch["snapshot_kv_token_len"]), len(sess["tokens"]))

    def commit_speculative_branch(self, request_id, branch):
        # The manual branch runs on the resident request/KV cache directly.
        # A cache hit therefore only means "keep the already advanced state".
        return

    def step_branch(
        self,
        branch,
        request_id,
        last_token,
        position,
        round_seq,
        num_tokens=None,
        on_token=None,
        cancel_event=None,
    ):
        with self._manual_lock:
            sess = self._ensure_session(request_id)
            before_tokens = self._request_tokens(sess["request"])
            sync_info = self._sync_to_target(sess, last_token, position)
            compute_prefix = self._request_tokens(sess["request"])
            n = int(num_tokens or self.max_draft_len)
            draft_tokens = []
            cancelled = False
            for idx in range(n):
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                token = self._manual_forward_one(sess, is_context=False)
                draft_tokens.append(token)
                if on_token is not None:
                    keep_going = on_token(int(token), idx, branch)
                    if keep_going is False:
                        cancelled = True
                        break
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
            _pearl_log(
                "draft",
                "backend_step",
                backend="trtllm-executor",
                runner="manual_pyexecutor_branch",
                request_id=int(request_id),
                branch_id=int(branch.get("branch_id", 0)) if branch else 0,
                round_seq=int(round_seq),
                received_last_token=int(last_token),
                received_position=int(position),
                requested_num_tokens=n,
                stream_restart_reason=sync_info["restart_reason"],
                pearl_preverify_match=bool(sync_info["pearl_preverify_match"]),
                manual_rewind_len=int(sync_info["rewind_len"]),
                manual_kv_token_len=int(sess.get("kv_token_len", 0)),
                sync_prefix_len_before_last=int(sync_info["prefix_len_before_last"]),
                sync_desired_len=int(sync_info["desired_len"]),
                cancelled=cancelled,
                forward_count=int(sess.get("forward_count", 0)),
                session_tokens_before=before_tokens,
                compute_prefix_tokens=compute_prefix,
                generated_draft_tokens=[int(t) for t in draft_tokens],
                session_tokens_after=self._request_tokens(sess["request"]),
            )
            return draft_tokens


class _TransformersBackend:
    """Real per-request stateful inference using transformers."""

    def __init__(self, model_path, prompt, max_draft_len, device="cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_path = str(model_path)
        self.max_draft_len = max_draft_len
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.vocab_size = int(self.tokenizer.vocab_size)
        self.eos_token_id = int(self.tokenizer.eos_token_id or 0)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(
            device
        )
        self.model.eval()
        self.fixed_prompt_ids = None
        if prompt:
            self.fixed_prompt_ids = self.tokenizer.encode(prompt)
        # Per-request KV cache, prefill state.
        self._sessions = {}

    def reset_sessions(self):
        self._sessions.clear()

    def matches_model_init(self, msg):
        return self.model_path == str(msg.model_path) and int(self.max_draft_len) == int(
            msg.max_draft_len
        )

    def init_session(self, request_id, prompt_tokens):
        """Phase 7 TCP-prefill entry point — prefill an explicit prompt."""
        torch = self._torch
        self._sessions.pop(int(request_id), None)
        ids = torch.tensor([list(prompt_tokens)], device=self.device)
        with torch.no_grad():
            out = self.model(ids, use_cache=True, past_key_values=None)
        self._sessions[request_id] = {
            "past_key_values": out.past_key_values,
            "position": len(prompt_tokens),
        }
        return len(prompt_tokens) - 1

    def _ensure_session(self, request_id, seed_position):
        sess = self._sessions.get(request_id)
        if sess is not None:
            return sess
        # Backward-compatible fallback: prefill from --prompt.
        torch = self._torch
        if self.fixed_prompt_ids is None:
            raise RuntimeError(
                "no session for request %d and no fixed prompt set; "
                "send a TcpPromptInit first" % request_id
            )
        ids = torch.tensor([self.fixed_prompt_ids], device=self.device)
        with torch.no_grad():
            out = self.model(ids, use_cache=True, past_key_values=None)
        sess = {
            "past_key_values": out.past_key_values,
            "position": len(self.fixed_prompt_ids),
        }
        self._sessions[request_id] = sess
        return sess

    def step(self, request_id, last_token, position, round_seq, num_tokens=None):
        torch = self._torch
        sess = self._ensure_session(request_id, position)
        if position < sess["position"]:
            # KV cache truncation on rejection.  Modern transformers
            # (Qwen3 / Llama / etc) use ``DynamicCache`` which exposes
            # ``.crop(target_len)``; older ``tuple[tuple[K,V]]`` style
            # is supported as a fallback.
            pkv = sess["past_key_values"]
            if hasattr(pkv, "crop"):
                pkv.crop(position)
            else:
                pkv = tuple((k[:, :, :position, :], v[:, :, :position, :]) for k, v in pkv)
                sess["past_key_values"] = pkv
            sess["position"] = position
        cur = torch.tensor([[last_token]], device=self.device)
        draft_tokens = []
        for _ in range(int(num_tokens or self.max_draft_len)):
            with torch.no_grad():
                out = self.model(cur, use_cache=True, past_key_values=sess["past_key_values"])
            sess["past_key_values"] = out.past_key_values
            next_token = int(out.logits[0, -1].argmax().item())
            draft_tokens.append(next_token)
            cur = torch.tensor([[next_token]], device=self.device)
            sess["position"] += 1
        return draft_tokens


# --- server ----------------------------------------------------------------


class IzzyCompatibleDraftServer:
    """RDMA draft server speaking the 96-byte Draft API protocol."""

    def __init__(self, args, backend=None):
        self._args = args
        self._backend = backend
        self._backend_lock = threading.Lock()
        self._backend_ready = threading.Event()
        if backend is not None:
            self._backend_ready.set()
        self._data_transport = None
        self._max_num_requests = 0
        self._endpoint = None
        self._channel = None
        self._stop = False
        self._round_count = 0
        self._tcp_listener_thread = None
        self._tcp_listener_socket = None

    def _new_channel(self):
        transport = str(self._data_transport or "").lower()
        if not transport:
            raise RuntimeError("data transport was not provided by target TcpModelInit")
        if transport == "tcp":
            endpoint = _TcpRdmaBackend(
                TcpEndpointConfig(
                    is_server=True,
                    bind_host="0.0.0.0",
                    bind_port=int(self._args.port),
                    recv_queue_depth=int(self._max_num_requests),
                    payload_bytes=DraftApiProtocol.kMessageBytes,
                )
            )
        elif transport == "ibverbs":
            endpoint = _IbverbsRdmaBackend(
                IbverbsEndpointConfig(
                    nic_name=self._args.nic,
                    is_server=True,
                    local_port=self._args.port,
                    remote_host="0.0.0.0",
                    remote_port=self._args.port,
                    payload_bytes=DraftApiProtocol.kMessageBytes,
                    recv_queue_depth=self._max_num_requests,
                    max_num_requests=self._max_num_requests,
                )
            )
        elif transport == "doca":
            from tensorrt_llm._torch.speculative.doca_endpoint import (
                DocaEndpointConfig,
                _DocaEndpointBackend,
            )

            endpoint = _DocaEndpointBackend(
                DocaEndpointConfig(
                    gpu_id=int(os.environ.get("GPU_ID", "0")),
                    nic_name=self._args.nic or os.environ.get("NIC", "mlx5_0"),
                    is_server=True,
                    bind_port=int(self._args.port),
                    remote_port=int(self._args.port),
                    remote_peer_name="draft_lpu",
                    payload_bytes=DraftApiProtocol.kMessageBytes,
                    recv_queue_depth=int(self._max_num_requests),
                    handshake_timeout_s=float(self._args.handshake_timeout_s),
                )
            )
        else:
            raise RuntimeError("unsupported data transport: %s" % transport)
        self._data_transport = transport
        return endpoint, SpecDecodeChannel(endpoint)

    def stop(self):
        self._stop = True
        # Wake the main thread if it is still waiting for TcpModelInit.  A
        # signal handler only runs on the main thread, so without this the
        # process can sit inside Event.wait() until backend-init-timeout.
        self._backend_ready.set()
        if self._channel is not None:
            try:
                self._channel.stop()
            except Exception:
                pass
        if self._tcp_listener_socket is not None:
            try:
                self._tcp_listener_socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._tcp_listener_socket.close()
            except Exception:
                pass

    def _start_tcp_prompt_listener(self):
        """TCP listener for model-init, prompt-init, and cancel messages."""
        if self._args.control_port <= 0:
            raise RuntimeError("--control-port is required for lazy model init")
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self._args.control_port))
        srv.listen(8)
        self._tcp_listener_socket = srv
        print("[draft-rdma] control listener: port=%d" % self._args.control_port, flush=True)

        def _serve_one(client):
            try:
                while True:
                    raw = draft_session_protocol.recv_line(client, timeout_s=60.0)
                    msg = draft_session_protocol.parse(raw)
                    if isinstance(msg, draft_session_protocol.TcpModelInit):
                        self._handle_model_init(client, msg)
                    elif isinstance(msg, draft_session_protocol.TcpPromptInit):
                        try:
                            if not self._backend_ready.is_set():
                                raise RuntimeError("backend not initialized")
                            if hasattr(self._backend, "reset_sessions"):
                                self._backend.reset_sessions()
                            last_pos = self._backend.init_session(
                                request_id=int(msg.request_id),
                                prompt_tokens=list(msg.prompt_tokens),
                            )
                            ack = draft_session_protocol.TcpPromptInitAck(
                                request_id=int(msg.request_id),
                                status="ok",
                                last_token_position=int(last_pos),
                            )
                        except Exception as exc:
                            ack = draft_session_protocol.TcpPromptInitAck(
                                request_id=int(msg.request_id),
                                status="error",
                                error=str(exc),
                                last_token_position=0,
                            )
                        _pearl_log(
                            "draft",
                            "control_prompt_init",
                            request_id=int(msg.request_id),
                            prompt_tokens=[int(t) for t in msg.prompt_tokens],
                            prompt_token_count=len(msg.prompt_tokens),
                            ack={
                                "status": ack.status,
                                "error": getattr(ack, "error", ""),
                                "last_token_position": int(ack.last_token_position),
                            },
                        )
                        client.sendall(draft_session_protocol.serialize(ack))
                    elif isinstance(msg, draft_session_protocol.TcpCancelRequest):
                        ack = draft_session_protocol.TcpCancelRequestAck(
                            request_id=int(msg.request_id),
                            status="ok",
                        )
                        client.sendall(draft_session_protocol.serialize(ack))
                    else:
                        # Unknown message — close the connection.
                        return
            except (ConnectionError, OSError):
                pass
            except Exception as exc:
                print("[draft-rdma] tcp client error: %r" % exc, flush=True)
            finally:
                try:
                    client.close()
                except Exception:
                    pass

        def _listen_loop():
            while not self._stop:
                try:
                    client, _ = srv.accept()
                except OSError:
                    return
                threading.Thread(target=_serve_one, args=(client,), daemon=True).start()

        self._tcp_listener_thread = threading.Thread(target=_listen_loop, daemon=True)
        self._tcp_listener_thread.start()

    def _make_backend_from_model_init(self, msg):
        if self._args.backend == "mock":
            return _MockBackend(max_draft_len=int(msg.max_draft_len))
        if self._args.backend == "transformers":
            return _TransformersBackend(
                model_path=msg.model_path,
                prompt=self._args.prompt,
                max_draft_len=int(msg.max_draft_len),
                device=self._args.device,
            )
        if self._args.backend in ("trtllm", "trtllm-executor"):
            backend_cls = (
                _TrtLlmExecutorBackend
                if self._args.backend == "trtllm-executor"
                else _TrtLlmBackend
            )
            return backend_cls(
                model_path=msg.model_path,
                prompt=self._args.prompt,
                max_draft_len=int(msg.max_draft_len),
                device=self._args.device,
                stream_max_tokens=int(self._args.trtllm_stream_max_tokens),
            )
        raise RuntimeError("unknown backend")

    @staticmethod
    def _allocate_free_port():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", 0))
            return int(sock.getsockname()[1])
        finally:
            sock.close()

    def _ensure_data_port(self):
        if int(self._args.port) <= 0:
            self._args.port = self._allocate_free_port()

    def _handle_model_init(self, client, msg):
        with self._backend_lock:
            try:
                extra = json.loads(msg.extra_kwargs_json or "{}")
            except Exception:
                extra = {}
            requested_transport = str(extra.get("transport", "")).lower()
            _pearl_log(
                "draft",
                "control_model_init_recv",
                model_path=str(msg.model_path),
                dtype=str(msg.dtype),
                max_draft_len=int(msg.max_draft_len),
                extra=extra,
            )
            if requested_transport not in ("tcp", "ibverbs", "doca"):
                ack = draft_session_protocol.TcpModelInitAck(
                    status="error",
                    error="target must provide data transport tcp|ibverbs|doca in TcpModelInit",
                )
                client.sendall(draft_session_protocol.serialize(ack))
                return
            self._data_transport = requested_transport
            try:
                extra_max = int(extra.get("max_num_requests", 0))
            except Exception:
                extra_max = 0
            if extra_max <= 0:
                ack = draft_session_protocol.TcpModelInitAck(
                    status="error",
                    error="target must provide max_num_requests in TcpModelInit",
                )
                client.sendall(draft_session_protocol.serialize(ack))
                return
            self._max_num_requests = extra_max
            self._ensure_data_port()

            if self._backend is not None:
                if self._backend.matches_model_init(msg):
                    self._backend.reset_sessions()
                    ack = draft_session_protocol.TcpModelInitAck(
                        status="ok",
                        vocab_size=int(getattr(self._backend, "vocab_size", 0)),
                        eos_token_id=int(getattr(self._backend, "eos_token_id", 0)),
                        data_port=int(self._args.port),
                    )
                    print(
                        "[draft-rdma] model_init reused existing model=%s; cleared sessions"
                        % msg.model_path,
                        flush=True,
                    )
                    client.sendall(draft_session_protocol.serialize(ack))
                    return
                print(
                    "[draft-rdma] model changed; reloading old=%s new=%s"
                    % (getattr(self._backend, "model_path", "<mock>"), msg.model_path),
                    flush=True,
                )
                self._backend = None
                self._backend_ready.clear()
            try:
                self._backend = self._make_backend_from_model_init(msg)
                self._backend_ready.set()
                ack = draft_session_protocol.TcpModelInitAck(
                    status="ok",
                    vocab_size=int(getattr(self._backend, "vocab_size", 0)),
                    eos_token_id=int(getattr(self._backend, "eos_token_id", 0)),
                    data_port=int(self._args.port),
                )
            except Exception as exc:
                self._backend = None
                ack = draft_session_protocol.TcpModelInitAck(status="error", error=str(exc))
                print("[draft-rdma] model_init failed: %r" % exc, flush=True)
            client.sendall(draft_session_protocol.serialize(ack))

    def run(self):
        self._start_tcp_prompt_listener()
        print(
            "[draft-rdma] waiting for TcpModelInit on port %d ..." % self._args.control_port,
            flush=True,
        )
        if not self._backend_ready.wait(timeout=float(self._args.backend_init_timeout_s)):
            raise RuntimeError("timed out waiting for TcpModelInit")
        if self._stop:
            return
        print(
            "[draft-rdma] backend ready; opening %s data-plane port %d"
            % (self._data_transport, self._args.port),
            flush=True,
        )
        self._endpoint, self._channel = self._new_channel()
        s = self._channel.start()
        if s != SpecDecodeChannel.Status.kOk:
            raise RuntimeError("channel.start failed: " + SpecDecodeChannel.to_string(s))
        s = self._channel.prime_recv(self._max_num_requests)
        if s not in (SpecDecodeChannel.Status.kOk, SpecDecodeChannel.Status.kEndpointQueueFull):
            raise RuntimeError("prime_recv failed: " + SpecDecodeChannel.to_string(s))

        print(
            "[draft-rdma] server listening: nic=%s port=%d max_requests=%d backend=%s"
            % (
                self._args.nic,
                self._args.port,
                self._max_num_requests,
                type(self._backend).__name__,
            ),
            flush=True,
        )

        # Auto-bind newly-seen request_ids to the slot they were sent on.
        # This implements lazy route binding on the server.
        seen_routes = {}

        while not self._stop:
            status, request_id, received = self._channel.pump_once_for_bound_request()

            if status == SpecDecodeChannel.Status.kEndpointNotStarted:
                break
            if status == SpecDecodeChannel.Status.kEndpointEmpty:
                time.sleep(0.0005)
                continue

            if status == SpecDecodeChannel.Status.kRouteNotFound and received is not None:
                route = SpecDecodeChannel.Route(
                    stream_id=int(received.imm_data.stream_id),
                    slot=int(received.imm_data.slot),
                )
                # On the server side, the slot itself is our request handle —
                # so use slot as request_id.  This matches the target's
                # default route probe (slot = request_id & 0x0FFF).
                handle = int(received.imm_data.slot)
                bind_status = self._channel.bind_request_route(handle, route)
                if bind_status != SpecDecodeChannel.Status.kOk:
                    print(
                        "[draft-rdma] bind failed slot=%s status=%s"
                        % (route.slot, SpecDecodeChannel.to_string(bind_status)),
                        flush=True,
                    )
                    continue
                seen_routes[route] = handle
                request_id = handle

            if status not in (
                SpecDecodeChannel.Status.kOk,
                SpecDecodeChannel.Status.kRouteNotFound,
            ):
                print(
                    "[draft-rdma] pump status=%s — skipping" % SpecDecodeChannel.to_string(status),
                    flush=True,
                )
                continue

            if received is None or request_id is None:
                continue

            if int(received.imm_data.msg_type) != int(DraftApiProtocol.MessageType.kTargetToDraft):
                print(
                    "[draft-rdma] unexpected msg_type=%s — dropping" % received.imm_data.msg_type,
                    flush=True,
                )
                continue

            self._round_count += 1
            last_token = int(received.message.tokens[0])
            position = int(received.message.position)
            round_seq = int(received.message.round_seq_num)

            try:
                draft_tokens = self._backend.step(
                    request_id=int(request_id),
                    last_token=last_token,
                    position=position,
                    round_seq=round_seq,
                )
            except Exception as exc:
                print("[draft-rdma] backend.step failed: %r" % exc, flush=True)
                continue

            tokens = list(draft_tokens) + [0] * (DraftApiProtocol.kMaxTokens - len(draft_tokens))
            response = DraftApiProtocol.Message(
                message_type=DraftApiProtocol.MessageType.kDraftToTarget,
                round_seq_num=round_seq,
                position=0,
                num_tokens=min(len(draft_tokens), DraftApiProtocol.kMaxTokens),
                tokens=tokens,
            )
            send_status = self._channel.send_for_request(
                int(request_id),
                msg_type=int(DraftApiProtocol.MessageType.kDraftToTarget),
                message=response,
            )
            if send_status != SpecDecodeChannel.Status.kOk:
                print(
                    "[draft-rdma] send failed: %s" % SpecDecodeChannel.to_string(send_status),
                    flush=True,
                )
                break
            if self._round_count <= 5 or self._round_count % 50 == 0:
                print(
                    "[draft-rdma] round=%d req=%d slot=%d round_seq=%d last_token=%d "
                    "→ tokens=%s"
                    % (
                        self._round_count,
                        int(request_id),
                        int(received.imm_data.slot),
                        round_seq,
                        last_token,
                        draft_tokens[:5],
                    ),
                    flush=True,
                )

        self._channel.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nic", default="", help="RDMA device. Empty means use transport default.")
    ap.add_argument(
        "--port",
        type=int,
        default=0,
        help="Data-plane port. 0 means allocate a free port and return it in TcpModelInitAck.",
    )
    ap.add_argument(
        "--backend",
        choices=["mock", "transformers", "trtllm", "trtllm-executor"],
        default="trtllm",
    )
    ap.add_argument(
        "--control-port",
        type=int,
        default=0,
        help="TCP control-plane port for model-init / prompt-init / cancel",
    )
    # transformers-backend args
    ap.add_argument(
        "--prompt", default=None, help="fixed prompt for stateful sessions (transformers backend)"
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--trtllm-stream-max-tokens",
        type=int,
        default=int(os.environ.get("PEARL_TRTLLM_STREAM_MAX_TOKENS", "2048")),
        help="Max tokens for each resident TRT-LLM streaming decode request before restart.",
    )
    ap.add_argument("--backend-init-timeout-s", type=float, default=600.0)
    ap.add_argument("--handshake-timeout-s", type=float, default=300.0)
    args = ap.parse_args()
    if args.control_port <= 0:
        ap.error("--control-port is required")

    server = IzzyCompatibleDraftServer(args)

    def _sig(*_):
        print("[draft-rdma] received signal, shutting down...", flush=True)
        server.stop()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    server.run()


if __name__ == "__main__":
    main()
