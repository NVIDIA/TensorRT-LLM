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

    from tensorrt_llm._torch.speculative.draft_api_protocol import (
        DraftApiProtocol,  # noqa: F401,E402
    )
    from tensorrt_llm._torch.speculative.ibverbs_endpoint import (  # noqa: F401,E402
        IbverbsEndpointConfig,
        _IbverbsRdmaBackend,
    )
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
        # Mock backend doesn't need real prefill; just remember length.
        self._sessions.pop(int(request_id), None)
        self._sessions[request_id] = len(prompt_tokens)
        return len(prompt_tokens) - 1  # last_token_position

    def step(self, request_id, last_token, position, round_seq):
        # Encode the round so we can verify ordering on the target side.
        return [((round_seq * 1000 + i) % (1 << 31)) for i in range(1, self.max_draft_len + 1)]


class _TrtLlmBackend:
    """Real per-request stateful inference using TensorRT-LLM.

    Compared to ``_TransformersBackend``, this:
    - Uses TRT-LLM's optimized PyTorch backend (much faster than HF transformers)
    - Lets TRT-LLM's KV-cache-block reuse handle the prefix-sharing across
      decode rounds — we just hand it the full conversation token list and
      it picks up the common prefix from cache.
    - Avoids the DynamicCache.crop() dance (that was specific to the
      transformers tuple/Cache API).

    Per-request "session" here is just an accumulated token list; the
    expensive KV cache lives inside TRT-LLM and is keyed by prefix hash.
    """

    def __init__(self, model_path, prompt, max_draft_len, device="cuda:0"):
        from transformers import AutoTokenizer

        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig

        self.model_path = str(model_path)
        self.max_draft_len = max_draft_len
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
        # Greedy sampling so the draft tokens are deterministic.
        self.sampling_params = SamplingParams(
            max_tokens=max_draft_len,
            temperature=0.0,
            top_p=1.0,
        )
        self.fixed_prompt_ids = None
        if prompt:
            self.fixed_prompt_ids = self.tokenizer.encode(prompt)
        # request_id -> {"tokens": list[int]}.  Tokens accumulates the
        # full conversation prefix; TRT-LLM block-reuse gives us the
        # per-round savings without manual KV cache bookkeeping.
        self._sessions = {}

    def reset_sessions(self):
        self._sessions.clear()

    def matches_model_init(self, msg):
        return self.model_path == str(msg.model_path) and int(self.max_draft_len) == int(
            msg.max_draft_len
        )

    def init_session(self, request_id, prompt_tokens):
        self._sessions.pop(int(request_id), None)
        self._sessions[request_id] = {"tokens": list(prompt_tokens)}
        return len(prompt_tokens) - 1

    def _ensure_session(self, request_id):
        sess = self._sessions.get(request_id)
        if sess is not None:
            return sess
        if self.fixed_prompt_ids is None:
            raise RuntimeError(
                "no session for request %d and no fixed prompt set; "
                "send a TcpPromptInit first" % request_id
            )
        sess = {"tokens": list(self.fixed_prompt_ids)}
        self._sessions[request_id] = sess
        return sess

    def step(self, request_id, last_token, position, round_seq):
        sess = self._ensure_session(request_id)
        # Rewind on reject: target tells us where it actually stopped.
        # ``position`` is target's "next position to be filled" — equal
        # to the number of tokens already committed on its side.  We
        # truncate our local accumulator to that length, then append
        # the target's accepted last_token.
        if position < len(sess["tokens"]):
            sess["tokens"] = sess["tokens"][: int(position)]
        if not sess["tokens"] or sess["tokens"][-1] != int(last_token):
            sess["tokens"].append(int(last_token))
        # Generate max_draft_len tokens given the current prefix.
        outputs = self.llm.generate([sess["tokens"]], sampling_params=self.sampling_params)
        draft_tokens = list(outputs[0].outputs[0].token_ids)[: self.max_draft_len]
        sess["tokens"].extend(int(t) for t in draft_tokens)
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

    def step(self, request_id, last_token, position, round_seq):
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
        for _ in range(self.max_draft_len):
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
        if self._tcp_listener_socket is not None:
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
        if self._args.backend == "trtllm":
            return _TrtLlmBackend(
                model_path=msg.model_path,
                prompt=self._args.prompt,
                max_draft_len=int(msg.max_draft_len),
                device=self._args.device,
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
    ap.add_argument("--backend", choices=["mock", "transformers", "trtllm"], default="trtllm")
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
