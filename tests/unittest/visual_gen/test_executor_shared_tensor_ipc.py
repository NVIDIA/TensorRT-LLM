# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-trip tests for worker->client media IPC via SharedTensorContainer handles.

The producer (worker) replaces ``PipelineOutput`` media tensors with handle dicts;
the client rebuilds them. CUDA IPC handles cannot be opened in the process that
created them, so the producer runs in a spawned subprocess and the consumer in the
main process, mirroring the real worker/client split.
"""

import multiprocessing as mp
import os
import unittest
from unittest import mock

import torch
import zmq

from tensorrt_llm._torch.visual_gen.executor import (
    DiffusionExecutor,
    DiffusionRequest,
    DiffusionResponse,
    find_free_port,
    run_diffusion_worker,
)
from tensorrt_llm._torch.visual_gen.output import PipelineOutput
from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.visual_gen import VisualGenArgs, VisualGenParams


def _producer(q, device):
    """Build a PipelineOutput, convert its media tensors to handle dicts, hand off."""
    try:
        video = torch.arange(2 * 3 * 4 * 5 * 3, dtype=torch.uint8).reshape(2, 3, 4, 5, 3).to(device)
        audio = torch.randn(2, 2, 100).to(device)
        output = PipelineOutput(video=video, audio=audio, frame_rate=24.0)
        output.to_handle()
        q.put(("ok", output, video.cpu(), audio.cpu()))
        # Stay alive until the consumer has rebuilt the views (IPC refcount).
        q.get()
    except Exception as e:  # noqa: BLE001 - surface producer errors to the test
        q.put(("error", repr(e), None, None))


class TestExecutorSharedTensorIPC(unittest.TestCase):
    def _run_roundtrip(self, device):
        mp.set_start_method("spawn", force=True)
        q = mp.Queue()
        try:
            producer = mp.Process(target=_producer, args=(q, device))
            producer.start()
            status, output, ref_video, ref_audio = q.get(timeout=120)
            self.assertEqual(status, "ok", msg=str(output))
            # Media fields cross the IPC as handle dicts, not tensors.
            self.assertIsInstance(output.video, dict)
            self.assertIn("method_key", output.video)
            self.assertIsInstance(output.audio, dict)

            resp = DiffusionResponse(request_id=0, output=output)
            resp.output.to_tensor()
            q.put("done")

            self.assertTrue(torch.equal(resp.output.video.cpu(), ref_video))
            self.assertTrue(torch.equal(resp.output.audio.cpu(), ref_audio))
            self.assertEqual(resp.output.frame_rate, 24.0)
            producer.join()
        finally:
            q.close()
            q.join_thread()

    def test_roundtrip_cpu(self):
        self._run_roundtrip("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_roundtrip_cuda(self):
        torch.cuda.set_device(0)
        self._run_roundtrip("cuda")

    def test_restore_noop_without_handles(self):
        """A response carrying plain tensors (handle-passing off) is untouched."""
        video = torch.zeros(1, 2, 3, 4, 3, dtype=torch.uint8)
        resp = DiffusionResponse(request_id=1, output=PipelineOutput(video=video))
        resp.output.to_tensor()
        self.assertIs(resp.output.video, video)

    def test_restore_skipped_for_non_pipeline_output(self):
        """READY responses carry a plain dict output; the client's isinstance guard skips it."""
        resp = DiffusionResponse(request_id=-1, output={"status": "READY"})
        if isinstance(resp.output, PipelineOutput):  # mirrors DiffusionRemoteClient guard
            resp.output.to_tensor()
        self.assertEqual(resp.output, {"status": "READY"})

    def test_roundtrip_local_same_process(self):
        """Same-process handoff: media stays in-process (local=True), not cross-process IPC."""
        video = torch.arange(2 * 3 * 4 * 5 * 3, dtype=torch.uint8).reshape(2, 3, 4, 5, 3)
        audio = torch.randn(2, 2, 100)
        output = PipelineOutput(video=video, audio=audio, frame_rate=24.0)
        output.to_handle(local=True)
        self.assertIsInstance(output.video, dict)
        self.assertIsInstance(output.audio, dict)
        resp = DiffusionResponse(request_id=2, output=output)
        resp.output.to_tensor()
        self.assertTrue(torch.equal(resp.output.video, video))
        self.assertTrue(torch.equal(resp.output.audio, audio))
        self.assertEqual(resp.output.frame_rate, 24.0)


def _expected_video() -> torch.Tensor:
    return torch.arange(2 * 3 * 4 * 5 * 3, dtype=torch.uint8).reshape(2, 3, 4, 5, 3)


class _StubPipeline:
    """Stand-in for BasePipeline covering exactly what the executor touches."""

    default_generation_params: dict = {}
    extra_param_specs: dict = {}
    _warmed_up_shapes: set = set()

    def warmup_cache_key(self, height, width, num_frames):
        return (height, width, num_frames)

    def infer(self, req):
        return PipelineOutput(video=_expected_video(), frame_rate=24.0)

    def cleanup(self):
        pass


def _stub_load_pipeline(self):
    self.pipeline = _StubPipeline()


def _diffusion_worker_entry(rank, world_size, master_port, req_addr, resp_addr, req_key, resp_key):
    """Run the real ``run_diffusion_worker`` with only the pipeline stubbed out.

    The env scrub mirrors a worker spawned by a plain single-process client;
    CUDA is forced off so dist init picks gloo and the stub stays on CPU.
    """
    for var in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "SLURM_PROCID",
        "SLURM_NTASKS",
    ):
        os.environ.pop(var, None)

    with (
        mock.patch("torch.cuda.is_available", return_value=False),
        mock.patch.object(DiffusionExecutor, "_load_pipeline", _stub_load_pipeline),
    ):
        run_diffusion_worker(
            rank=rank,
            world_size=world_size,
            master_addr="127.0.0.1",
            master_port=master_port,
            request_queue_addr=req_addr,
            response_queue_addr=resp_addr,
            visual_gen_args=VisualGenArgs(model="/tmp/model"),
            req_hmac_key=req_key,
            resp_hmac_key=resp_key,
            local_rank=rank,
        )


class TestSpawnWorkerResponseHandleMode(unittest.TestCase):
    """A spawned worker must return cross-process response handles.

    Drives the real ``run_diffusion_worker`` + serve loop + ZMQ response path
    with world_size=2 (gloo, CPU) and asserts the client-side rebuild
    succeeds. Any wrongly-local handle mode (e.g. derived from the
    environment, which the worker entry mutates for dist init) ships a
    same-process REBUILD_LOCAL handle the client cannot rebuild.
    """

    def test_spawn_worker_response_rebuilds_in_client(self):
        ctx = mp.get_context("spawn")
        world_size = 2
        master_port = find_free_port()
        req_port, resp_port = find_free_port(), find_free_port()
        req_key, resp_key = os.urandom(32), os.urandom(32)

        # Client-side server sockets, mirroring DiffusionRemoteClient.
        requests_ipc = ZeroMqQueue(
            (f"tcp://0.0.0.0:{req_port}", req_key),
            is_server=True,
            socket_type=zmq.PUSH,
            use_hmac_encryption=True,
        )
        responses_ipc = ZeroMqQueue(
            (f"tcp://0.0.0.0:{resp_port}", resp_key),
            is_server=True,
            socket_type=zmq.PULL,
            use_hmac_encryption=True,
        )

        workers = [
            ctx.Process(
                target=_diffusion_worker_entry,
                args=(
                    rank,
                    world_size,
                    master_port,
                    f"tcp://127.0.0.1:{req_port}",
                    f"tcp://127.0.0.1:{resp_port}",
                    req_key,
                    resp_key,
                ),
            )
            for rank in range(world_size)
        ]
        try:
            for p in workers:
                p.start()

            requests_ipc.put(
                DiffusionRequest(
                    request_id=7,
                    prompt=["stub"],
                    params=VisualGenParams(height=8, width=8, num_frames=2),
                )
            )

            resp = None
            while resp is None or resp.request_id != 7:
                self.assertTrue(responses_ipc.poll(timeout=180), "no response from spawned workers")
                resp = responses_ipc.get()

            self.assertIsNone(resp.error_msg, msg=str(resp.error_msg))
            # A wrongly-local handle raises here (crossed a process boundary).
            resp.output.to_tensor()
            self.assertTrue(torch.equal(resp.output.video, _expected_video()))

            requests_ipc.put(None)  # shutdown broadcast
            for p in workers:
                p.join(timeout=60)
                self.assertEqual(p.exitcode, 0)
        finally:
            for p in workers:
                if p.is_alive():
                    p.terminate()
            for p in workers:
                p.join(timeout=30)
            for q in (requests_ipc, responses_ipc):
                if q.socket:
                    q.socket.setsockopt(zmq.LINGER, 0)
                q.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    unittest.main()
