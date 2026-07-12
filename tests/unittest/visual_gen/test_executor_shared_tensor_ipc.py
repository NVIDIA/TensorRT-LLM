# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-trip tests for worker->client media IPC via SharedTensorContainer handles.

The producer (worker) replaces ``PipelineOutput`` media tensors with handle dicts;
the client rebuilds them. CUDA IPC handles cannot be opened in the process that
created them, so the producer runs in a spawned subprocess and the consumer in the
main process, mirroring the real worker/client split.
"""

import multiprocessing as mp
import unittest

import torch

from tensorrt_llm._torch.visual_gen.executor import DiffusionResponse
from tensorrt_llm._torch.visual_gen.output import PipelineOutput


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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    unittest.main()
