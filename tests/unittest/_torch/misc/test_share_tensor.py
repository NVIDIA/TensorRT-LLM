# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from typing import Literal

import torch

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer


class TestShareTensor(unittest.TestCase):
    """Test cases for sharing tensors between processes."""

    def setUp(self):
        """Set up test fixtures."""
        self.ref_tensor = torch.randn(3, 4, 5)
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            torch.cuda.set_device(0)

    @staticmethod
    def _producer(q: Queue, consumed: Event, tensor: torch.Tensor,
                  device: Literal["cpu", "cuda"]) -> None:
        """Keep the shared tensor alive until the consumer has finished."""
        try:
            tensor = tensor.to(device)
            container = SharedTensorContainer.from_tensor(tensor)
            q.put(('success', container.dump_to_dict()))

            # A separate signal prevents the producer from consuming its own reply.
            consumed.wait()
        except Exception as e:
            q.put(('error', str(e)))

    @contextmanager
    def _shared_tensor(
            self, tensor: torch.Tensor,
            device: Literal["cpu", "cuda"]) -> Iterator[torch.Tensor]:
        context = mp.get_context('spawn')
        queue = context.Queue()
        consumed = context.Event()
        producer = context.Process(target=self._producer,
                                   args=(queue, consumed, tensor, device))
        producer.start()
        try:
            status, data = queue.get(timeout=100)
            self.assertEqual(status, 'success', data)
            yield SharedTensorContainer.from_dict(data).get_local_view()
        finally:
            consumed.set()
            producer.join(timeout=10)
            if producer.is_alive():
                producer.terminate()
                producer.join(timeout=10)
            queue.close()
            queue.join_thread()
        self.assertEqual(producer.exitcode, 0)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_share_cuda_tensor(self) -> None:
        """Test CUDA tensor sharing between processes."""
        with self._shared_tensor(self.ref_tensor, "cuda") as reconstructed:
            self.assertTrue(torch.allclose(reconstructed.cpu(),
                                           self.ref_tensor))

    def test_share_cpu_tensor(self) -> None:
        """Test CPU tensor sharing between processes."""
        with self._shared_tensor(self.ref_tensor, "cpu") as reconstructed:
            self.assertTrue(torch.allclose(reconstructed, self.ref_tensor))

    def test_share_tensor_different_shapes(self) -> None:
        """Test CPU and CUDA tensor sharing with different shapes."""
        test_shapes = [(1, ), (2, 3), (1, 2, 3, 4), (10, )]
        devices = ("cpu", "cuda") if self.cuda_available else ("cpu", )
        for shape in test_shapes:
            for device in devices:
                with self.subTest(shape=shape, device=device):
                    test_tensor = torch.randn(shape)
                    with self._shared_tensor(test_tensor,
                                             device) as reconstructed:
                        self.assertTrue(
                            torch.allclose(reconstructed.cpu(), test_tensor))

    def test_share_tensor_different_dtypes(self) -> None:
        """Test CPU and CUDA tensor sharing with different data types."""
        test_dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
        devices = ("cpu", "cuda") if self.cuda_available else ("cpu", )
        for dtype in test_dtypes:
            for device in devices:
                # xdist cannot serialize torch.dtype in subtest reports.
                with self.subTest(dtype=str(dtype), device=device):
                    test_tensor = torch.randn(2, 3).to(dtype)
                    with self._shared_tensor(test_tensor,
                                             device) as reconstructed:
                        self.assertTrue(
                            torch.allclose(reconstructed.cpu(), test_tensor))
                        self.assertEqual(reconstructed.dtype, test_tensor.dtype)

    def test_producer_cleanup_after_consumer_error(self) -> None:
        """A consumer failure propagates after its producer exits cleanly."""
        existing_pids = {child.pid for child in mp.active_children()}
        producer = None
        error = RuntimeError("Consumer failed")
        with self.assertRaises(RuntimeError) as raised:
            with self._shared_tensor(self.ref_tensor, "cpu"):
                producers = [
                    child for child in mp.active_children()
                    if child.pid not in existing_pids
                ]
                self.assertEqual(len(producers), 1)
                producer = producers[0]
                raise error

        self.assertIs(raised.exception, error)
        self.assertIsNotNone(producer)
        self.assertFalse(producer.is_alive())
        self.assertEqual(producer.exitcode, 0)
        self.assertNotIn(producer.pid,
                         {child.pid
                          for child in mp.active_children()})

    @staticmethod
    def _stand_by_producer(conn):
        """Long-lived producer that creates new tensors on demand."""
        try:
            while True:
                msg = conn.recv()
                if msg == "get":
                    # Create a new tensor each time
                    tensor = torch.randn(100, 100, 100).cuda()  # ~4MB tensor
                    container = SharedTensorContainer.from_tensor(tensor)
                    serialized_data = container.dump_to_dict()
                    memory_usage = torch.cuda.memory_allocated() / (1024 * 1024)
                    conn.send(('success', serialized_data, memory_usage))
                elif msg == "exit":
                    break
                else:
                    print(f"Unknown command: {msg}")
        except Exception as e:
            conn.send(('error', str(e)))
        finally:
            conn.close()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_memory_leak_repeated_producer(self):
        """Test to check no memory leak when producer creates new tensors repeatedly.

        This test keeps the producer alive and requests multiple tensors.
        Each iteration, the producer creates a new tensor and shares it.
        If the consumer properly rebuild and cleanup, GPU memory usage will likely be stable.
        """
        import gc

        import numpy as np

        mp.set_start_method('spawn', force=True)

        # Reset GPU state before test
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Record initial memory state
        initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)

        parent_conn, child_conn = mp.Pipe()
        producer = mp.Process(target=self._stand_by_producer,
                              args=(child_conn, ))
        producer.start()

        memory_measurements = []
        try:
            for i in range(10):
                parent_conn.send("get")
                status, data, memory_usage = parent_conn.recv()
                memory_measurements.append(memory_usage)
                self.assertEqual(status, 'success')

                container = SharedTensorContainer.from_dict(
                    data).get_local_view()
                del container
                gc.collect()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            relative_measurements = [
                m - initial_memory for m in memory_measurements
            ]

            warmup_iterations = 4
            stable_measurements = relative_measurements[warmup_iterations:]

            x = np.arange(len(stable_measurements))
            slope, _ = np.polyfit(x, stable_measurements, 1)

            self.assertLess(
                abs(slope), 0.2,
                f"Memory leak detected! Relative slope: {slope:.3f} MB/iteration. "
                f"Relative measurements: {relative_measurements}")

        finally:
            parent_conn.send("exit")
            producer.join()
            parent_conn.close()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    unittest.main()
