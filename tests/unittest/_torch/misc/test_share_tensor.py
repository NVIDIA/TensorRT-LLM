import multiprocessing as mp
import unittest

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
    def _producer(q, tensor, device=None):
        """Producer: create tensor and share it."""
        try:
            if device is not None:
                if device == "cuda":
                    tensor = tensor.cuda()
                elif device == "cpu":
                    tensor = tensor.cpu()
            container = SharedTensorContainer.from_tensor(tensor)
            q.put(('success', container.dump_to_dict()))

            # Wait for consumer to signal it's done
            # This keeps the producer alive until ownership is transferred to consumer
            q.get()

        except Exception as e:
            q.put(('error', str(e)))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_share_cuda_tensor(self):
        """Test CUDA tensor sharing between processes."""
        mp.set_start_method('spawn', force=True)
        queue = mp.Queue()

        try:
            # Producer process
            producer = mp.Process(target=self._producer,
                                  args=(queue, self.ref_tensor, "cuda"))
            producer.start()
            status, data = queue.get(timeout=100)
            # Verify
            self.assertEqual(status, 'success')
            reconstructed = SharedTensorContainer.from_dict(
                data).get_local_view()
            queue.put(
                'done'
            )  # producer can be released as early as here as ownership is transferred to consumer
            self.assertTrue(torch.allclose(reconstructed.cpu(),
                                           self.ref_tensor))
            del reconstructed
            producer.join()
        finally:
            # Explicit cleanup to prevent QueueFeederThread leak
            queue.close()
            queue.join_thread()

    def test_share_cpu_tensor(self):
        """Test CPU tensor sharing between processes."""
        mp.set_start_method('spawn', force=True)
        queue = mp.Queue()

        try:
            # Producer process
            producer = mp.Process(target=self._producer,
                                  args=(queue, self.ref_tensor, "cpu"))
            producer.start()
            status, data = queue.get(timeout=100)
            # Verify
            self.assertEqual(status, 'success')
            reconstructed = SharedTensorContainer.from_dict(
                data).get_local_view()
            queue.put(
                'done'
            )  # producer can be released as early as here as ownership is transferred to consumer
            self.assertTrue(torch.allclose(reconstructed, self.ref_tensor))
            producer.join()
        finally:
            # Explicit cleanup to prevent QueueFeederThread leak
            queue.close()
            queue.join_thread()

    def test_share_tensor_different_shapes(self):
        """Test CPU tensor sharing with different tensor shapes."""
        mp.set_start_method('spawn', force=True)
        queue = mp.Queue()
        test_shapes = [
            (1, ),
            (2, 3),
            (1, 2, 3, 4),
            (10, ),
        ]

        try:
            for shape in test_shapes:
                with self.subTest(shape=shape):
                    test_tensor = torch.randn(shape)
                    producer = mp.Process(target=self._producer,
                                          args=(queue, test_tensor, "cpu"))
                    producer.start()
                    status, data = queue.get(timeout=100)

                    self.assertEqual(status, 'success')
                    reconstructed = SharedTensorContainer.from_dict(
                        data).get_local_view()
                    self.assertTrue(torch.allclose(reconstructed, test_tensor))
                    queue.put('done')
                    producer.join()

                with self.subTest(shape=shape):
                    test_tensor = torch.randn(shape)
                    producer = mp.Process(target=self._producer,
                                          args=(queue, test_tensor, "cuda"))
                    producer.start()
                    status, data = queue.get(timeout=100)
                    self.assertEqual(status, 'success')
                    reconstructed = SharedTensorContainer.from_dict(
                        data).get_local_view()
                    self.assertTrue(
                        torch.allclose(reconstructed, test_tensor.cuda()))
                    del reconstructed
                    queue.put('done')
                    producer.join()
        finally:
            # Explicit cleanup to prevent QueueFeederThread leak
            queue.close()
            queue.join_thread()

    def test_share_tensor_different_dtypes(self):
        """Test CPU tensor sharing with different data types."""
        mp.set_start_method('spawn', force=True)
        queue = mp.Queue()

        # Test different data types
        test_dtypes = [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        ]

        try:
            for dtype in test_dtypes:
                with self.subTest(dtype=dtype):
                    test_tensor = torch.randn(2, 3).to(dtype)
                    producer = mp.Process(target=self._producer,
                                          args=(queue, test_tensor, "cpu"))
                    producer.start()
                    status, data = queue.get(timeout=100)

                    self.assertEqual(status, 'success')
                    reconstructed = SharedTensorContainer.from_dict(
                        data).get_local_view()
                    self.assertTrue(torch.allclose(reconstructed, test_tensor))
                    self.assertEqual(reconstructed.dtype, test_tensor.dtype)
                    queue.put('done')
                    producer.join()

                with self.subTest(dtype=dtype):
                    test_tensor = torch.randn(2, 3).to(dtype)
                    producer = mp.Process(target=self._producer,
                                          args=(queue, test_tensor, "cuda"))
                    producer.start()
                    status, data = queue.get(timeout=100)
                    self.assertEqual(status, 'success')
                    reconstructed = SharedTensorContainer.from_dict(
                        data).get_local_view()
                    self.assertTrue(
                        torch.allclose(reconstructed, test_tensor.cuda()))
                    self.assertEqual(reconstructed.dtype, test_tensor.dtype)
                    del reconstructed
                    queue.put('done')
                    producer.join()
        finally:
            # Explicit cleanup to prevent QueueFeederThread leak
            queue.close()
            queue.join_thread()

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
