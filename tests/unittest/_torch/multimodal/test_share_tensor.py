import unittest
import multiprocessing as mp
import torch

from tensorrt_llm._torch.multimodal.mm_utils import SharedTensorContainer, _SharedTensorRebuildMethodRegistry


class TestShareTensor(unittest.TestCase):
    """Test cases for sharing tensors between processes."""

    @classmethod
    def setUpClass(cls):
        """Initialize the registry before running tests."""
        _SharedTensorRebuildMethodRegistry.initialize()

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
        except Exception as e:
            q.put(('error', str(e)))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_share_cuda_tensor(self):
        """Test CUDA tensor sharing between processes."""
        mp.set_start_method('spawn', force=True)
        queue = mp.Queue()

        # Producer process
        producer = mp.Process(target=self._producer, args=(queue, self.ref_tensor, "cuda"))
        producer.start()
        status, data = queue.get(timeout=100)
        # Verify
        self.assertEqual(status, 'success')
        reconstructed = SharedTensorContainer.from_dict(data).get_local_view()
        self.assertTrue(torch.allclose(reconstructed.cpu(), self.ref_tensor))
        del reconstructed
        producer.join()

    def test_share_cpu_tensor(self):
        """Test CPU tensor sharing between processes."""
        mp.set_start_method('spawn', force=True)
        queue = mp.Queue()

        # Producer process
        producer = mp.Process(target=self._producer, args=(queue, self.ref_tensor, "cpu"))
        producer.start()
        status, data = queue.get(timeout=100)
        # Verify
        self.assertEqual(status, 'success')
        reconstructed = SharedTensorContainer.from_dict(data).get_local_view()
        self.assertTrue(torch.allclose(reconstructed, self.ref_tensor))
        producer.join()

    def test_share_tensor_different_shapes(self):
        """Test CPU tensor sharing with different tensor shapes."""
        mp.set_start_method('spawn', force=True)
        queue = mp.Queue()
        test_shapes = [
            (1,),           
            (2, 3),         
            (1, 2, 3, 4),   
            (10,),          
        ]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                test_tensor = torch.randn(shape)
                producer = mp.Process(target=self._producer, args=(queue, test_tensor, "cpu"))
                producer.start()
                status, data = queue.get(timeout=100)
                
                self.assertEqual(status, 'success')
                reconstructed = SharedTensorContainer.from_dict(data).get_local_view()
                self.assertTrue(torch.allclose(reconstructed, test_tensor))
                producer.join()
            
            with self.subTest(shape=shape):
                test_tensor = torch.randn(shape)
                producer = mp.Process(target=self._producer, args=(queue, test_tensor, "cuda"))
                producer.start()
                status, data = queue.get(timeout=100)
                self.assertEqual(status, 'success')
                reconstructed = SharedTensorContainer.from_dict(data).get_local_view()
                self.assertTrue(torch.allclose(reconstructed, test_tensor.cuda()))
                del reconstructed
                producer.join()

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

        for dtype in test_dtypes:
            with self.subTest(dtype=dtype):
                test_tensor = torch.randn(2, 3).to(dtype)
                producer = mp.Process(target=self._producer, args=(queue, test_tensor, "cpu"))
                producer.start()
                status, data = queue.get(timeout=100)
                
                self.assertEqual(status, 'success')
                reconstructed = SharedTensorContainer.from_dict(data).get_local_view()
                self.assertTrue(torch.allclose(reconstructed, test_tensor))
                self.assertEqual(reconstructed.dtype, test_tensor.dtype)
                producer.join()
            
            with self.subTest(dtype=dtype):
                test_tensor = torch.randn(2, 3).to(dtype)
                producer = mp.Process(target=self._producer, args=(queue, test_tensor, "cuda"))
                producer.start()
                status, data = queue.get(timeout=100)
                self.assertEqual(status, 'success')
                reconstructed = SharedTensorContainer.from_dict(data).get_local_view()
                self.assertTrue(torch.allclose(reconstructed, test_tensor.cuda()))
                self.assertEqual(reconstructed.dtype, test_tensor.dtype)
                del reconstructed
                producer.join()

        

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    unittest.main() 