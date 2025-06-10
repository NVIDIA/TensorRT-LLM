import unittest
import multiprocessing as mp
import torch

from tensorrt_llm._torch.multimodal.mm_utils import SharedTensorContainer, _SharedTensorRebuildMethodRegistry


class TestShareCudaTensor(unittest.TestCase):
    """Test cases for sharing CUDA tensor between processes."""

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
        """Producer: create CUDA tensor and share it."""
        try:
            if device is not None:
                torch.cuda.set_device(device)
                tensor = tensor.cuda()
            container = SharedTensorContainer.from_tensor(tensor)
            q.put(('success', container.dump_to_dict()))
        except Exception as e:
            q.put(('error', str(e)))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_share_cuda_tensor(self):
        """Test tensor sharing between processes."""
        mp.set_start_method('spawn', force=True)
        queue = mp.Queue()

        # Producer process
        producer = mp.Process(target=self._producer, args=(queue, self.ref_tensor, 0))
        producer.start()
        status, data = queue.get(timeout=100)
        # Verify
        self.assertEqual(status, 'success')
        reconstructed = SharedTensorContainer.from_dict(data).get_local_view()
        self.assertTrue(torch.allclose(reconstructed.cpu(), self.ref_tensor))
        producer.join()
        

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    unittest.main() 