import unittest

import torch

from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams


class TestMultimodalParamsHandleConversion(unittest.TestCase):
    """Test cases for to_handle and to_tensor methods in MultimodalParams."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample cpu tensors for testing (shared cuda tensor using cudaIPC only works between processes)
        self.mm_embedding = torch.randn(3, 4, 5)
        self.mrope_config = {
            "mrope_rotary_cos_sin": torch.randn(2, 3),
            "mrope_position_deltas": torch.randn(5),
        }
        self.image = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "image_height": [224],
            "image_width": [224],
        }
        # Create sample multimodal data structure
        self.sample_multimodal_data = {
            "multimodal_embedding": self.mm_embedding,
            "mrope_config": self.mrope_config,
            "image": self.image,
        }

    def test_to_handle_none_multimodal_data(self):
        """Test to_handle with None multimodal_data."""
        params = MultimodalParams()
        params.multimodal_data = None

        params.to_handle("multimodal_data")
        self.assertIsNone(params.multimodal_data)
        params.multimodal_data = {}
        params.to_handle("multimodal_data")
        self.assertEqual(params.multimodal_data, {})

        params = MultimodalParams()
        multimodal_input = MultimodalInput(
            multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]] * 2,
            multimodal_positions=[0, 10],
            multimodal_lengths=[2, 2])
        params.multimodal_input = multimodal_input
        params.to_handle("multimodal_input")
        self.assertEqual(params.multimodal_input, multimodal_input)

    def test_to_tensor_basic_handle(self):
        """Test converting a basic handle back to tensor."""
        params = MultimodalParams()
        params.multimodal_data = {"multimodal_embedding": self.mm_embedding}

        # Convert to handle
        params.to_handle("multimodal_data", key="multimodal_embedding")
        # Convert back to tensor
        params.to_tensor("multimodal_data", key="multimodal_embedding")

        result = params.multimodal_data["multimodal_embedding"]
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.allclose(result, self.mm_embedding))

    def test_to_tensor_all_handles(self):
        """Test that to_handle followed by to_tensor preserves data integrity."""
        params = MultimodalParams()
        params.multimodal_data = self.sample_multimodal_data.copy()

        params.to_handle("multimodal_data", key=None)
        params.to_tensor("multimodal_data", key=None)

        self.assertTrue(
            torch.allclose(params.multimodal_data["multimodal_embedding"],
                           self.mm_embedding))
        self.assertTrue(
            torch.allclose(
                params.multimodal_data["mrope_config"]["mrope_rotary_cos_sin"],
                self.mrope_config["mrope_rotary_cos_sin"]))
        self.assertTrue(
            torch.allclose(
                params.multimodal_data["mrope_config"]["mrope_position_deltas"],
                self.mrope_config["mrope_position_deltas"]))
        self.assertTrue(
            torch.allclose(params.multimodal_data["image"]["pixel_values"],
                           self.image["pixel_values"]))
        self.assertEqual(params.multimodal_data["image"]["image_height"],
                         self.image["image_height"])
        self.assertEqual(params.multimodal_data["image"]["image_width"],
                         self.image["image_width"])


if __name__ == "__main__":
    unittest.main()
