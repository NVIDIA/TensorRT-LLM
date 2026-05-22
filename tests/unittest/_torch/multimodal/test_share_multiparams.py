import gc
import pickle
import unittest

import torch

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams
from tensorrt_llm.sampling_params import SamplingParams


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

    def test_to_handle_unsupported_element(self):
        """Test to_handle raises ValueError for unsupported elements."""
        params = MultimodalParams()
        multimodal_input = MultimodalInput(
            multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]] * 2,
            multimodal_positions=[0, 10],
            multimodal_lengths=[2, 2])
        params.multimodal_input = multimodal_input

        with self.assertRaises(ValueError) as context:
            params.to_handle("multimodal_input")

        self.assertIn("Unsupported element 'multimodal_input'",
                      str(context.exception))

    def test_to_tensor_basic_handle(self):
        """Test converting a basic handle back to tensor."""
        params = MultimodalParams()
        params.multimodal_data = {"multimodal_embedding": self.mm_embedding}

        # Convert to handle
        params.to_handle("multimodal_data")
        # Convert back to tensor
        params.to_tensor("multimodal_data")

        result = params.multimodal_data["multimodal_embedding"]
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.allclose(result, self.mm_embedding))

    def test_to_tensor_can_clone_shared_storage(self):
        """Test restoring handles into tensors detached from shared storage."""
        source = torch.arange(6, dtype=torch.float32)
        params = MultimodalParams(multimodal_data={
            "mrope_config": {
                "mrope_position_deltas": source,
            }
        })

        params.to_handle("multimodal_data")
        params._shared_tensor_lifetime_refs.clear()
        params.to_tensor("multimodal_data", clone_shared_tensors=True)

        result = params.multimodal_data["mrope_config"]["mrope_position_deltas"]
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.allclose(result, source))
        self.assertEqual(len(params._shared_tensor_lifetime_refs), 0)

        result.add_(10)
        self.assertTrue(
            torch.allclose(source, torch.arange(6, dtype=torch.float32)))

    def test_to_handle_retains_shared_tensor_lifetime_refs(self):
        """Shared tensor handles remain restorable after source locals exit."""
        params = MultimodalParams()

        def populate_params():
            source = torch.arange(6, dtype=torch.float32)
            params.multimodal_data = {"multimodal_embedding": source}
            params.to_handle("multimodal_data")

        populate_params()
        gc.collect()

        handle = params.multimodal_data["multimodal_embedding"]
        restored = SharedTensorContainer.from_dict(handle).get_local_view()
        self.assertTrue(
            torch.allclose(restored, torch.arange(6, dtype=torch.float32)))
        self.assertEqual(len(params._shared_tensor_lifetime_refs), 1)

    def test_generation_result_retains_request_shared_tensor_refs(self):
        """Request handles survive generate_async's multimodal_params cleanup."""

        def build_result_and_handle():
            source = torch.arange(6, dtype=torch.float32)
            params = MultimodalParams(
                multimodal_data={"multimodal_embedding": source})
            params.to_handle("multimodal_data")
            handle = params.multimodal_data["multimodal_embedding"]
            request = GenerationRequest(
                prompt_token_ids=[1],
                sampling_params=SamplingParams(max_tokens=1),
                multimodal_params=params,
            )
            request.set_id(1)
            result = GenerationResult(request)
            transport_params = pickle.loads(pickle.dumps(params))
            self.assertEqual(len(transport_params._shared_tensor_lifetime_refs),
                             0)
            self.assertEqual(len(params._shared_tensor_lifetime_refs), 1)
            del request.multimodal_params
            return result, handle

        result, handle = build_result_and_handle()
        gc.collect()

        restored = SharedTensorContainer.from_dict(handle).get_local_view()
        self.assertTrue(
            torch.allclose(restored, torch.arange(6, dtype=torch.float32)))
        self.assertEqual(len(result._shared_tensor_lifetime_refs), 1)

    def test_generation_result_retains_response_shared_tensor_refs(self):
        """Encoder-result handles survive clearing request-side disagg refs."""

        class FakeResult:

            def __init__(self, mm_embedding_handles):
                self.is_final = True
                self.context_phase_params = None
                self.decoding_iter = 0
                self.cached_tokens = 0
                self.avg_decoded_tokens_per_iter = None
                self.finish_reasons = [tllm.FinishReason.LENGTH]
                self.output_token_ids = [[1]]
                self.cum_log_probs = None
                self.log_probs = None
                self.generation_logits = None
                self.context_logits = None
                self.request_perf_metrics = None
                self.sequence_index = 0
                self.mm_embedding_handles = mm_embedding_handles

        class FakeResponse:

            def __init__(self, result):
                self.result = result

            def has_error(self):
                return False

        producer_params = MultimodalParams(
            multimodal_data={
                "multimodal_embedding": torch.arange(6, dtype=torch.float32)
            })
        producer_params.to_handle("multimodal_data")
        source_handle = producer_params.multimodal_data["multimodal_embedding"]

        request = GenerationRequest(
            prompt_token_ids=[1],
            sampling_params=SamplingParams(max_tokens=1),
        )
        request.set_id(1)
        result = GenerationResult(request)
        result._handle_response(FakeResponse(FakeResult([source_handle])))

        handle = result.disaggregated_params.multimodal_embedding_handles[0]
        self.assertEqual(
            len(result.disaggregated_params._shared_tensor_lifetime_refs), 1)
        result.disaggregated_params._shared_tensor_lifetime_refs.clear()
        del producer_params
        gc.collect()

        restored = SharedTensorContainer.from_dict(handle).get_local_view()
        self.assertTrue(
            torch.allclose(restored, torch.arange(6, dtype=torch.float32)))
        self.assertEqual(len(result._shared_tensor_lifetime_refs), 1)

    def test_generation_result_retains_disagg_shared_tensor_refs(self):
        """E/P/D handoff handles survive transport without shipping refs."""

        def build_result_and_handle():
            source = torch.arange(6, dtype=torch.float32)
            producer_params = MultimodalParams(
                multimodal_data={"multimodal_embedding": source})
            producer_params.to_handle("multimodal_data")
            handle = producer_params.multimodal_data["multimodal_embedding"]

            disaggregated_params = DisaggregatedParams(
                request_type="context_and_generation",
                multimodal_embedding_handles=[handle],
            )
            disaggregated_params._shared_tensor_lifetime_refs.extend(
                producer_params._shared_tensor_lifetime_refs)
            params = MultimodalParams(
                multimodal_data={"multimodal_embedding": [handle]})
            request = GenerationRequest(
                prompt_token_ids=[1],
                sampling_params=SamplingParams(max_tokens=1),
                disaggregated_params=disaggregated_params,
                multimodal_params=params,
            )
            request.set_id(1)
            result = GenerationResult(request,
                                      disaggregated_params=disaggregated_params)
            transport_params = pickle.loads(pickle.dumps(disaggregated_params))
            self.assertEqual(len(transport_params._shared_tensor_lifetime_refs),
                             0)
            self.assertEqual(
                len(disaggregated_params._shared_tensor_lifetime_refs), 1)
            del request.disaggregated_params
            del request.multimodal_params
            return result, handle

        result, handle = build_result_and_handle()
        gc.collect()

        restored = SharedTensorContainer.from_dict(handle).get_local_view()
        self.assertTrue(
            torch.allclose(restored, torch.arange(6, dtype=torch.float32)))
        self.assertEqual(len(result._shared_tensor_lifetime_refs), 1)

    def test_to_tensor_all_handles(self):
        """Test that to_handle followed by to_tensor preserves data integrity."""
        params = MultimodalParams()
        params.multimodal_data = self.sample_multimodal_data.copy()

        params.to_handle("multimodal_data")
        params.to_tensor("multimodal_data")

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


class TestMultimodalParamsDeviceTransfer(unittest.TestCase):
    """Test cases for to_device method in MultimodalParams."""

    def setUp(self):
        """Set up test fixtures."""
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
        self.sample_multimodal_data = {
            "multimodal_embedding": self.mm_embedding,
            "mrope_config": self.mrope_config,
            "image": self.image,
        }

    def test_to_device_basic(self):
        """Test converting a basic data to device."""
        params = MultimodalParams()
        params.multimodal_data = {"multimodal_embedding": self.mm_embedding}

        params.to_device("multimodal_data", device="cuda:0", pin_memory=True)

        result = params.multimodal_data["multimodal_embedding"]
        self.assertEqual(result.device, torch.device("cuda:0"))

    def test_to_device_all_data(self):
        """Test converting all data to device."""
        params = MultimodalParams()
        params.multimodal_data = self.sample_multimodal_data.copy()

        params.to_device("multimodal_data", device="cuda:0", pin_memory=True)

        result = params.multimodal_data["multimodal_embedding"]
        self.assertEqual(result.device, torch.device("cuda:0"))

        result = params.multimodal_data["mrope_config"]["mrope_rotary_cos_sin"]
        self.assertEqual(result.device, torch.device("cuda:0"))

        result = params.multimodal_data["mrope_config"]["mrope_position_deltas"]
        self.assertEqual(result.device, torch.device("cuda:0"))

        result = params.multimodal_data["image"]["pixel_values"]
        self.assertEqual(result.device, torch.device("cuda:0"))

    def test_to_device_with_target_keywords(self):
        """Test converting data to device with keyword."""
        params = MultimodalParams()
        params.multimodal_data = self.sample_multimodal_data.copy()

        params.to_device("multimodal_data",
                         device="cuda:0",
                         pin_memory=True,
                         target_keywords=["image.pixel_values"])

        result = params.multimodal_data["image"]["pixel_values"]
        self.assertEqual(result.device, torch.device("cuda:0"))

        result = params.multimodal_data["mrope_config"]["mrope_rotary_cos_sin"]
        self.assertEqual(result.device, torch.device("cpu"))

        result = params.multimodal_data["mrope_config"]["mrope_position_deltas"]
        self.assertEqual(result.device, torch.device("cpu"))

    def test_to_device_keeps_cumsum_on_cpu(self):
        """`multimodal_embed_mask_cumsum` is CPU-only metadata and must stay on CPU.

        Covers both containment paths:
        - full recursive move (`target_keywords=None`) silently skips the key
        - explicit targeting via `target_keywords` warns once and skips
        """
        cumsum = torch.arange(8, dtype=torch.int64)

        params = MultimodalParams()
        params.multimodal_data = {
            "image": {
                "pixel_values": torch.randn(1, 3, 8, 8),
            },
            "multimodal_embed_mask_cumsum": cumsum,
        }

        params.to_device("multimodal_data", device="cuda:0", pin_memory=True)

        self.assertEqual(
            params.multimodal_data["image"]["pixel_values"].device,
            torch.device("cuda:0"),
        )
        self.assertEqual(
            params.multimodal_data["multimodal_embed_mask_cumsum"].device,
            torch.device("cpu"),
        )

        # Targeted path: explicitly asking to move the CPU-only key is a no-op.
        params.to_device(
            "multimodal_data",
            device="cuda:0",
            target_keywords=["multimodal_embed_mask_cumsum"],
        )
        self.assertEqual(
            params.multimodal_data["multimodal_embed_mask_cumsum"].device,
            torch.device("cpu"),
        )


if __name__ == "__main__":
    unittest.main()
