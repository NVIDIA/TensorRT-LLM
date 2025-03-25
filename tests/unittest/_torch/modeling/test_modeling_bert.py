import unittest
from copy import deepcopy
from dataclasses import dataclass

import torch
from parameterized import parameterized
from transformers import BertConfig
from transformers import \
    BertForSequenceClassification as HFBertForSequenceClassification
from transformers import BertTokenizer

from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_bert import \
    BertForSequenceClassification

BERT_CONFIG = {
    "architectures": ["BertForSequenceClassification"],
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.27.1",
    "type_vocab_size": 2,
    "use_cache": True,
    "vocab_size": 30522,
    "num_labels": 2
}


@dataclass(repr=False)
class Scenario:
    backend: str

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}"


class TestBertForSequenceClassification(unittest.TestCase):

    def setUp(self):
        super().setUp()
        torch.random.manual_seed(1234)

    @parameterized.expand(
        [Scenario(backend="VANILLA"),
         Scenario(backend='TRTLLM')], lambda testcase_func, param_num, param:
        f"{testcase_func.__name__}[{param.args[0]}]")
    def test_bert_allclose_to_hf(self, scenario: Scenario):
        """Compare output to HF"""
        backend = scenario.backend
        metadata_cls = get_attention_backend(backend).Metadata

        # Create configs
        torch.random.manual_seed(0)
        config_dict = deepcopy(BERT_CONFIG)
        hf_config = BertConfig.from_dict(config_dict)
        dtype = hf_config.torch_dtype
        device = torch.device('cuda')

        # Prepare HF model
        hf_model = HFBertForSequenceClassification(hf_config).to(dtype).to(
            device).eval()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Prepare tllm pytorch model
        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend=backend,
        )

        tllm_model = BertForSequenceClassification(model_config).to(dtype).to(
            device)
        tllm_model.load_weights(hf_model.state_dict())

        # Prepare inputs
        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        token_type_ids = torch.zeros_like(input=input_ids)
        position_ids = [torch.arange(0, input_ids.size(-1))]
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()

        # Run HF inference
        with torch.inference_mode():
            # HF model forward
            hf_outputs = hf_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )

        # Fill the metadata for tllm attn
        request_ids = [1]
        prompt_lens = [input_ids.size(-1)]

        attn_metadata = metadata_cls(
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=None,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
        )
        attn_metadata.max_seq_len = input_ids.size(-1)
        attn_metadata.prepare()

        # Flat the inputs for tllm model
        input_ids = input_ids.squeeze(0)
        token_type_ids = token_type_ids.squeeze(0)

        # Run inference
        with torch.inference_mode():
            # TRT-LLM model forward
            tllm_outputs = tllm_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
            )

        # Compare outputs
        torch.testing.assert_close(
            hf_outputs.logits.float(),
            tllm_outputs.float(),
            rtol=1.5e-2,
            atol=1.5e-2,
            msg=f"TRT-LLM and HF logits mismatch for {dtype}")


if __name__ == "__main__":
    unittest.main()
