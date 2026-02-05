import os

import onnx
import pytest
from _model_test_utils import get_small_model_config

from tensorrt_llm._torch.auto_deploy.export import export_onnx
from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs


@pytest.mark.parametrize(
    "model, output_dir, num_attn_ops",
    [
        (
            "Qwen/Qwen2.5-3B-Instruct",
            "/tmp/test_ad_export_onnx_qwen2.5-3b",
            36,
        ),
    ],
)
def test_ad_export_onnx(model: str, output_dir: str, num_attn_ops: int):
    ad_config = LlmArgs(
        model=get_small_model_config(model)["args"]["model"],
        mode="export_edgellm_onnx",
        max_batch_size=13,
        max_seq_len=4,
    )
    ad_config.transforms["export_to_onnx"]["output_dir"] = output_dir
    export_onnx(ad_config)

    # check if the output directory exists
    assert os.path.exists(output_dir)

    # check if the output json files exist
    assert os.path.exists(os.path.join(output_dir, "added_tokens.json"))
    assert os.path.exists(os.path.join(output_dir, "config.json"))
    assert os.path.exists(os.path.join(output_dir, "processed_chat_template.json"))
    assert os.path.exists(os.path.join(output_dir, "special_tokens_map.json"))
    assert os.path.exists(os.path.join(output_dir, "tokenizer.json"))
    assert os.path.exists(os.path.join(output_dir, "tokenizer_config.json"))
    assert os.path.exists(os.path.join(output_dir, "vocab.json"))

    # check if the output onnx file exists
    assert os.path.exists(os.path.join(output_dir, "model.onnx"))

    # check if the onnx file has 24 AttentionPlugin operators
    onnx_model = onnx.load(os.path.join(output_dir, "model.onnx"))
    attn_ops = [node for node in onnx_model.graph.node if node.op_type == "AttentionPlugin"]
    assert len(attn_ops) == num_attn_ops

    # check input and output names of the model
    actual_inputs_names = {input.name for input in onnx_model.graph.input}
    actual_outputs_names = {output.name for output in onnx_model.graph.output}
    expect_inputs_names = {
        "input_ids",
        "context_lengths",
        "rope_rotary_cos_sin",
        "kvcache_start_index",
        "last_token_ids",
    }
    expect_outputs_names = {"logits"}
    expect_inputs_names.update({f"past_key_values_{i}" for i in range(num_attn_ops)})
    expect_outputs_names.update({f"present_key_values_{i}" for i in range(num_attn_ops)})
    assert actual_inputs_names == expect_inputs_names
    assert actual_outputs_names == expect_outputs_names
