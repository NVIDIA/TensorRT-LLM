import os

import onnx
import pytest

from tensorrt_llm._torch.auto_deploy import LLM, AutoDeployConfig


@pytest.mark.parametrize(
    "model, max_batch_size, max_seq_len, output_dir, num_attn_ops",
    [
        (
            "/home/scratch.trt_llm_data/llm-models/Qwen2.5-0.5B-Instruct",
            13,
            4,
            "/tmp/test_ad_export_onnx_qwen2.5-0.5b",
            24,
        ),
    ],
)
def test_ad_export_onnx(
    model: str, max_batch_size: int, max_seq_len: int, output_dir: str, num_attn_ops: int
):
    ad_config = AutoDeployConfig(
        model=model,
        mode="export_driveos_llm_onnx",
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    ad_config.transforms["export_to_onnx"]["output_dir"] = output_dir
    _ = LLM(**ad_config.to_llm_kwargs())

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
