import torch
from _model_test_utils import GQA
from _torch_test_utils import all_close

from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import (
    PositionalEmbeddingConfig,
    SequenceInfo,
)
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transformations.export import torch_export, torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transformations.library.fused_mha import identify_and_fuse_mha


@torch.inference_mode()
def test_identify_and_fuse_mha():
    # some config
    batch_size, seq_len = 16, 64
    num_attention_heads = 32
    hidden_size = 1024
    num_key_value_heads = 32
    max_position_embeddings = 128

    # set up sequence+cache objects for dynamic shapes
    ci = SequenceInfo(
        max_seq_len=max_position_embeddings,
        max_batch_size=batch_size,
    )
    cm = CachedSequenceInterface(sequence_info=ci, device="cuda")

    # model and input
    model = GQA(num_attention_heads, hidden_size, num_key_value_heads).to(
        device="cuda", dtype=torch.float16
    )
    x = torch.rand(batch_size, seq_len, hidden_size).to(device="cuda", dtype=torch.float16)

    # get the model's regular output
    y_model = model(x)  # b, s, d

    # export to graph module using the same dynamic_shapes approach as original test
    gm = torch_export_to_gm(
        model,
        args=(x,),
        clone=True,
        dynamic_shapes=cm.dynamic_shapes[:1],
    )
    y_gm = gm(x)
    assert all_close(y_model, y_gm)

    # Create positional embedding config for the fused MHA
    pos_embd_config = PositionalEmbeddingConfig()

    # run fuse MHA transformation
    gm_fused = identify_and_fuse_mha(gm, pos_embd_config)
    gm_fused.to("cuda")

    # Make sure the output matches the original
    y_fused = gm_fused(x)
    assert all_close(y_model, y_fused)

    # check if we can still export the model as expected
    torch_export(gm_fused, args=(x,))
    torch_export_to_gm(gm_fused, args=(x,))
