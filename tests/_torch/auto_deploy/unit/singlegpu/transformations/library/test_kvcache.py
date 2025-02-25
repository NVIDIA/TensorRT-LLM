import torch
from _model_test_utils import GQA
from _torch_test_utils import all_close

from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import SequenceInfo
from tensorrt_llm._torch.auto_deploy.custom_ops.triton_attention import TritonWithFlattenedInputs
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transformations.export import torch_export, torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transformations.library import insert_mha_with_kv_cache


@torch.inference_mode()
def test_model_with_kv_cache():
    # some config
    batch_size, seq_len = 16, 64
    num_reset_steps = 2
    num_random_steps = 10
    num_attention_heads = 32
    hidden_size = 1024
    num_key_value_heads = 32
    max_position_embeddings = 128

    # set up sequence+cache objects
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

    # export + check (we clone the state dict to have a bit more freedom in testing below)
    gm = torch_export_to_gm(
        model,
        args=(x,),
        clone=True,
        dynamic_shapes=cm.dynamic_shapes[:1],
    )
    y_gm = gm(x)
    assert all_close(y_model, y_gm)

    # run kv cache transformation
    gm_transformed = insert_mha_with_kv_cache(gm, cm, attention_op=TritonWithFlattenedInputs)
    gm_transformed.to("cuda")
    cm.initialize_caches()

    def _call_and_unnest(x):
        cm.info.nest_sequences(x)
        y = gm_transformed(*cm.args)
        return torch.stack(cm.info.unnest_sequences(y))

    # run regular inference
    cm.info.reset()
    y_no_cache = _call_and_unnest(x)
    assert all_close(y_model, y_no_cache)

    # run inference with kv cache
    cm.info.reset()
    y_with_cache = torch.empty_like(y_model)
    for i in range(x.shape[1]):
        y_with_cache[:, i : i + 1] = _call_and_unnest(x[:, i : i + 1])
        cm.info.update_pos(1)
    assert all_close(y_model, y_with_cache)

    # try running some garbage through the caches and then bring back input_pos to see
    # if that works
    cm.info.update_pos(-num_reset_steps)  # should be x.shape[1] - num_reset
    for i in range(num_random_steps):
        _call_and_unnest(torch.rand_like(x[:, :1]))
        cm.info.update_pos(1)

    # go back and run inference again
    cm.info.reset()
    cm.info.update_pos(x.shape[1] - num_reset_steps)
    for i in range(x.shape[1] - 2, x.shape[1]):
        y_with_cache[:, i : i + 1] = _call_and_unnest(x[:, i : i + 1])
        cm.info.update_pos(1)
    assert all_close(y_model, y_with_cache)

    # check if we can still export the model as expected
    torch_export(gm_transformed, args=cm.args)
    torch_export_to_gm(gm_transformed, args=cm.args)
