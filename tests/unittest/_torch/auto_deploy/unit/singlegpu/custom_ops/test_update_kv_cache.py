import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.torch_attention import update_kv_cache


def test_update_kv_cache():
    K_D_HEAD = 4
    V_D_HEAD = 2
    MAX_BATCH_SIZE = 2
    MAX_SEQ_LEN = 4
    seq_length = 4
    n_heads = 3
    batch_size = 1

    # Initialize KV cache
    k_cache = torch.zeros(MAX_BATCH_SIZE, MAX_SEQ_LEN, n_heads, K_D_HEAD)
    v_cache = torch.zeros(MAX_BATCH_SIZE, MAX_SEQ_LEN, n_heads, V_D_HEAD)

    # Generate q,k,v test vectors
    k = torch.ones(batch_size, seq_length, n_heads, K_D_HEAD)
    v = torch.ones(batch_size, seq_length, n_heads, V_D_HEAD)

    print("k_cache: " + str(k_cache))
    print("v_cache: " + str(v_cache))
    print("input_pos: " + str(torch.tensor([0, 0])))
    print("cache_loc: " + str(torch.tensor([0, 1])))
    print("seq_start: " + str(torch.tensor([0, 3])))

    update_kv_cache(
        k.view(batch_size * seq_length, n_heads, K_D_HEAD),
        v.view(batch_size * seq_length, n_heads, V_D_HEAD),
        k_cache,
        v_cache,
        torch.tensor([3, 1]).long(),
        torch.tensor([0, 0]),
        cache_loc=torch.tensor([0, 1]),
        seq_start=torch.tensor([0, 3]).long(),
    )

    print("k_cache: " + str(k_cache))
    print("v_cache: " + str(v_cache))
