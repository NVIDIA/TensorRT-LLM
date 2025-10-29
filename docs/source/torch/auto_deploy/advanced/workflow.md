### Incorporating `auto_deploy` into your own workflow

AutoDeploy can be seamlessly integrated into existing workflows using TRT-LLM's LLM high-level API. This section provides an example for configuring and invoking AutoDeploy in custom applications.

The following example demonstrates how to build an LLM object with AutoDeploy integration:

```
from tensorrt_llm._torch.auto_deploy import LLM


# Construct the LLM high-level interface object with autodeploy as backend
llm = LLM(
    model=<HF_MODEL_CARD_OR_DIR>,
    world_size=<DESIRED_WORLD_SIZE>,
    model_factory="AutoModelForCausalLM", # choose appropriate model factory
    model_kwargs={"num_hidden_layers": 2}, # test with smaller model configuration
    transforms={
        "insert_cached_attention": {"backend": "flashinfer"},  # or "triton"
        "insert_cached_mla_attention": {"backend": "MultiHeadLatentAttention"},
        "resize_kv_cache": {"free_mem_ratio": 0.8},
        "compile_model": {"backend": "torch-compile"},
        "detect_sharding": {"simple_shard_only": False},

    },
    attn_page_size=64, # page size for attention
    skip_loading_weights=False,
    max_seq_len=<MAX_SEQ_LEN>,
    max_batch_size=<MAX_BATCH_SIZE>,
)

```

For more information about configuring AutoDeploy via the `LLM` API using `**kwargs`, see the AutoDeploy LLM API in `tensorrt_llm._torch.auto_deploy.llm` and the `AutoDeployConfig` class in `tensorrt_llm._torch.auto_deploy.llm_args`.
