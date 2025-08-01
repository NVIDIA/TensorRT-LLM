### Incorporating `auto_deploy` into your own workflow

AutoDeploy can be seamlessly integrated into your existing workflows using TRT-LLM's LLM high-level API. This section provides a blueprint for configuring and invoking AutoDeploy within your custom applications.

Here is an example of how you can build an LLM object with AutoDeploy integration:

```
from tensorrt_llm._torch.auto_deploy import LLM


# Construct the LLM high-level interface object with autodeploy as backend
llm = LLM(
    model=<HF_MODEL_CARD_OR_DIR>,
    world_size=<DESIRED_WORLD_SIZE>,
    compile_backend="torch-compile",
    model_kwargs={"num_hidden_layers": 2}, # test with smaller model configuration
    attn_backend="flashinfer", # choose between "triton" and "flashinfer"
    attn_page_size=64, # page size for attention (tokens_per_block, should be == max_seq_len for triton)
    skip_loading_weights=False,
    model_factory="AutoModelForCausalLM", # choose appropriate model factory
    mla_backend="MultiHeadLatentAttention", # for models that support MLA
    free_mem_ratio=0.8, # fraction of available memory for cache
    simple_shard_only=False, # tensor parallelism sharding strategy
    max_seq_len=<MAX_SEQ_LEN>,
    max_batch_size=<MAX_BATCH_SIZE>,
)

```

Please consult the AutoDeploy `LLM` API in `tensorrt_llm._torch.auto_deploy.llm` and the
`AutoDeployConfig` class in `tensorrt_llm._torch.auto_deploy.llm_args`
for more detail on how AutoDeploy is configured via the `**kwargs` of the `LLM` API.
