# Example Run Script

To build and run AutoDeploy example, use the `examples/auto_deploy/build_and_run_ad.py` script:

```bash
cd examples/auto_deploy
python build_and_run_ad.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

You can configure your experiment with various options. Use the `-h/--help` flag to see available options:

```bash
python build_and_run_ad.py --help
```

The following is a non-exhaustive list of common configuration options:

| Configuration Key | Description |
|-------------------|-------------|
| `--model` | The HF model card or path to a HF checkpoint folder |
| `--args.model-factory` | Choose model factory implementation (`"AutoModelForCausalLM"`, ...) |
| `--args.skip-loading-weights` | Only load the architecture, not the weights |
| `--args.model-kwargs` | Extra kwargs that are being passed to the model initializer in the model factory |
| `--args.tokenizer-kwargs` | Extra kwargs that are being passed to the tokenizer initializer in the model factory |
| `--args.world-size` | The number of GPUs used for auto-sharding the model |
| `--args.runtime` | Specifies which type of Engine to use during runtime (`"demollm"` or `"trtllm"`) |
| `--args.compile-backend` | Specifies how to compile the graph at the end |
| `--args.attn-backend` | Specifies kernel implementation for attention |
| `--args.mla-backend` | Specifies implementation for multi-head latent attention |
| `--args.max-seq-len` | Maximum sequence length for inference/cache |
| `--args.max-batch-size` | Maximum dimension for statically allocated KV cache |
| `--args.attn-page-size` | Page size for attention |
| `--prompt.batch-size` | Number of queries to generate |
| `--benchmark.enabled` | Whether to run the built-in benchmark (true/false) |

For default values and additional configuration options, refer to the `ExperimentConfig` class in `examples/auto_deploy/build_and_run_ad.py` file.

The following is a more complete example of using the script:

```bash
cd examples/auto_deploy
python build_and_run_ad.py \
--model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
--args.world-size 2 \
--args.runtime "demollm" \
--args.compile-backend "torch-compile" \
--args.attn-backend "flashinfer" \
--benchmark.enabled True
```
