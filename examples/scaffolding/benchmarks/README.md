# Scaffolding Benchmarks for Agentic/Chat Workloads

Benchmarks for evaluating TensorRT-LLM scaffolding with multi-turn conversation and agentic workloads.

## Quick Start

```bash
# 1. Start TensorRT-LLM OpenAI-compatible server (default: http://localhost:8000/v1)

# 2. Download source text for synthetic conversations
cd examples/scaffolding/benchmarks
wget https://www.gutenberg.org/ebooks/1184.txt.utf-8 -O pg1184.txt

# 3. Run benchmark
python -m examples.scaffolding.benchmarks --model_dir /path/to/model --enable_chat
```

## Available Benchmarks

| Benchmark | Flag | Description |
|-----------|------|-------------|
| Normal Agent | `--enable_normal_agent` | Agent benchmark using DeepResearch scaffolding with MCP tools |
| Burst Agent | `--enable_burst_agent` | Sudden traffic spike simulation (starts after `--burst_delay` seconds) |
| Chat | `--enable_chat` | Single-turn chat baseline |
| Multiround Chat | `--enable_multiround_chat` | Multi-turn conversations with configurable distributions |

Multiple benchmarks can run concurrently by combining flags.

## Benchmark Details

### Agent Benchmarks

Uses `create_open_deep_research_scaffolding_llm` with MCP server at `http://0.0.0.0:8082/sse`.

Prompts loaded from `examples/scaffolding/contrib/open_deep_research/data/open_deep_research_bench.json` (falls back to default prompt if unavailable).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--agent_prompt_num` | 100 | Number of prompts |
| `--normal_agent_concurrency` | 32 | Concurrent requests |
| `--max_tokens_agent` | 65536 | Max output tokens |
| `--burst_delay` | 240 | Seconds before burst starts |
| `--burst_prompt_num` | 32 | Prompts in burst |
| `--burst_agent_concurrency` | 32 | Burst concurrency |

### Chat Benchmark

Single-turn generation using `NativeChatController` with temperature 0.9.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--chat_prompt_num` | 100 | Number of requests |
| `--chat_concurrency` | 32 | Concurrent requests |
| `--max_tokens_chat` | 8192 | Max output tokens |

### Multiround Chat Benchmark

Requires either `--multiround_synthetic` or `--multiround_sharegpt_file`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--multiround_num_conversations` | 100 | Number of conversations |
| `--multiround_concurrency` | 32 | Concurrent conversations |
| `--multiround_max_rounds` | 20 | Max turns per conversation |
| `--multiround_seed` | None | Random seed for reproducibility |

**Data Sources:**
- `--multiround_synthetic`: Generate conversations from text files (`--multiround_text_files`, default: `pg1184.txt`)
- `--multiround_sharegpt_file PATH`: Load from ShareGPT-format JSON

## Distribution Configuration

Multiround benchmark uses configurable distributions for turns, tokens, and delays. Each distribution type has specific parameters:

| Distribution | Parameters | Use Case |
|--------------|------------|----------|
| `uniform` | `_min`, `_max` | Even spread |
| `constant` | `_value` | Fixed value |
| `lognormal` | `_average`, `_max` | Right-skewed (realistic token counts) |
| `zipf` | `_max` | Power-law (popularity effects) |
| `poisson` | `_value` (lambda), `_max` | Count data |
| `exponential` | `_lambda`, `_cap` | Inter-arrival times |

### Configurable Distributions

**Turn count** (`--multiround_num_turns_*`): `uniform` (default), `constant`, `zipf`, `poisson`
- Default: min=12, max=18, value=10

**Input/Output tokens** (`--multiround_input_tokens_*`, `--multiround_output_tokens_*`): `uniform` (default), `constant`, `lognormal`
- Default: min=200, max=400, average=300, value=300

**Prefix tokens** (`--multiround_prefix_tokens_*`): `lognormal` (default), `uniform`, `constant`
- Default: min=500, max=5000, average=1000, value=1000
- Added as `<conv_prefix>` block to first user message

**User delay** (`--multiround_user_delay_*`): `exponential` (default), `poisson`, `constant`, `uniform`, `lognormal`
- Default: lambda=1.0, constant=1.0, min=0.5, max=2.0, cap=10.0, mean=1.0, sigma=0.5
- Disable with `--multiround_user_delay_disabled`

## Common Options

```bash
# Server connection
--openai_api_key tensorrt_llm     # Default API key
--base_url http://localhost:8000/v1
--model gpt-oss-20b
--model_dir /path/to/model        # Required

# Global settings
--max_parallel_requests 1024
--enable_statistics               # Print task metrics summary
--enable_query_collector          # Dump query traces (agent only)
--export_task_metrics_path FILE   # Export metrics to JSON

# KV cache hints
--kv_cache_hint_enabled           # Enable for all benchmarks
--kv_cache_hint_agent             # Agent only
--kv_cache_hint_chat              # Chat only
--kv_cache_hint_multiround        # Multiround only

# Poisson arrival rate (instead of concurrent burst)
--enable_poisson_arrival
--poisson_warmup_window 60.0      # Window for all arrivals (seconds)
--poisson_arrival_seed 42
```

## Examples

```bash
# Synthetic multiround with custom distributions
python -m examples.scaffolding.benchmarks \
    --model_dir /path/to/model \
    --enable_multiround_chat \
    --multiround_synthetic \
    --multiround_num_turns_distribution uniform \
    --multiround_num_turns_min 8 --multiround_num_turns_max 20 \
    --multiround_user_delay_lambda 2.0 \
    --multiround_print_stats

# ShareGPT conversations
python -m examples.scaffolding.benchmarks \
    --model_dir /path/to/model \
    --enable_multiround_chat \
    --multiround_sharegpt_file /path/to/sharegpt.json

# Concurrent agent + chat
python -m examples.scaffolding.benchmarks \
    --model_dir /path/to/model \
    --enable_normal_agent --enable_chat \
    --agent_prompt_num 50 --chat_prompt_num 100

# Burst test during normal operation
python -m examples.scaffolding.benchmarks \
    --model_dir /path/to/model \
    --enable_normal_agent --enable_burst_agent \
    --burst_delay 120 --burst_prompt_num 50
```

## Output

All benchmarks report:
- Per-request start time and execution time
- Total requests, total time, average time

Multiround additionally reports average turns per conversation.

Use `--enable_statistics` for task metrics summary, `--export_task_metrics_path` for JSON export.
