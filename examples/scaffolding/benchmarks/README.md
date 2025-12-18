# Scaffolding Benchmarks for Agentic/Chatbot Workloads

This directory contains benchmarks for evaluating TensorRT-LLM scaffolding with realistic multi-turn conversation and agentic workloads.

## Available Benchmarks

| Benchmark | Flag | Description |
|-----------|------|-------------|
| Normal Agent | `--enable_normal_agent` | Standard agent benchmark |
| Burst Agent | `--enable_burst_agent` | Simulates sudden traffic spikes |
| Chatbot | `--enable_chatbot` | Single-turn chat benchmark |
| Multiround Chatbot | `--enable_multiround_chatbot` | Multi-turn conversation benchmark |

## Prerequisites

### 1. Start the TensorRT-LLM Server

Ensure you have a TensorRT-LLM OpenAI-compatible server running:

Default endpoint: http://localhost:8000/v1

### 2. Download Source Text (for Synthetic Conversations)

The benchmark uses Project Gutenberg text to generate realistic conversation content:

```
cd examples/scaffolding/benchmarks
wget https://www.gutenberg.org/ebooks/1184.txt.utf-8
mv 1184.txt.utf-8 pg1184.txt
```

## Usage

### Multi-Turn Conversation Benchmark

```
python -m examples.scaffolding.benchmarks.benchmark_agent_chat \
    --model_dir /path/to/model \
    --enable_multiround_chatbot \
    --multiround_data_source json_config \
    --multiround_json_config_file examples/scaffolding/benchmarks/generate_multi_turn.json
```
    
### Data Sources

The multiround benchmark supports three data sources:

| Source | Flag | Description |
|--------|------|-------------|
| `synthetic` | `--multiround_data_source synthetic` | Auto-generated conversations with configurable distributions |
| `json_config` | `--multiround_data_source json_config` | Load from a JSON configuration file |
| `sharegpt` | `--multiround_data_source sharegpt` | Load from ShareGPT-format dataset |

## Common Options

### Server connection
```
--base_url http://localhost:8000/v1
--model gpt-oss-20b
```

### Benchmark parameters
```
--chat_prompt_num 24              # Number of conversations
--chat_concurrency 32             # Concurrent requests
--chat_multiround_rounds 10       # Max rounds per conversation
--max_tokens_chat 512             # Max output tokens per turn
```

### Distribution Configuration

For synthetic data generation, you can configure distributions for turns, tokens, and delays:

#### Turn count distribution
```
--multiround_turns_distribution uniform   # uniform, constant, zipf, poisson, lognormal
--multiround_min_turns 4
--multiround_max_turns 12
```

#### Input/output token distributions
```
--multiround_input_tokens_distribution uniform
--multiround_min_input_tokens 50
--multiround_max_input_tokens 200

--multiround_output_tokens_distribution uniform
--multiround_min_output_tokens 50
--multiround_max_output_tokens 200
```

#### User delay simulation (typing/thinking time)
```
--multiround_user_delay_distribution exponential  # exponential, poisson, constant, uniform
--multiround_user_delay_lambda 1.0
--multiround_user_delay_disabled                  # To disable delays## JSON Configuration File
```

For reproducible workloads, use a JSON config file (see `generate_multi_turn.json`):

```
{
    "filetype": "generate_conversations",
    "num_conversations": 24,
    "text_files": ["examples/scaffolding/benchmarks/pg1184.txt"],
    "print_stats": false,
    "prompt_input": {
        "num_turns": { "distribution": "uniform", "min": 12, "max": 18 },
        "prefix_num_tokens": { "distribution": "lognormal", "average": 1000, "max": 5000 },
        "num_tokens": { "distribution": "uniform", "min": 120, "max": 160 }
    },
    "prompt_output": {
        "num_tokens": { "distribution": "uniform", "min": 80, "max": 120 }
    }
}
```
