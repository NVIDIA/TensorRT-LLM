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

```bash
cd examples/scaffolding/benchmarks
wget https://www.gutenberg.org/ebooks/1184.txt.utf-8
mv 1184.txt.utf-8 pg1184.txt
```

---

## Benchmark Details

### 1. Normal Agent Benchmark (`--enable_normal_agent`)

**Purpose**: Evaluates agentic workflows using the DeepResearch scaffolding with MCP (Model Context Protocol) tools.

**Architecture**:
- Uses `create_open_deep_research_scaffolding_llm` for complex multi-step reasoning
- Connects to MCP server at `http://0.0.0.0:8082/sse` for tool execution
- Supports parallel request processing with configurable concurrency

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--agent_prompt_num` | 100 | Number of prompts to run |
| `--normal_agent_concurrency` | 32 | Max concurrent agent requests |
| `--max_tokens` | 8192 | Maximum output tokens per generation |
| `--max_parallel_requests` | 1024 | Maximum parallel requests to LLM |
| `--times` | 1 | Number of benchmark iterations |

**Prompt Loading**: Prompts are loaded from `contrib/DeepResearch/data/open_deepresearch_bench.json`. If unavailable, a default economic analysis prompt is used.

**Metrics Collected**:
- Per-request execution time
- Total execution time
- Task metrics (with `--enable_statistics`)
- Query traces (with `--enable_query_collector`)

---

### 2. Burst Agent Benchmark (`--enable_burst_agent`)

**Purpose**: Simulates sudden traffic spikes to test system behavior under burst conditions.

**Behavior**:
1. Waits for a configurable delay period (default: 240 seconds)
2. Launches all requests simultaneously after the delay
3. Measures performance under sudden load

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--burst_delay` | 240 | Seconds to wait before burst starts |
| `--burst_prompt_num` | 32 | Number of prompts in the burst |
| `--burst_agent_concurrency` | 32 | Concurrency during burst |

**Use Case**: Testing how the scheduler handles sudden load increases while other benchmarks (normal agent, chat) are running concurrently.

---

### 3. Chat Benchmark (`--enable_chatbot`)

**Purpose**: Simple single-turn chat benchmark for baseline latency measurements.

**Architecture**:
- Uses `NativeChatController` for straightforward request-response
- Single worker (`TRTOpenaiWorker`) connected to OpenAI-compatible endpoint

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--chat_prompt_num` | 20 | Number of chat requests |
| `--chat_concurrency` | 32 | Concurrent chat requests |
| `--max_tokens_chat` | 1024 | Max output tokens per response |

**Sampling Configuration**:
- Temperature: 0.9
- Uses same prompt loading as agent benchmarks

---

### 4. Multiround Chat Benchmark (`--enable_multiround_chatbot`)

**Purpose**: Simulates realistic multi-turn conversations with configurable distributions for turns, tokens, and user delays.

**Architecture**:
```
MultiroundChatController
├── WorkerTag.GENERATION → TRTOpenaiWorker (LLM inference)
├── WorkerTag.MULTIROUND → MultiroundChatWorker (conversation management)
└── DropKVCacheWorkerTag → TRTOpenaiWorker (KV cache management)
```

**Conversation Flow**:
1. Controller initializes a conversation with system message
2. MultiroundChatWorker loads user message for current turn
3. TRTOpenaiWorker generates assistant response
4. Repeat until max rounds or conversation end
5. Optional user delay simulates thinking/typing time between turns

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--chat_prompt_num` | 20 | Number of conversations |
| `--chat_concurrency` | 32 | Concurrent conversations |
| `--chat_multiround_rounds` | 3 | Maximum turns per conversation |
| `--max_tokens_chat` | 1024 | Max output tokens per turn |

---

## Data Sources for Multiround Benchmark

### Source 1: Synthetic Generation (`--multiround_data_source synthetic`)

Generates conversations programmatically using configurable probability distributions.

**Text Content**: Loaded from `--multiround_text_files` (default: `pg1184.txt` - Project Gutenberg text). Used to create realistic message content.

**How It Works**:
1. Samples number of turns from the turns distribution
2. For each conversation, generates a prefix (context) sampled from prefix distribution
3. Each user turn contains: header tag, prefix (first turn only), base prompt, and content from text file
4. Assistant turns are simulated with placeholder content (actual generation happens at runtime)

### Source 2: JSON Config (`--multiround_data_source json_config`)

Load generation parameters from a JSON configuration file for reproducible workloads.

**Example** (`generate_multi_turn.json`):
```json
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

### Source 3: ShareGPT (`--multiround_data_source sharegpt`)

Load real conversations from ShareGPT-format datasets.

**Parameters**:
| Parameter | Description |
|-----------|-------------|
| `--multiround_sharegpt_file` | Path to ShareGPT JSON file |

---

## Distribution Types

The multiround benchmark supports multiple probability distributions for realistic workload simulation:

### Uniform Distribution
Samples uniformly between min and max values.
```bash
--multiround_turns_distribution uniform
--multiround_min_turns 4
--multiround_max_turns 12
```

### Constant Distribution
Always returns the same value.
```bash
# (Use with JSON config - specify "distribution": "constant", "value": X)
```

### Zipf Distribution
Power-law distribution - few items have high values, many have low values. Models "popularity" effects.
```bash
--multiround_turns_distribution zipf
--multiround_turns_alpha 2.0    # Higher alpha = steeper distribution
--multiround_max_turns 12       # Cap maximum value
```

### Poisson Distribution
Discrete distribution for count data. Good for modeling arrival processes.
```bash
--multiround_input_tokens_distribution poisson
--multiround_input_tokens_alpha 100   # Lambda (mean) parameter
--multiround_max_input_tokens 200     # Cap maximum value
```

### Lognormal Distribution
Right-skewed continuous distribution. Models quantities that are products of random effects.
```bash
--multiround_prefix_distribution lognormal
--multiround_prefix_average 1000      # Target average value
--multiround_prefix_max 5000          # Cap maximum value
```

**Lognormal Parameters**:
- `average`: Target average value (internally calculates mean/sigma)
- `median_ratio`: Ratio of median to average (default: 0.85)
- Alternatively, can specify raw `mean` and `sigma` in JSON config

### Exponential Distribution
Models inter-arrival times. Used primarily for user delays.
```bash
--multiround_user_delay_distribution exponential
--multiround_user_delay_lambda 1.0    # Mean delay in seconds
--multiround_user_delay_cap 10.0      # Maximum delay cap
```

---

## User Delay Simulation

Simulates realistic user thinking/typing time between conversation turns.

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--multiround_user_delay_enabled` | True | Enable delay simulation |
| `--multiround_user_delay_disabled` | - | Flag to disable delays |
| `--multiround_user_delay_distribution` | exponential | Distribution type |
| `--multiround_user_delay_lambda` | 1.0 | Mean for exponential/poisson (seconds) |
| `--multiround_user_delay_constant` | 1.0 | Value for constant distribution |
| `--multiround_user_delay_min` | 0.5 | Min for uniform distribution |
| `--multiround_user_delay_max` | 2.0 | Max for uniform distribution |
| `--multiround_user_delay_cap` | 10.0 | Maximum cap for any delay |

**Delay is applied**: Between turns (not before the first user message).

---

## Prefix Tokens (Conversation Context)

The prefix represents initial context/preamble for each conversation (e.g., system context, document content).

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--multiround_prefix_distribution` | lognormal | Distribution type |
| `--multiround_prefix_min` | 100 | Min tokens (uniform) |
| `--multiround_prefix_max` | 5000 | Max tokens |
| `--multiround_prefix_average` | 1000 | Average (lognormal) |
| `--multiround_prefix_alpha` | 2.0 | Alpha (zipf/poisson) |

**Behavior**: Prefix tokens are added only to the first user message in each conversation as a `<conv_prefix>` tagged block.

---

## Trace Collection

The multiround benchmark includes comprehensive trace collection via `ChatTaskTraceCollector`:

**Captured Data**:
- Controller name and timestamp
- Duration (milliseconds)
- Token counts: prompt, completion, reasoning, total
- Message count before/after each yield
- Finish reason
- Performance metrics
- Full message history (optional)

**Output**:
- Summary printed to console after benchmark
- Full traces exported to JSON file (default: `multiround_traces.json`)

---

## Complete CLI Reference

### Server Connection
```bash
--openai_api_key tensorrt_llm     # API key (default: tensorrt_llm)
--base_url http://localhost:8000/v1
--model gpt-oss-20b               # Model name for API calls
--model_dir /path/to/model        # Required: path to tokenizer/model
```

### Benchmark Selection
```bash
--enable_normal_agent             # Run normal agent benchmark
--enable_burst_agent              # Run burst agent benchmark
--enable_chatbot                  # Run single-turn chat benchmark
--enable_multiround_chatbot       # Run multi-turn chat benchmark
```

### Common Parameters
```bash
--times 1                         # Number of benchmark iterations
--max_parallel_requests 1024      # Global max parallel requests
--enable_statistics               # Print task metrics summary
--enable_query_collector          # Dump query traces to JSON
```

### Agent Parameters
```bash
--agent_prompt_num 100
--normal_agent_concurrency 32
--max_tokens 8192
```

### Burst Parameters
```bash
--burst_delay 240
--burst_prompt_num 32
--burst_agent_concurrency 32
```

### Chat Parameters
```bash
--chat_prompt_num 20
--chat_concurrency 32
--max_tokens_chat 1024
```

### Multiround Parameters
```bash
# Data source
--multiround_data_source synthetic|json_config|sharegpt
--multiround_sharegpt_file /path/to/sharegpt.json
--multiround_json_config_file examples/scaffolding/benchmarks/generate_multi_turn.json
--multiround_text_files file1.txt file2.txt
--multiround_print_stats

# Turn distribution
--multiround_turns_distribution uniform|constant|zipf|poisson|lognormal
--multiround_min_turns 4
--multiround_max_turns 12
--multiround_turns_alpha 2.0
--multiround_turns_average 8

# Input token distribution
--multiround_input_tokens_distribution uniform|constant|zipf|poisson|lognormal
--multiround_min_input_tokens 50
--multiround_max_input_tokens 200
--multiround_input_tokens_alpha 2.0
--multiround_input_tokens_average 100

# Output token distribution
--multiround_output_tokens_distribution uniform|constant|zipf|poisson|lognormal
--multiround_min_output_tokens 50
--multiround_max_output_tokens 200
--multiround_output_tokens_alpha 2.0
--multiround_output_tokens_average 100

# Prefix token distribution
--multiround_prefix_distribution lognormal|uniform|constant|zipf|poisson
--multiround_prefix_min 100
--multiround_prefix_max 5000
--multiround_prefix_alpha 2.0
--multiround_prefix_average 1000

# User delay distribution
--multiround_user_delay_enabled
--multiround_user_delay_disabled
--multiround_user_delay_distribution exponential|poisson|constant|uniform
--multiround_user_delay_lambda 1.0
--multiround_user_delay_constant 1.0
--multiround_user_delay_min 0.5
--multiround_user_delay_max 2.0
--multiround_user_delay_cap 10.0

# Conversation control
--chat_multiround_rounds 10       # Max rounds per conversation
```

---

## Usage Examples

### Multi-Turn Conversation Benchmark with JSON Config
```bash
python -m examples.scaffolding.benchmarks.benchmark_agent_chat \
    --model_dir /path/to/model \
    --enable_multiround_chatbot \
    --multiround_data_source json_config \
    --multiround_json_config_file examples/scaffolding/benchmarks/generate_multi_turn.json
```

### Synthetic Multiround with Custom Distributions
```bash
python -m examples.scaffolding.benchmarks.benchmark_agent_chat \
    --model_dir /path/to/model \
    --enable_multiround_chatbot \
    --multiround_data_source synthetic \
    --multiround_text_files examples/scaffolding/benchmarks/pg1184.txt \
    --chat_prompt_num 50 \
    --chat_concurrency 16 \
    --chat_multiround_rounds 15 \
    --multiround_turns_distribution lognormal \
    --multiround_turns_average 8 \
    --multiround_max_turns 20 \
    --multiround_user_delay_distribution exponential \
    --multiround_user_delay_lambda 2.0 \
    --multiround_print_stats
```

### Concurrent Agent and Chat Benchmarks
```bash
python -m examples.scaffolding.benchmarks.benchmark_agent_chat \
    --model_dir /path/to/model \
    --enable_normal_agent \
    --enable_chatbot \
    --agent_prompt_num 50 \
    --chat_prompt_num 100 \
    --normal_agent_concurrency 16 \
    --chat_concurrency 32
```

### Burst Testing with Normal Agent
```bash
python -m examples.scaffolding.benchmarks.benchmark_agent_chat \
    --model_dir /path/to/model \
    --enable_normal_agent \
    --enable_burst_agent \
    --agent_prompt_num 100 \
    --burst_delay 120 \
    --burst_prompt_num 50 \
    --burst_agent_concurrency 64
```

---

## Output Metrics

### Agent/Chat Benchmarks
- Per-request start time and execution time
- Total requests count
- Total execution time
- Average execution time

### Multiround Benchmark
All of the above, plus:
- Average turns per conversation
- P50, P90, P99 latency percentiles
- Trace summary with token counts
- Detailed traces exported to JSON

---

## File Structure

```
benchmarks/
├── benchmark_agent_chat.py      # Main entry point and CLI parser
├── agent_benchmark.py           # Agent and burst agent implementations
├── chat_benchmark.py            # Single-turn chat implementation
├── multiround_chat_benchmark.py # Multi-turn conversation benchmark
├── benchmark_utils.py           # Shared utilities (prompt loading, printing)
├── __init__.py                  # Package exports
├── generate_multi_turn.json     # Example JSON config file
├── pg1184.txt                   # Source text for synthetic generation
└── README.md                    # This documentation
```
