"""Multi-turn conversation benchmark for scaffolding.

This benchmark simulates multi-turn conversations with realistic user behavior
and conversation context management. It supports various distribution types for
turns, tokens, prefixes, and user delays, as well as JSON config file support
for complex workload specifications.

Adapted from: https://github.com/vllm-project/vllm/tree/main/benchmarks/multi_turn
"""

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from tensorrt_llm.scaffolding import Controller, ScaffoldingLlm, TRTOpenaiWorker, Worker
from tensorrt_llm.scaffolding.benchmark import ScaffoldingBenchRequest, async_scaffolding_benchmark
from tensorrt_llm.scaffolding.load_generation_strategy import ConcurrentStrategy
from tensorrt_llm.scaffolding.task import ChatTask, SystemMessage, Task, TaskStatus, UserMessage
from tensorrt_llm.scaffolding.task_collection import (
    DropKVCacheWorkerTag,
    TaskCollection,
    drop_kv_cache_scope,
    with_task_collection,
)

# =============================================================================
# Type Aliases
# =============================================================================
ConvId = str
MessagesList = List[Dict[str, str]]
ConversationsMap = Dict[ConvId, MessagesList]


# =============================================================================
# Distribution Classes (aligned with reference implementation)
# =============================================================================
class Distribution(ABC):
    """Abstract base class for probability distributions."""

    @abstractmethod
    def sample(self, size: int = 1) -> np.ndarray:
        """Sample values from the distribution."""
        pass


class ConstantDistribution(Distribution):
    """Returns a constant value."""

    def __init__(self, value: Union[int, float]) -> None:
        self.value = value
        self.max_val = value

    def sample(self, size: int = 1) -> np.ndarray:
        return np.full(shape=size, fill_value=self.value)

    def __repr__(self) -> str:
        return f"Constant[{self.value}]"


class UniformDistribution(Distribution):
    """Uniform distribution between min and max values."""

    def __init__(
        self,
        min_val: Union[int, float],
        max_val: Union[int, float],
        is_integer: bool = True,
    ) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.is_integer = is_integer

    def sample(self, size: int = 1) -> np.ndarray:
        if self.is_integer:
            return np.random.randint(int(self.min_val), int(self.max_val + 1), size=size)
        else:
            return np.random.uniform(self.min_val, self.max_val, size=size)

    def __repr__(self) -> str:
        return f"Uniform[{self.min_val}, {self.max_val}]"


class ZipfDistribution(Distribution):
    """Zipf (power-law) distribution."""

    def __init__(self, alpha: float, max_val: Optional[int] = None) -> None:
        self.alpha = alpha
        self.max_val = max_val

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.zipf(self.alpha, size=size)
        if self.max_val:
            samples = np.minimum(samples, self.max_val)
        return samples

    def __repr__(self) -> str:
        return f"Zipf[alpha={self.alpha}, max={self.max_val}]"


class PoissonDistribution(Distribution):
    """Poisson distribution."""

    def __init__(self, alpha: float, max_val: Optional[int] = None) -> None:
        self.alpha = alpha
        self.max_val = max_val

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.poisson(self.alpha, size=size)
        if self.max_val:
            samples = np.minimum(samples, self.max_val)
        return samples

    def __repr__(self) -> str:
        return f"Poisson[alpha={self.alpha}, max={self.max_val}]"


class ExponentialDistribution(Distribution):
    """Exponential distribution for inter-arrival times."""

    def __init__(self, lambda_param: float, max_val: Optional[float] = None) -> None:
        self.lambda_param = lambda_param
        self.max_val = max_val

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.exponential(self.lambda_param, size=size)
        if self.max_val:
            samples = np.minimum(samples, self.max_val)
        return samples

    def __repr__(self) -> str:
        return f"Exponential[lambda={self.lambda_param}, max={self.max_val}]"


class LognormalDistribution(Distribution):
    """Lognormal distribution with flexible parameterization."""

    def __init__(
        self,
        mean: Optional[float] = None,
        sigma: Optional[float] = None,
        average: Optional[int] = None,
        median_ratio: Optional[float] = None,
        max_val: Optional[int] = None,
    ) -> None:
        self.average = average
        self.median_ratio = median_ratio
        self.max_val = max_val

        if average is not None:
            if average < 1:
                raise ValueError("Lognormal average must be positive")

            if mean or sigma:
                raise ValueError("When using lognormal average, you can't provide mean/sigma")

            if self.median_ratio is None:
                # Default value that provides relatively wide range of values
                self.median_ratio = 0.85

            # Calculate mean/sigma of np.random.lognormal based on the average
            mean, sigma = self._generate_lognormal_by_median(
                target_average=self.average, median_ratio=self.median_ratio
            )
        else:
            if mean is None or sigma is None:
                raise ValueError("Must provide both mean and sigma if average is not used")

            if mean <= 0 or sigma < 0:
                raise ValueError("Lognormal mean must be positive and sigma must be non-negative")

        self.mean = mean
        self.sigma = sigma

    @staticmethod
    def _generate_lognormal_by_median(
        target_average: int, median_ratio: float
    ) -> Tuple[float, float]:
        """Compute (mu, sigma) for a lognormal distribution."""
        if median_ratio <= 0 or median_ratio >= 1:
            raise ValueError("median_ratio must be in range (0, 1)")

        target_median = target_average * median_ratio
        sigma = np.sqrt(2 * np.log(target_average / target_median))
        mu = np.log(target_median)

        return mu, sigma

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.lognormal(mean=self.mean, sigma=self.sigma, size=size)

        if self.average is not None:
            # Scale to average
            samples *= self.average / samples.mean()

        if self.max_val:
            samples = np.minimum(samples, self.max_val)

        return np.round(samples).astype(int)

    def __repr__(self) -> str:
        if self.average:
            return f"Lognormal[avg={self.average}, ratio={self.median_ratio}, max={self.max_val}]"
        return f"Lognormal[mean={self.mean}, sigma={self.sigma}, max={self.max_val}]"


# =============================================================================
# Configuration Classes
# =============================================================================
class GenConvArgs(NamedTuple):
    """Arguments for synthetic conversation generation."""

    num_conversations: int
    text_files: List[str]
    input_num_turns: Distribution
    input_prefix_num_tokens: Distribution
    input_num_tokens: Distribution
    output_num_tokens: Distribution
    print_stats: bool = False


@dataclass
class UserDelayConfig:
    """Configuration for user response delay distribution."""

    enabled: bool = True
    distribution: str = "poisson"  # exponential, poisson, constant, uniform
    # Parameters based on distribution type
    lambda_param: float = 1.0  # For exponential/poisson (mean delay in seconds)
    constant_value: float = 1.0  # For constant
    min_val: float = 0.5  # For uniform
    max_val: float = 2.0  # For uniform
    cap: Optional[float] = 10.0  # Maximum delay cap

    def get_distribution(self) -> Distribution:
        """Create the distribution object based on config."""
        if self.distribution == "exponential":
            return ExponentialDistribution(self.lambda_param, self.cap)
        elif self.distribution == "poisson":
            return PoissonDistribution(self.lambda_param, int(self.cap) if self.cap else None)
        elif self.distribution == "constant":
            return ConstantDistribution(self.constant_value)
        elif self.distribution == "uniform":
            return UniformDistribution(self.min_val, self.max_val, is_integer=False)
        else:
            return ExponentialDistribution(self.lambda_param, self.cap)


@dataclass
class MultiroundDataConfig:
    """Configuration for multiround chat data source."""

    # Data source type: "sharegpt", "synthetic", or "json_config"
    data_source: str = "synthetic"

    # For ShareGPT: path to the JSON file
    sharegpt_file: Optional[str] = None

    # For JSON config file (like generate_multi_turn.json)
    json_config_file: Optional[str] = None

    # For synthetic generation (used when no json_config_file)
    num_conversations: int = 24
    text_files: List[str] = field(default_factory=list)

    # Distribution configs for synthetic generation
    # Turn distribution
    num_turns_distribution: str = "zipfian"  # uniform, constant, zipf, poisson, lognormal
    num_turns_min: int = 4
    num_turns_max: int = 12
    num_turns_alpha: float = 1.0  # For zipf/poisson
    num_turns_average: int = 8  # For lognormal

    # Input token distribution
    input_tokens_distribution: str = "uniform"
    input_tokens_min: int = 50
    input_tokens_max: int = 200
    input_tokens_alpha: float = 2.0
    input_tokens_average: int = 100

    # Output token distribution
    output_tokens_distribution: str = "uniform"
    output_tokens_min: int = 50
    output_tokens_max: int = 200
    output_tokens_alpha: float = 2.0
    output_tokens_average: int = 100

    # Prefix token distribution
    prefix_tokens_distribution: str = "lognormal"
    prefix_tokens_min: int = 100
    prefix_tokens_max: int = 5000
    prefix_tokens_alpha: float = 2.0
    prefix_tokens_average: int = 1000

    # Print statistics
    print_stats: bool = False

    # User delay configuration
    user_delay: UserDelayConfig = field(default_factory=UserDelayConfig)


# =============================================================================
# Distribution Factory
# =============================================================================
def create_distribution(
    dist_type: str,
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    alpha: Optional[float] = None,
    average: Optional[int] = None,
    value: Optional[Union[int, float]] = None,
    median_ratio: Optional[float] = None,
    is_integer: bool = True,
) -> Distribution:
    """Factory function to create distribution objects."""
    if dist_type == "constant":
        if value is None:
            raise ValueError("Constant distribution requires 'value'")
        return ConstantDistribution(value)

    elif dist_type == "uniform":
        if min_val is None or max_val is None:
            raise ValueError("Uniform distribution requires 'min_val' and 'max_val'")
        return UniformDistribution(min_val, max_val, is_integer)

    elif dist_type == "zipf":
        if alpha is None:
            raise ValueError("Zipf distribution requires 'alpha'")
        return ZipfDistribution(alpha, max_val=int(max_val) if max_val else None)

    elif dist_type == "poisson":
        if alpha is None:
            raise ValueError("Poisson distribution requires 'alpha'")
        return PoissonDistribution(alpha, max_val=int(max_val) if max_val else None)

    elif dist_type == "lognormal":
        if average is not None:
            return LognormalDistribution(
                average=average,
                median_ratio=median_ratio,
                max_val=int(max_val) if max_val else None,
            )
        elif min_val is not None and max_val is not None:
            # Use min/max as rough bounds, estimate mean/sigma
            mean = np.log((min_val + max_val) / 2)
            sigma = 0.5
            return LognormalDistribution(mean=mean, sigma=sigma, max_val=int(max_val))
        else:
            raise ValueError("Lognormal requires 'average' or 'min_val/max_val'")

    elif dist_type == "exponential":
        if alpha is None:
            alpha = 1.0
        return ExponentialDistribution(alpha, max_val)

    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def get_distribution_from_json(
    conf: dict, section: str, subsection: str, optional: bool = False
) -> Distribution:
    """Parse distribution configuration from JSON."""
    section_conf = conf.get(section, {})

    if optional and subsection not in section_conf:
        return ConstantDistribution(0)

    if subsection not in section_conf:
        raise ValueError(f"Missing subsection {subsection} in section {section}")

    sub_conf = section_conf[subsection]
    distribution = sub_conf.get("distribution")

    if distribution is None:
        raise ValueError(f"Missing 'distribution' in {section}/{subsection}")

    if distribution == "constant":
        return ConstantDistribution(sub_conf["value"])

    elif distribution == "uniform":
        min_value = sub_conf["min"]
        max_value = sub_conf["max"]
        is_integer = isinstance(min_value, int) and isinstance(max_value, int)
        return UniformDistribution(min_value, max_value, is_integer)

    elif distribution == "zipf":
        return ZipfDistribution(sub_conf["alpha"], max_val=sub_conf.get("max"))

    elif distribution == "poisson":
        return PoissonDistribution(sub_conf["alpha"], max_val=sub_conf.get("max"))

    elif distribution == "lognormal":
        max_val = sub_conf.get("max")
        if "average" in sub_conf:
            return LognormalDistribution(
                average=sub_conf["average"],
                median_ratio=sub_conf.get("median_ratio"),
                max_val=max_val,
            )
        return LognormalDistribution(
            mean=sub_conf["mean"], sigma=sub_conf["sigma"], max_val=max_val
        )

    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def parse_json_config_file(config_path: str) -> GenConvArgs:
    """Parse a JSON configuration file for conversation generation."""
    with open(config_path, "r") as f:
        conf = json.load(f)

    assert conf.get("filetype") == "generate_conversations", "Invalid config file type"

    return GenConvArgs(
        num_conversations=conf["num_conversations"],
        text_files=conf["text_files"],
        input_num_turns=get_distribution_from_json(conf, "prompt_input", "num_turns"),
        input_prefix_num_tokens=get_distribution_from_json(
            conf, "prompt_input", "prefix_num_tokens"
        ),
        input_num_tokens=get_distribution_from_json(conf, "prompt_input", "num_tokens"),
        output_num_tokens=get_distribution_from_json(conf, "prompt_output", "num_tokens"),
        print_stats=conf.get("print_stats", False),
    )


def build_gen_conv_args_from_config(config: MultiroundDataConfig) -> GenConvArgs:
    """Build GenConvArgs from MultiroundDataConfig."""
    # Build turn distribution
    input_num_turns = create_distribution(
        config.num_turns_distribution,
        min_val=config.num_turns_min,
        max_val=config.num_turns_max,
        alpha=config.num_turns_alpha,
        average=config.num_turns_average,
    )

    # Build input token distribution
    input_num_tokens = create_distribution(
        config.input_tokens_distribution,
        min_val=config.input_tokens_min,
        max_val=config.input_tokens_max,
        alpha=config.input_tokens_alpha,
        average=config.input_tokens_average,
    )

    # Build output token distribution
    output_num_tokens = create_distribution(
        config.output_tokens_distribution,
        min_val=config.output_tokens_min,
        max_val=config.output_tokens_max,
        alpha=config.output_tokens_alpha,
        average=config.output_tokens_average,
    )

    # Build prefix token distribution
    input_prefix_num_tokens = create_distribution(
        config.prefix_tokens_distribution,
        min_val=config.prefix_tokens_min,
        max_val=config.prefix_tokens_max,
        alpha=config.prefix_tokens_alpha,
        average=config.prefix_tokens_average,
    )

    return GenConvArgs(
        num_conversations=config.num_conversations,
        text_files=config.text_files,
        input_num_turns=input_num_turns,
        input_prefix_num_tokens=input_prefix_num_tokens,
        input_num_tokens=input_num_tokens,
        output_num_tokens=output_num_tokens,
        print_stats=config.print_stats,
    )


# =============================================================================
# Conversation Generation
# =============================================================================
def generate_conversations(args: GenConvArgs, tokenizer: AutoTokenizer) -> ConversationsMap:
    """Generate synthetic conversations using distributions (like reference)."""
    base_prompt_text = "Please rewrite the following text and add more content: "
    base_prompt_token_count = len(tokenizer.encode(base_prompt_text, add_special_tokens=False))

    print(f"Generating {args.num_conversations} conversations...")
    print(f"  Turns distribution: {args.input_num_turns}")
    print(f"  Input tokens distribution: {args.input_num_tokens}")
    print(f"  Output tokens distribution: {args.output_num_tokens}")
    print(f"  Prefix tokens distribution: {args.input_prefix_num_tokens}")

    # Load text from files
    list_of_tokens: List[int] = []
    for filename in args.text_files:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as file:
                data = file.read()
                tokens_in_file = tokenizer.encode(data, add_special_tokens=False)
                list_of_tokens.extend(tokens_in_file)

    # If no text files, generate placeholder tokens
    if not list_of_tokens:
        placeholder_text = (
            "This is placeholder content for synthetic conversation generation. " * 1000
        )
        list_of_tokens = tokenizer.encode(placeholder_text, add_special_tokens=False)

    conversations: ConversationsMap = {}

    # Sample turn counts for all conversations
    turn_count: np.ndarray = args.input_num_turns.sample(args.num_conversations)
    turn_count = np.maximum(turn_count, 2)  # At least 2 turns
    turn_count = turn_count + (turn_count % 2)  # Round up to even number

    # Sample prefix tokens for all conversations
    conv_prefix_tokens: np.ndarray = args.input_prefix_num_tokens.sample(args.num_conversations)

    base_offset = 0
    for conv_id in range(args.num_conversations):
        messages: MessagesList = []
        nturns = int(turn_count[conv_id])

        # Sample token counts for this conversation
        input_token_count: np.ndarray = args.input_num_tokens.sample(nturns)
        input_token_count = np.maximum(input_token_count, base_prompt_token_count)

        output_token_count: np.ndarray = args.output_num_tokens.sample(nturns)
        output_token_count = np.maximum(output_token_count, 1)

        user_turn = True
        for turn_id in range(nturns):
            if user_turn:
                role = "user"
                num_tokens = int(input_token_count[turn_id])

                # Build user prompt
                content = f"<header>conv_{conv_id}, user_message_{turn_id // 2}</header>"
                num_tokens -= len(tokenizer.encode(content, add_special_tokens=False))

                if turn_id == 0:
                    prefix_num_tokens = int(conv_prefix_tokens[conv_id])
                    if prefix_num_tokens > 0:
                        start_offset = base_offset
                        end_offset = start_offset + prefix_num_tokens
                        if len(list_of_tokens) > end_offset:
                            content += (
                                f"<conv_prefix conv_id={conv_id}>"
                                + tokenizer.decode(list_of_tokens[start_offset:end_offset])
                                + "</conv_prefix>"
                            )
                            base_offset += prefix_num_tokens

                content += base_prompt_text
                num_tokens -= base_prompt_token_count

                if num_tokens > 0:
                    start_offset = base_offset + turn_id * int(input_token_count.max())
                    end_offset = start_offset + num_tokens
                    if len(list_of_tokens) > end_offset:
                        content += tokenizer.decode(list_of_tokens[start_offset:end_offset])
            else:
                role = "assistant"
                num_tokens = int(output_token_count[turn_id])
                if len(list_of_tokens) > num_tokens:
                    content = tokenizer.decode(list_of_tokens[:num_tokens])
                else:
                    content = "Response content placeholder."

            messages.append({"role": role, "content": content})
            user_turn = not user_turn

        conversations[f"CONV_ID_{conv_id}"] = messages
        base_offset += nturns

    if args.print_stats:
        print_conversation_stats(conversations, tokenizer)

    return conversations


def print_conversation_stats(conversations: ConversationsMap, tokenizer: AutoTokenizer) -> None:
    """Print statistics about generated conversations."""
    print("\n" + "=" * 60)
    print("Conversation Statistics:")
    print("=" * 60)

    stats = []
    for conv_id, messages in conversations.items():
        user_tokens = []
        assistant_tokens = []

        for m in messages:
            num_tokens = len(tokenizer(m["content"]).input_ids)
            if m["role"] == "user":
                user_tokens.append(num_tokens)
            elif m["role"] == "assistant":
                assistant_tokens.append(num_tokens)

        stats.append(
            {
                "turns": len(messages),
                "avg_user_tokens": np.mean(user_tokens) if user_tokens else 0,
                "avg_assistant_tokens": np.mean(assistant_tokens) if assistant_tokens else 0,
            }
        )

    turns = [s["turns"] for s in stats]
    user_tokens = [s["avg_user_tokens"] for s in stats]
    assistant_tokens = [s["avg_assistant_tokens"] for s in stats]

    print(f"  Conversations: {len(stats)}")
    print(f"  Turns - min: {min(turns)}, max: {max(turns)}, avg: {np.mean(turns):.1f}")
    print(
        f"  User tokens (avg) - min: {min(user_tokens):.0f}, "
        f"max: {max(user_tokens):.0f}, avg: {np.mean(user_tokens):.0f}"
    )
    print(
        f"  Assistant tokens (avg) - min: {min(assistant_tokens):.0f}, "
        f"max: {max(assistant_tokens):.0f}, avg: {np.mean(assistant_tokens):.0f}"
    )
    print("=" * 60 + "\n")


# =============================================================================
# ShareGPT Loading
# =============================================================================
def load_sharegpt_conversations(
    file_path: str, max_conversations: Optional[int] = None
) -> ConversationsMap:
    """Load conversations from a ShareGPT format JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations: ConversationsMap = {}

    if isinstance(data, list):
        for item in data:
            if max_conversations and len(conversations) >= max_conversations:
                break

            conv_id = item.get("id", str(len(conversations)))
            messages = item.get("messages", [])

            if messages:
                conversations[conv_id] = messages

    print(f"Loaded {len(conversations)} conversations from ShareGPT file")
    return conversations


# =============================================================================
# Worker, Controller and TaskCollection
# =============================================================================
class ChatTaskTraceCollector(TaskCollection):
    """Task collection that captures comprehensive trace information for all ChatTask instances.

    Records messages, token counts, timing, and other relevant trace data for debugging and analysis.
    """

    # Global traces: controller_name -> List[trace_dict]
    traces: Dict[str, List[Dict[str, Any]]] = {}

    def __init__(
        self, controller_name: str, enable_print: bool = False, capture_messages: bool = True
    ):
        super().__init__()
        self.controller_name = controller_name
        self.enable_print = enable_print
        self.capture_messages = capture_messages
        self.start_time_map: Dict[int, float] = {}
        self.pre_message_count_map: Dict[int, int] = {}

        if controller_name not in ChatTaskTraceCollector.traces:
            ChatTaskTraceCollector.traces[controller_name] = []

    def _is_task_already_traced(self, task: ChatTask) -> bool:
        return getattr(task, "_trace_in_progress", False)

    def _mark_task_trace_start(self, task: ChatTask):
        task._trace_in_progress = True

    def _mark_task_trace_end(self, task: ChatTask):
        task._trace_in_progress = False

    def before_yield(self, tasks: List[Task]):
        for task in tasks:
            if not isinstance(task, ChatTask):
                continue
            if self._is_task_already_traced(task):
                continue

            self._mark_task_trace_start(task)
            task_id = id(task)
            self.start_time_map[task_id] = time.time()
            self.pre_message_count_map[task_id] = len(task.messages)

            # Enable token counting to capture token usage
            task.enable_token_counting = True

    def after_yield(self, tasks: List[Task]):
        for task in tasks:
            if not isinstance(task, ChatTask):
                continue

            task_id = id(task)
            if task_id not in self.start_time_map:
                continue

            end_time = time.time()
            start_time = self.start_time_map[task_id]
            duration = end_time - start_time
            pre_message_count = self.pre_message_count_map[task_id]

            del self.start_time_map[task_id]
            del self.pre_message_count_map[task_id]
            self._mark_task_trace_end(task)

            # Build trace record
            trace_record = {
                "controller": self.controller_name,
                "timestamp": end_time,
                "duration_ms": duration * 1000,
                "prompt_tokens": getattr(task, "prompt_tokens_num", 0),
                "completion_tokens": getattr(task, "completion_tokens_num", 0),
                "reasoning_tokens": getattr(task, "reasoning_tokens_num", 0),
                "total_tokens": getattr(task, "prompt_tokens_num", 0)
                + getattr(task, "completion_tokens_num", 0),
                "message_count_before": pre_message_count,
                "message_count_after": len(task.messages),
                "finish_reason": getattr(task, "finish_reason", None),
                "unique_id": getattr(task, "unique_id", None),
                "sub_request_markers": getattr(task, "sub_request_markers", []),
                "perf_metrics": getattr(task, "perf_metrics", None),
            }

            # Capture messages if enabled
            if self.capture_messages:
                trace_record["messages"] = [self._serialize_message(msg) for msg in task.messages]
                # Capture only the new messages added during this yield
                if len(task.messages) > pre_message_count:
                    trace_record["new_messages"] = [
                        self._serialize_message(msg) for msg in task.messages[pre_message_count:]
                    ]
                else:
                    trace_record["new_messages"] = []

            ChatTaskTraceCollector.traces[self.controller_name].append(trace_record)

            if self.enable_print:
                self._print_trace(trace_record)

    def _serialize_message(self, message) -> Dict[str, Any]:
        """Serialize a RoleMessage to a dictionary."""
        result = {
            "role": getattr(message, "role", None),
            "content": getattr(message, "content", None),
        }
        # Capture additional fields for AssistantMessage
        if hasattr(message, "reasoning") and message.reasoning is not None:
            result["reasoning"] = message.reasoning
        if hasattr(message, "reasoning_content") and message.reasoning_content is not None:
            result["reasoning_content"] = message.reasoning_content
        if hasattr(message, "tool_calls") and message.tool_calls is not None:
            result["tool_calls"] = [str(tc) for tc in message.tool_calls]
        return result

    def _print_trace(self, trace_record: Dict[str, Any]):
        """Print a single trace record."""
        print(f"\n{'=' * 60}")
        print(f"[ChatTask Trace] Controller: {trace_record['controller']}")
        print(f"Duration: {trace_record['duration_ms']:.2f}ms")
        print(
            f"Tokens: prompt={trace_record['prompt_tokens']}, "
            f"completion={trace_record['completion_tokens']}, "
            f"reasoning={trace_record['reasoning_tokens']}, "
            f"total={trace_record['total_tokens']}"
        )
        print(
            f"Messages: {trace_record['message_count_before']} -> {trace_record['message_count_after']}"
        )
        if "new_messages" in trace_record and trace_record["new_messages"]:
            print("New Messages:")
            for msg in trace_record["new_messages"]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate long content for display
                if content and len(content) > 200:
                    content = content[:200] + "..."
                print(f"  [{role}]: {content}")
        print(f"{'=' * 60}\n")

    @staticmethod
    def get_traces(controller_name: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get traces for a specific controller or all controllers."""
        if controller_name is not None:
            return {controller_name: ChatTaskTraceCollector.traces.get(controller_name, [])}
        return ChatTaskTraceCollector.traces

    @staticmethod
    def get_all_traces() -> List[Dict[str, Any]]:
        """Get all traces across all controllers as a flat list."""
        all_traces = []
        for traces in ChatTaskTraceCollector.traces.values():
            all_traces.extend(traces)
        # Sort by timestamp
        all_traces.sort(key=lambda x: x.get("timestamp", 0))
        return all_traces

    @staticmethod
    def export_traces(file_path: str, controller_name: str = None):
        """Export traces to a JSON file."""
        if controller_name is not None:
            data = ChatTaskTraceCollector.traces.get(controller_name, [])
        else:
            data = ChatTaskTraceCollector.traces
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def print_summary():
        """Print summary of all collected traces."""
        print("\n" + "=" * 80)
        print("CHAT TASK TRACE SUMMARY")
        print("=" * 80)

        for controller_name, traces in ChatTaskTraceCollector.traces.items():
            if not traces:
                continue

            print(f"\n📋 {controller_name} ({len(traces)} traces)")
            print("-" * 70)

            total_duration = sum(t["duration_ms"] for t in traces)
            total_prompt_tokens = sum(t["prompt_tokens"] for t in traces)
            total_completion_tokens = sum(t["completion_tokens"] for t in traces)
            total_reasoning_tokens = sum(t["reasoning_tokens"] for t in traces)

            print(f"  Total Duration: {total_duration:.2f}ms")
            print(f"  Total Prompt Tokens: {total_prompt_tokens}")
            print(f"  Total Completion Tokens: {total_completion_tokens}")
            print(f"  Total Reasoning Tokens: {total_reasoning_tokens}")
            print(f"  Total Tokens: {total_prompt_tokens + total_completion_tokens}")

            if traces:
                avg_duration = total_duration / len(traces)
                print(f"  Average Duration: {avg_duration:.2f}ms")

        print("\n" + "=" * 80 + "\n")

    @staticmethod
    def reset(controller_name: str = None):
        """Reset traces for a specific controller or all controllers."""
        if controller_name is not None:
            if controller_name in ChatTaskTraceCollector.traces:
                ChatTaskTraceCollector.traces[controller_name] = []
        else:
            ChatTaskTraceCollector.traces.clear()

    @staticmethod
    def get_global_info() -> Any:
        return ChatTaskTraceCollector.traces


class MultiroundChatWorker(Worker):
    """Worker that manages multi-turn conversations with real data."""

    def __init__(
        self,
        model_dir: str,
        conversations: ConversationsMap,
        user_delay_config: UserDelayConfig,
        max_output_tokens: int = 512,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            legacy=False,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=False,
            use_fast=True,
        )
        self.conversations = conversations
        self.user_delay_config = user_delay_config
        self.user_delay_distribution = user_delay_config.get_distribution()
        self.max_output_tokens = max_output_tokens
        self.conv_ids = list(conversations.keys())

    def _sample_user_delay(self) -> float:
        """Sample a delay from configured distribution."""
        if not self.user_delay_config.enabled:
            return 0.0
        delay = float(self.user_delay_distribution.sample(1)[0])
        return max(0.0, delay)

    def _get_conversation_for_request(self, request_index: int) -> Tuple[str, MessagesList]:
        """Get the conversation data for a given request index."""
        conv_id = self.conv_ids[request_index % len(self.conv_ids)]
        return conv_id, self.conversations[conv_id]

    async def multiround_handler(self, task: ChatTask) -> TaskStatus:
        """Fill the task with the conversation data for the next turn."""
        # Initialize the conversation state if it is not set
        conv_state = task.customized_result_fields.get("conversation_state")
        if conv_state is None:
            request_index = task.customized_result_fields.get("request_index", 0)
            conv_id, messages = self._get_conversation_for_request(request_index)
            conv_state = {
                "conv_id": conv_id,
                "messages": [m.copy() for m in messages],
                "current_turn_index": 0,
            }
            task.customized_result_fields["conversation_state"] = conv_state

        current_turn = conv_state["current_turn_index"]
        messages = conv_state["messages"]

        if current_turn >= len(messages):
            task.customized_result_fields["is_conversation_end"] = True
            return TaskStatus.SUCCESS

        # Skip the assistant messages
        while current_turn < len(messages):
            msg = messages[current_turn]
            if msg["role"] == "user":
                break
            current_turn += 1

        if current_turn >= len(messages):
            task.customized_result_fields["is_conversation_end"] = True
            return TaskStatus.SUCCESS

        # Apply user delay (simulates user thinking/typing)
        if current_turn > 0:
            delay = self._sample_user_delay()
            if delay > 0:
                await asyncio.sleep(delay)

        task.add_message(UserMessage(messages[current_turn]["content"]))
        task.max_tokens = self.max_output_tokens

        conv_state["current_turn_index"] = current_turn + 1
        task.customized_result_fields["is_conversation_end"] = False

        return TaskStatus.SUCCESS

    task_handlers = {
        ChatTask: multiround_handler,
    }


@drop_kv_cache_scope()
@with_task_collection(
    "trace", ChatTaskTraceCollector, controller_name="MultiroundChatController", enable_print=False
)
class MultiroundChatController(Controller):
    """Controller for multi-turn chat conversations."""

    request_index = 0

    class WorkerTag(Enum):
        GENERATION = "generation"
        MULTIROUND = "multiround"

    def __init__(self, max_rounds: int = 10):
        super().__init__()
        self.max_rounds = max_rounds

    def generate(self, prompt: str):
        task = ChatTask.create_from_prompt(
            None,  # Pseudo user prompt
            [SystemMessage(content="You are a helpful assistant.")],
            None,  # No tools
        )
        task.customized_result_fields["request_index"] = MultiroundChatController.request_index
        MultiroundChatController.request_index += 1

        yield from self.process([task])
        return task.create_scaffolding_output()

    def process(self, tasks) -> TaskStatus:
        task = tasks[0]

        for _ in range(self.max_rounds):
            task.worker_tag = self.WorkerTag.MULTIROUND
            yield [task]

            if task.customized_result_fields and task.customized_result_fields.get(
                "is_conversation_end", False
            ):
                break

            task.worker_tag = self.WorkerTag.GENERATION
            yield [task]

        return TaskStatus.SUCCESS


# =============================================================================
# Main Benchmark Function
# =============================================================================
def load_or_generate_conversations(
    config: MultiroundDataConfig,
    tokenizer: AutoTokenizer,
) -> ConversationsMap:
    """Load conversations from file or generate synthetic ones."""
    # Option 1: JSON config file
    if config.data_source == "json_config" and config.json_config_file:
        if os.path.exists(config.json_config_file):
            print(f"Loading config from: {config.json_config_file}")
            gen_args = parse_json_config_file(config.json_config_file)
            return generate_conversations(gen_args, tokenizer)
        else:
            print(f"Warning: JSON config file not found: {config.json_config_file}")

    # Option 2: ShareGPT file
    if config.data_source == "sharegpt" and config.sharegpt_file:
        if os.path.exists(config.sharegpt_file):
            print(f"Loading ShareGPT from: {config.sharegpt_file}")
            return load_sharegpt_conversations(
                config.sharegpt_file, max_conversations=config.num_conversations
            )
        else:
            print(f"Warning: ShareGPT file not found: {config.sharegpt_file}")

    # Option 3: Synthetic generation with distributions
    print(f"Generating {config.num_conversations} synthetic conversations")
    gen_args = build_gen_conv_args_from_config(config)
    return generate_conversations(gen_args, tokenizer)


async def async_multiround_chat_benchmark(args):
    """Run multiround chat benchmark with real conversation data."""
    # Build user delay config
    user_delay_config = UserDelayConfig(
        enabled=getattr(args, "multiround_user_delay_enabled", True),
        distribution=getattr(args, "multiround_user_delay_distribution", "exponential"),
        lambda_param=getattr(args, "multiround_user_delay_lambda", 1.0),
        constant_value=getattr(args, "multiround_user_delay_constant", 1.0),
        min_val=getattr(args, "multiround_user_delay_min", 0.5),
        max_val=getattr(args, "multiround_user_delay_max", 2.0),
        cap=getattr(args, "multiround_user_delay_cap", 10.0),
    )

    # Build data config
    data_config = MultiroundDataConfig(
        data_source=getattr(args, "multiround_data_source", "synthetic"),
        sharegpt_file=getattr(args, "multiround_sharegpt_file", None),
        json_config_file=getattr(args, "multiround_json_config_file", None),
        num_conversations=getattr(args, "chat_prompt_num", 24),
        text_files=getattr(args, "multiround_text_files", []),
        # Turn distribution
        num_turns_distribution=getattr(args, "multiround_turns_distribution", "uniform"),
        num_turns_min=getattr(args, "multiround_min_turns", 4),
        num_turns_max=getattr(args, "multiround_max_turns", 12),
        num_turns_alpha=getattr(args, "multiround_turns_alpha", 2.0),
        num_turns_average=getattr(args, "multiround_turns_average", 8),
        # Input token distribution
        input_tokens_distribution=getattr(args, "multiround_input_tokens_distribution", "uniform"),
        input_tokens_min=getattr(args, "multiround_min_input_tokens", 50),
        input_tokens_max=getattr(args, "multiround_max_input_tokens", 200),
        input_tokens_alpha=getattr(args, "multiround_input_tokens_alpha", 2.0),
        input_tokens_average=getattr(args, "multiround_input_tokens_average", 100),
        # Output token distribution
        output_tokens_distribution=getattr(
            args, "multiround_output_tokens_distribution", "uniform"
        ),
        output_tokens_min=getattr(args, "multiround_min_output_tokens", 50),
        output_tokens_max=getattr(args, "multiround_max_output_tokens", 200),
        output_tokens_alpha=getattr(args, "multiround_output_tokens_alpha", 2.0),
        output_tokens_average=getattr(args, "multiround_output_tokens_average", 100),
        # Prefix token distribution
        prefix_tokens_distribution=getattr(args, "multiround_prefix_distribution", "lognormal"),
        prefix_tokens_min=getattr(args, "multiround_prefix_min", 100),
        prefix_tokens_max=getattr(args, "multiround_prefix_max", 5000),
        prefix_tokens_alpha=getattr(args, "multiround_prefix_alpha", 2.0),
        prefix_tokens_average=getattr(args, "multiround_prefix_average", 1000),
        print_stats=getattr(args, "multiround_print_stats", False),
        user_delay=user_delay_config,
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        legacy=False,
        trust_remote_code=False,
        use_fast=True,
    )

    # Load or generate conversations
    conversations = load_or_generate_conversations(data_config, tokenizer)

    if not conversations:
        print("Error: No conversations loaded or generated")
        return

    print(f"Loaded {len(conversations)} conversations for benchmark")

    # Print statistics
    total_turns = sum(len(msgs) for msgs in conversations.values())
    avg_turns = total_turns / len(conversations)
    print(f"Average turns per conversation: {avg_turns:.1f}")

    # Initialize workers
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)
    generation_worker = TRTOpenaiWorker(client, args.model)

    multiround_worker = MultiroundChatWorker(
        model_dir=args.model_dir,
        conversations=conversations,
        user_delay_config=data_config.user_delay,
        max_output_tokens=getattr(args, "max_tokens_chat", 512),
    )

    # Create controller and LLM
    max_rounds = getattr(args, "chat_multiround_rounds", 10)
    chat_controller = MultiroundChatController(max_rounds=max_rounds)

    chat_llm = ScaffoldingLlm(
        chat_controller,
        {
            MultiroundChatController.WorkerTag.GENERATION: generation_worker,
            MultiroundChatController.WorkerTag.MULTIROUND: multiround_worker,
            DropKVCacheWorkerTag.DROP_KV_CACHE: generation_worker,
        },
    )

    # Create benchmark requests
    num_requests = min(len(conversations), getattr(args, "chat_prompt_num", 24))
    requests = [ScaffoldingBenchRequest(prompt=f"conversation_{i}") for i in range(num_requests)]

    task_collection_types = {}
    strategy = ConcurrentStrategy(concurrency=getattr(args, "chat_concurrency", 32))

    print("\nStarting multiround chat benchmark:")
    print(f"  Conversations: {num_requests}")
    print(f"  Max rounds: {max_rounds}")
    print(f"  Concurrency: {strategy}")
    print(
        f"  User delay: {data_config.user_delay.distribution} "
        f"(enabled={data_config.user_delay.enabled}, lambda={data_config.user_delay.lambda_param})"
    )

    (
        results,
        requests_start_time,
        requests_execution_time,
        total_time,
    ) = await async_scaffolding_benchmark(
        chat_llm, task_collection_types, requests, strategy=strategy
    )

    # Print results
    print("\n" + "=" * 60)
    print("Multiround Chat Benchmark Results:")
    print("=" * 60)
    print(f"Total conversations: {len(results)}")
    print(f"Total execution time: {total_time:.3f}s")

    if requests_execution_time:
        avg_time = sum(requests_execution_time) / len(requests_execution_time)
        print(f"Average conversation time: {avg_time:.3f}s")
        print(f"Min conversation time: {min(requests_execution_time):.3f}s")
        print(f"Max conversation time: {max(requests_execution_time):.3f}s")

        # Percentiles
        sorted_times = sorted(requests_execution_time)
        p50_idx = int(len(sorted_times) * 0.5)
        p90_idx = int(len(sorted_times) * 0.9)
        p99_idx = int(len(sorted_times) * 0.99)
        print(f"P50: {sorted_times[p50_idx]:.3f}s")
        print(f"P90: {sorted_times[min(p90_idx, len(sorted_times) - 1)]:.3f}s")
        print(f"P99: {sorted_times[min(p99_idx, len(sorted_times) - 1)]:.3f}s")

    print("=" * 60 + "\n")

    # Dump trace data
    ChatTaskTraceCollector.print_summary()
    trace_file = getattr(args, "trace_output_file", "multiround_traces.json")
    ChatTaskTraceCollector.export_traces(trace_file)
    print(f"Traces exported to: {trace_file}")

    chat_llm.shutdown(shutdown_workers=True)

    return results, requests_start_time, requests_execution_time, total_time
