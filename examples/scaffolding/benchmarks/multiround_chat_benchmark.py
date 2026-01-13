"""Multi-turn conversation benchmark for scaffolding.

This benchmark simulates multi-turn conversations with realistic user behavior
and conversation context management. It supports various distribution types for
turns, tokens, prefixes, and user delays.

Data sources:
- Synthetic: Generate synthetic conversations with configurable distributions via CLI args
- ShareGPT: Load real conversations from ShareGPT format JSON files

Adapted from: https://github.com/vllm-project/vllm/tree/main/benchmarks/multi_turn
"""

import asyncio
import itertools
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from tensorrt_llm.scaffolding import Controller, ScaffoldingLlm, TRTOpenaiWorker, Worker
from tensorrt_llm.scaffolding.benchmark import ScaffoldingBenchRequest, async_scaffolding_benchmark
from tensorrt_llm.scaffolding.load_generation_strategy import (
    ConcurrentStrategy,
    PoissonWarmupStrategy,
)
from tensorrt_llm.scaffolding.task import ChatTask, SystemMessage, TaskStatus, UserMessage
from tensorrt_llm.scaffolding.task_collection import (
    DropKVCacheWorkerTag,
    TaskMetricsCollector,
    drop_kv_cache_scope,
    sub_request_node,
    with_task_collection,
)

from .benchmark_utils import print_benchmark_results, shutdown_llm

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


class LognormalDelayDistribution(Distribution):
    """Lognormal distribution for delay times (returns floats, not integers)."""

    def __init__(
        self,
        mean: float,
        sigma: float,
        max_val: Optional[float] = None,
    ) -> None:
        if sigma < 0:
            raise ValueError("Lognormal sigma must be non-negative")
        self.mean = mean
        self.sigma = sigma
        self.max_val = max_val

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.lognormal(mean=self.mean, sigma=self.sigma, size=size)
        if self.max_val:
            samples = np.minimum(samples, self.max_val)
        return samples

    def __repr__(self) -> str:
        return f"LognormalDelay[mean={self.mean}, sigma={self.sigma}, max={self.max_val}]"


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
    distribution: str = "exponential"  # exponential, poisson, constant, uniform, lognormal
    # Parameters based on distribution type
    lambda_param: float = 1.0  # For exponential/poisson (mean delay in seconds)
    constant_value: float = 1.0  # For constant
    min_val: float = 0.5  # For uniform
    max_val: float = 2.0  # For uniform
    cap: Optional[float] = 10.0  # Maximum delay cap
    # For lognormal distribution
    mean: float = 1.0  # Mean (mu) for lognormal
    sigma: float = 0.5  # Sigma for lognormal

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
        elif self.distribution == "lognormal":
            return LognormalDelayDistribution(self.mean, self.sigma, self.cap)
        else:
            return ExponentialDistribution(self.lambda_param, self.cap)


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic conversation generation."""

    enabled: bool = False
    text_files: List[str] = field(
        default_factory=lambda: ["examples/scaffolding/benchmarks/pg1184.txt"]
    )
    print_stats: bool = False
    seed: Optional[int] = None  # Random seed for reproducibility

    # num_turns distribution
    num_turns_distribution: str = "uniform"
    num_turns_min: int = 12
    num_turns_max: int = 18
    num_turns_value: int = 10

    # prefix_num_tokens distribution
    prefix_tokens_distribution: str = "lognormal"
    prefix_tokens_average: int = 1000
    prefix_tokens_max: int = 5000
    prefix_tokens_min: int = 500
    prefix_tokens_value: int = 1000

    # input_num_tokens distribution
    input_tokens_distribution: str = "uniform"
    input_tokens_min: int = 200
    input_tokens_max: int = 400
    input_tokens_average: int = 300
    input_tokens_value: int = 300

    # output_num_tokens distribution
    output_tokens_distribution: str = "uniform"
    output_tokens_min: int = 200
    output_tokens_max: int = 400
    output_tokens_average: int = 300
    output_tokens_value: int = 300

    def get_num_turns_distribution(self) -> Distribution:
        """Create distribution for number of turns."""
        if self.num_turns_distribution == "constant":
            return ConstantDistribution(self.num_turns_value)
        elif self.num_turns_distribution == "uniform":
            return UniformDistribution(self.num_turns_min, self.num_turns_max, is_integer=True)
        elif self.num_turns_distribution == "zipf":
            return ZipfDistribution(1.5, max_val=self.num_turns_max)
        elif self.num_turns_distribution == "poisson":
            return PoissonDistribution(self.num_turns_value, max_val=self.num_turns_max)
        return UniformDistribution(self.num_turns_min, self.num_turns_max, is_integer=True)

    def get_prefix_tokens_distribution(self) -> Distribution:
        """Create distribution for prefix tokens."""
        if self.prefix_tokens_distribution == "constant":
            return ConstantDistribution(self.prefix_tokens_value)
        elif self.prefix_tokens_distribution == "uniform":
            return UniformDistribution(
                self.prefix_tokens_min, self.prefix_tokens_max, is_integer=True
            )
        elif self.prefix_tokens_distribution == "lognormal":
            return LognormalDistribution(
                average=self.prefix_tokens_average, max_val=self.prefix_tokens_max
            )
        return LognormalDistribution(
            average=self.prefix_tokens_average, max_val=self.prefix_tokens_max
        )

    def get_input_tokens_distribution(self) -> Distribution:
        """Create distribution for input tokens."""
        if self.input_tokens_distribution == "constant":
            return ConstantDistribution(self.input_tokens_value)
        elif self.input_tokens_distribution == "uniform":
            return UniformDistribution(
                self.input_tokens_min, self.input_tokens_max, is_integer=True
            )
        elif self.input_tokens_distribution == "lognormal":
            return LognormalDistribution(
                average=self.input_tokens_average, max_val=self.input_tokens_max
            )
        return UniformDistribution(self.input_tokens_min, self.input_tokens_max, is_integer=True)

    def get_output_tokens_distribution(self) -> Distribution:
        """Create distribution for output tokens."""
        if self.output_tokens_distribution == "constant":
            return ConstantDistribution(self.output_tokens_value)
        elif self.output_tokens_distribution == "uniform":
            return UniformDistribution(
                self.output_tokens_min, self.output_tokens_max, is_integer=True
            )
        elif self.output_tokens_distribution == "lognormal":
            return LognormalDistribution(
                average=self.output_tokens_average, max_val=self.output_tokens_max
            )
        return UniformDistribution(self.output_tokens_min, self.output_tokens_max, is_integer=True)

    def to_gen_conv_args(self, num_conversations: int) -> "GenConvArgs":
        """Convert to GenConvArgs for synthetic generation."""
        return GenConvArgs(
            num_conversations=num_conversations,
            text_files=self.text_files,
            input_num_turns=self.get_num_turns_distribution(),
            input_prefix_num_tokens=self.get_prefix_tokens_distribution(),
            input_num_tokens=self.get_input_tokens_distribution(),
            output_num_tokens=self.get_output_tokens_distribution(),
            print_stats=self.print_stats,
        )


@dataclass
class MultiroundDataConfig:
    """Configuration for multiround chat data source.

    Data source is determined by:
    - If synthetic.enabled is True: generate synthetic conversations
    - If sharegpt_file is set: load conversations from ShareGPT format
    """

    # For ShareGPT: path to the JSON file
    sharegpt_file: Optional[str] = None

    # Synthetic data configuration
    synthetic: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)

    # Max conversations to load/generate
    num_conversations: int = 24

    # User delay configuration
    user_delay: UserDelayConfig = field(default_factory=UserDelayConfig)


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

    def _get_conversation_for_worker(self, worker_id: int) -> Tuple[str, MessagesList]:
        """Get the conversation data for a given conversation index."""
        conv_id = self.conv_ids[worker_id % len(self.conv_ids)]
        return conv_id, self.conversations[conv_id]

    async def multiround_handler(self, task: ChatTask) -> TaskStatus:
        """Fill the task with the conversation data for the next turn."""
        # Initialize the conversation state if it is not set
        conv_state = task.customized_result_fields.get("conversation_state")
        if conv_state is None:
            worker_id = task.customized_result_fields.get("worker_id", 0)
            conv_id, messages = self._get_conversation_for_worker(worker_id)
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


@sub_request_node("multiround_chat", is_top_level=True)
@drop_kv_cache_scope()
@with_task_collection(
    "trace",
    TaskMetricsCollector,
    controller_name="MultiroundChatController",
    enable_print=False,
    capture_messages=False,
)
class MultiroundChatController(Controller):
    """Controller for multi-turn chat conversations."""

    _worker_id_counter = itertools.count()

    class WorkerTag(Enum):
        GENERATION = "generation"
        MULTIROUND = "multiround"

    def __init__(self, max_rounds: int = 10):
        super().__init__()
        self.max_rounds = max_rounds
        self.worker_id = next(self._worker_id_counter)

    def clone(self):
        return MultiroundChatController(max_rounds=self.max_rounds)

    def generate(self, prompt: str):
        task = ChatTask.create_from_prompt(
            None,  # Pseudo user prompt
            [SystemMessage(content="You are a helpful assistant.")],
            None,  # No tools
        )
        task.customized_result_fields["worker_id"] = self.worker_id

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
    """Load conversations from file or generate synthetic ones.

    Data source is determined by:
    - If synthetic.enabled is True: generate synthetic conversations
    - If sharegpt_file is set: load conversations from ShareGPT format
    - If neither is set: raise an error
    """
    # Option 1: Synthetic generation
    if config.synthetic.enabled:
        # Set random seed for reproducibility if specified
        if config.synthetic.seed is not None:
            np.random.seed(config.synthetic.seed)
            print(f"Using random seed: {config.synthetic.seed}")
        print("Generating synthetic conversations...")
        gen_args = config.synthetic.to_gen_conv_args(config.num_conversations)
        return generate_conversations(gen_args, tokenizer)

    # Option 2: ShareGPT file
    if config.sharegpt_file:
        if os.path.exists(config.sharegpt_file):
            print(f"Loading ShareGPT from: {config.sharegpt_file}")
            return load_sharegpt_conversations(
                config.sharegpt_file, max_conversations=config.num_conversations
            )
        else:
            raise FileNotFoundError(f"ShareGPT file not found: {config.sharegpt_file}")

    # No data source provided
    raise ValueError(
        "No data source specified. Use --multiround_synthetic for synthetic data "
        "or --multiround_sharegpt_file for ShareGPT data."
    )


async def async_multiround_chat_benchmark(args):
    """Run multiround chat benchmark with real conversation data."""
    # Build user delay config
    user_delay_config = UserDelayConfig(
        enabled=args.multiround_user_delay_enabled,
        distribution=args.multiround_user_delay_distribution,
        lambda_param=args.multiround_user_delay_lambda,
        constant_value=args.multiround_user_delay_constant,
        min_val=args.multiround_user_delay_min,
        max_val=args.multiround_user_delay_max,
        cap=args.multiround_user_delay_cap,
        mean=args.multiround_user_delay_mean,
        sigma=args.multiround_user_delay_sigma,
    )

    # Build synthetic data config from CLI args
    synthetic_config = SyntheticDataConfig(
        enabled=args.multiround_synthetic,
        text_files=args.multiround_text_files,
        print_stats=args.multiround_print_stats,
        seed=getattr(args, "multiround_seed", None),
        # num_turns distribution
        num_turns_distribution=args.multiround_num_turns_distribution,
        num_turns_min=args.multiround_num_turns_min,
        num_turns_max=args.multiround_num_turns_max,
        num_turns_value=args.multiround_num_turns_value,
        # prefix_num_tokens distribution
        prefix_tokens_distribution=args.multiround_prefix_tokens_distribution,
        prefix_tokens_average=args.multiround_prefix_tokens_average,
        prefix_tokens_max=args.multiround_prefix_tokens_max,
        prefix_tokens_min=args.multiround_prefix_tokens_min,
        prefix_tokens_value=args.multiround_prefix_tokens_value,
        # input_num_tokens distribution
        input_tokens_distribution=args.multiround_input_tokens_distribution,
        input_tokens_min=args.multiround_input_tokens_min,
        input_tokens_max=args.multiround_input_tokens_max,
        input_tokens_average=args.multiround_input_tokens_average,
        input_tokens_value=args.multiround_input_tokens_value,
        # output_num_tokens distribution
        output_tokens_distribution=args.multiround_output_tokens_distribution,
        output_tokens_min=args.multiround_output_tokens_min,
        output_tokens_max=args.multiround_output_tokens_max,
        output_tokens_average=args.multiround_output_tokens_average,
        output_tokens_value=args.multiround_output_tokens_value,
    )

    # Build data config
    data_config = MultiroundDataConfig(
        sharegpt_file=args.multiround_sharegpt_file,
        synthetic=synthetic_config,
        num_conversations=args.multiround_num_conversations,
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
    generation_worker = TRTOpenaiWorker(client, args.model, args.kv_cache_hint_multiround)

    multiround_worker = MultiroundChatWorker(
        model_dir=args.model_dir,
        conversations=conversations,
        user_delay_config=data_config.user_delay,
        max_output_tokens=args.max_tokens_chat,
    )

    # Create controller and LLM
    max_rounds = args.multiround_max_rounds
    chat_controller = MultiroundChatController(max_rounds=max_rounds)

    chat_llm = ScaffoldingLlm(
        chat_controller,
        {
            MultiroundChatController.WorkerTag.GENERATION: generation_worker,
            MultiroundChatController.WorkerTag.MULTIROUND: multiround_worker,
            DropKVCacheWorkerTag.DROP_KV_CACHE: generation_worker,
        },
        max_parallel_requests=args.max_parallel_requests,
    )

    # Create benchmark requests
    num_requests = len(conversations)
    requests = [ScaffoldingBenchRequest(prompt=f"conversation_{i}") for i in range(num_requests)]

    task_collection_types = {}
    concurrency = args.multiround_concurrency

    # Select strategy based on Poisson arrival flag
    if getattr(args, "enable_poisson_arrival", False):
        strategy = PoissonWarmupStrategy(
            num_requests=num_requests,
            warmup_window=getattr(args, "poisson_warmup_window", 60.0),
            max_concurrency=concurrency,
            random_seed=getattr(args, "poisson_arrival_seed", 42),
        )
    else:
        strategy = ConcurrentStrategy(concurrency=concurrency)

    print("\nStarting multiround chat benchmark:")
    print(f"  Conversations: {num_requests}")
    print(f"  Max rounds: {max_rounds}")
    print(f"  Concurrency: {concurrency}")
    if getattr(args, "enable_poisson_arrival", False):
        print(
            f"  Arrival: Poisson warmup (window={args.poisson_warmup_window}s, seed={args.poisson_arrival_seed})"
        )
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

    print_benchmark_results(
        "Multiround-Chat", results, requests_start_time, requests_execution_time, total_time
    )

    await shutdown_llm(chat_llm)

    return results, requests_start_time, requests_execution_time, total_time
