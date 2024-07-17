import hashlib
import os
import signal
import sys
import tempfile
import traceback
import weakref
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Optional

import filelock
import huggingface_hub
import torch
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from tensorrt_llm.bindings import executor as tllme
from tensorrt_llm.logger import Singleton, set_level


def print_traceback_on_error(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            raise e

    return wrapper


@dataclass(slots=True)
class SamplingParams:
    """
    Sampling parameters for text generation.

    Args:
        end_id (int): The end token id.
        pad_id (int): The pad token id.
        max_new_tokens (int): The maximum number of tokens to generate.
        bad_words (List[List[int]]): A list of bad words tokens. Each "word" can be composed of multiple tokens.
        stop_words (List[List[int]]): A list of stop words tokens. Each "word" can be composed of multiple tokens.
        embedding_bias (torch.Tensor): The embedding bias tensor. Expected type is kFP32 and shape is [vocab_size].
        external_draft_tokens_config (ExternalDraftTokensConfig): The speculative decoding configuration.
        prompt_tuning_config (PromptTuningConfig): The prompt tuning configuration.
        lora_config (LoraConfig): The LoRA configuration.
        logits_post_processor_name (str): The logits postprocessor name. Must correspond to one of the logits postprocessor name provided to the ExecutorConfig.

        beam_width (int): The beam width. Default is 1 which disables beam search.
        top_k (int): Controls number of logits to sample from. Default is 0 (all logits).
        top_p (float): Controls the top-P probability to sample from. Default is 0.f
        top_p_min (float): Controls decay in the top-P algorithm. topPMin is lower-bound. Default is 1.e-6.
        top_p_reset_ids (int): Controls decay in the top-P algorithm. Indicates where to reset the decay. Default is 1.
        top_p_decay (float): Controls decay in the top-P algorithm. The decay value. Default is 1.f
        random_seed (int): Controls the random seed used by the random number generator in sampling
        temperature (float): Controls the modulation of logits when sampling new tokens. It can have values > 0.f. Default is 1.0f
        min_length (int): Lower bound on the number of tokens to generate. Values < 1 have no effect. Default is 1.
        beam_search_diversity_rate (float): Controls the diversity in beam search.
        repetition_penalty (float): Used to penalize tokens based on how often they appear in the sequence. It can have any value > 0.f. Values < 1.f encourages repetition, values > 1.f discourages it. Default is 1.f
        presence_penalty (float): Used to penalize tokens already present in the sequence (irrespective of the number of appearances). It can have any values. Values < 0.f encourage repetition, values > 0.f discourage it. Default is 0.f
        frequency_penalty (float): Used to penalize tokens already present in the sequence (dependent on the number of appearances). It can have any values. Values < 0.f encourage repetition, values > 0.f discourage it. Default is 0.f
        length_penalty (float): Controls how to penalize longer sequences in beam search. Default is 0.f
        early_stopping (int): Controls whether the generation process finishes once beamWidth sentences are generated (ends with end_token)
        no_repeat_ngram_size (int): Controls how many repeat ngram size are acceptable. Default is 1 << 30.

        return_log_probs (bool): Controls if Result should contain log probabilities. Default is false.
        return_context_logits (bool): Controls if Result should contain the context logits. Default is false.
        return_generation_logits (bool): Controls if Result should contain the generation logits. Default is false.
        exclude_input_from_output (bool): Controls if output tokens in Result should include the input tokens. Default is true.
        return_encoder_output (bool): Controls if Result should contain encoder output hidden states (for encoder-only and encoder-decoder models). Default is false.
    """
    # [TO DEVELOPER] This class provides an interface to HLAPI users.
    # Internally, it manages and dispatches fields to Python bindings of C++ objects, currently including:
    # (1) all fields of tllme.SamplingConfig;
    # (2) all fields of tllme.OutputConfig;
    # (3) some fields of tllme.Request.
    # If you changed the implementation of C++ objects and corresponding Python bindings, please update:
    # (1) the fields and corresponding docstring of this class, and
    # (2) the expected_fields defined in _get_xxx_config methods.

    end_id: Optional[int] = None
    pad_id: Optional[int] = None
    max_new_tokens: int = 32
    bad_words: Optional[List[List[int]]] = None
    stop_words: Optional[List[List[int]]] = None
    embedding_bias: Optional[torch.Tensor] = None
    external_draft_tokens_config: Optional[
        tllme.ExternalDraftTokensConfig] = None
    prompt_tuning_config: Optional[tllme.PromptTuningConfig] = None
    lora_config: Optional[tllme.LoraConfig] = None
    logits_post_processor_name: Optional[str] = None

    # Keep the below fields in sync with tllme.SamplingConfig
    beam_width: int = 1
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    top_p_min: Optional[float] = None
    top_p_reset_ids: Optional[int] = None
    top_p_decay: Optional[float] = None
    random_seed: Optional[int] = None
    temperature: Optional[float] = None
    min_length: Optional[int] = None
    beam_search_diversity_rate: Optional[float] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    early_stopping: Optional[int] = None
    no_repeat_ngram_size: Optional[int] = None

    # Keep the below fields in sync with tllme.OutputConfig
    return_log_probs: bool = False
    return_context_logits: bool = False
    return_generation_logits: bool = False
    exclude_input_from_output: bool = True
    return_encoder_output: bool = False

    def __post_init__(self):
        if self.pad_id is None:
            self.pad_id = self.end_id

    def _get_sampling_config(self):
        expected_fields = [
            "beam_width", "top_k", "top_p", "top_p_min", "top_p_reset_ids",
            "top_p_decay", "random_seed", "temperature", "min_length",
            "beam_search_diversity_rate", "repetition_penalty",
            "presence_penalty", "frequency_penalty", "length_penalty",
            "early_stopping", "no_repeat_ngram_size"
        ]
        found_fields = [
            f for f in dir(tllme.SamplingConfig) if not f.startswith('__')
        ]
        if set(found_fields) != set(expected_fields):
            raise RuntimeError(
                "Found fields in `tllme.SamplingConfig` different than expected; "
                f"if `tllme.SamplingConfig` is changed, please update {self.__class__.__name__} accordingly. "
                "See [TO DEVELOPER] comments for detailed instructions.")
        return tllme.SamplingConfig(
            **{f: getattr(self, f)
               for f in expected_fields})

    def _get_output_config(self):
        expected_fields = [
            "return_log_probs", "return_context_logits",
            "return_generation_logits", "exclude_input_from_output",
            "return_encoder_output"
        ]
        found_fields = [
            f for f in dir(tllme.OutputConfig) if not f.startswith('__')
        ]
        if set(found_fields) != set(expected_fields):
            raise RuntimeError(
                "Found fields in `tllme.OutputConfig` different than expected; "
                f"if `tllme.OutputConfig` is changed, please update {self.__class__.__name__} accordingly. "
                "See [TO DEVELOPER] comments for detailed instructions.")
        return tllme.OutputConfig(
            **{f: getattr(self, f)
               for f in expected_fields})


def print_colored(message, color: str = None):
    colors = dict(
        grey="\x1b[38;20m",
        yellow="\x1b[33;20m",
        red="\x1b[31;20m",
        bold_red="\x1b[31;1m",
        bold_green="\033[1;32m",
        green="\033[0;32m",
    )
    reset = "\x1b[0m"

    if color:
        sys.stderr.write(colors[color] + message + reset)
    else:
        sys.stderr.write(message)


def file_with_glob_exists(directory, glob) -> bool:
    path = Path(directory)
    for file_path in path.glob(glob):
        if file_path.is_file():
            return True
    return False


def file_with_suffix_exists(directory, suffix) -> bool:
    return file_with_glob_exists(directory, f'*{suffix}')


def get_device_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def get_total_gpu_memory(device: int) -> float:
    return torch.cuda.get_device_properties(device).total_memory


class GpuArch:

    @staticmethod
    def get_arch() -> int:
        return get_gpu_arch()

    @staticmethod
    def is_post_hopper() -> bool:
        return get_gpu_arch() >= 9

    @staticmethod
    def is_post_ampere() -> bool:
        return get_gpu_arch() >= 8

    @staticmethod
    def is_post_volta() -> bool:
        return get_gpu_arch() >= 7


def get_gpu_arch(device: int = 0) -> int:
    return torch.cuda.get_device_properties(device).major


class ContextManager:
    ''' A helper to create a context manager for a resource. '''

    def __init__(self, resource):
        self.resource = resource

    def __enter__(self):
        return self.resource.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.resource.__exit__(exc_type, exc_value, traceback)


def is_directory_empty(directory: Path) -> bool:
    return not any(directory.iterdir())


def init_log_level():
    ''' Set the log level if the environment variable is not set.  '''
    if "TLLM_LOG_LEVEL" not in os.environ:
        set_level("warning")
        os.environ["TLLM_LOG_LEVEL"] = "WARNING"


class ExceptionHandler(metaclass=Singleton):

    def __init__(self):
        self._sys_excepthook: Callable = sys.excepthook
        self._obj_refs_to_shutdown: List[weakref.ReferenceType] = []

    def __call__(self, exc_type, exc_value, traceback):
        self._sys_excepthook(exc_type, exc_value, traceback)

        for obj_ref in self._obj_refs_to_shutdown:
            if (obj := obj_ref()) is not None:
                obj.shutdown()

    def register(self, obj: Any):
        self._obj_refs_to_shutdown.append(weakref.ref(obj))


exception_handler = ExceptionHandler()
sys.excepthook = exception_handler


def sigint_handler(signal, frame):
    sys.stderr.write("\nSIGINT received, quit LLM!\n")
    sys.exit(1)


# Register the signal handler to handle SIGINT
# This helps to deal with user's Ctrl+C
signal.signal(signal.SIGINT, sigint_handler)
# Use the system temporary directory to share the cache
temp_dir = tempfile.gettempdir()


def get_file_lock(model_name: str,
                  cache_dir: Optional[str] = None) -> filelock.FileLock:
    # Hash the model name to avoid invalid characters in the lock file path
    hashed_model_name = hashlib.sha256(model_name.encode()).hexdigest()

    cache_dir = cache_dir or temp_dir
    os.makedirs(cache_dir, exist_ok=True)

    lock_file_path = os.path.join(cache_dir, f"{hashed_model_name}.lock")

    return filelock.FileLock(lock_file_path)


class DisabledTqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def download_hf_model(model: str, revision: Optional[str] = None) -> Path:
    with get_file_lock(model):
        hf_folder = snapshot_download(
            model,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            revision=revision,
            tqdm_class=DisabledTqdm)
    return Path(hf_folder)


def download_hf_pretrained_config(model: str,
                                  revision: Optional[str] = None) -> Path:
    with get_file_lock(model):
        hf_folder = snapshot_download(
            model,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            revision=revision,
            allow_patterns=["config.json"],
            tqdm_class=DisabledTqdm)
    return Path(hf_folder)
