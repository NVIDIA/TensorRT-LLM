from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from pydantic import BaseModel, model_validator


class TextSample(BaseModel):
    """
    A dataset sample consisting only of text.
    """

    input_len: int
    input_ids: Tuple[int, ...]
    output_len: int
    task_id: int


class MultimodalSample(BaseModel):
    """
    A dataset sample consisting of text and embedded media represented by their paths.
    """

    task_id: int
    prompt: str
    media_paths: Tuple[str, ...]
    output_len: int


# An alias for the union of the samples types.
# TODO (jdebache): use explicit `type` alias (https://peps.python.org/pep-0613/) when `yapf` starts supporting Python 3.12.
Sample = TextSample | MultimodalSample


class Workload(BaseModel):
    metadata: Dict[str, Any]
    samples: Tuple[Sample, ...]


def _setup_workload_name(metadata: Dict[str, Any]):
    # Keys to ignore
    ignore_keys = ["tokenizer"]
    # Create a string by concatenating keys and values with "__"
    workload_name = "__".join(f"{key}:{value}"
                              for key, value in metadata.items()
                              if key not in ignore_keys)
    metadata.setdefault("workload_name", workload_name)


def create_workload(
    metadata: Dict[str, Any],
    samples: Tuple[Sample, ...],
) -> Workload:
    _setup_workload_name(metadata)
    return Workload(metadata=metadata, samples=list(samples))


class GptManagerBenchmarkExportFormat(BaseModel):
    """
    Describes exporting a workload for consumption by GptManagerBenchmark.
    Note that GptManagerBenchmark is NOT the preferred benchmarking tool for TensorRT-LLM anymore. Whenever possible, use tllm-bench.
    """

    output_file_path: str


class TllmBenchExportFormat(BaseModel):
    """
    Describes exporting a workload for consumption by tllm-bench, the preferred benchmarking entrypoint for TensorRT-LLM.
    """

    output_file_path: str


# An alias for the union of the export formats.
# TODO (jdebache): use explicit `type` alias (https://peps.python.org/pep-0613/) when `yapf` starts supporting Python 3.12.
ExportFormat = GptManagerBenchmarkExportFormat | TllmBenchExportFormat


class UniformLengthDistribution(BaseModel):
    """
    A length distribution that is uniform between a minimum and maximum length.
    """

    min_len: int
    max_len: int


class NormalLengthDistribution(BaseModel):
    """
    A length distribution that is a normal distribution between with the given mean and standard deviation.
    """

    mean: int
    std_dev: int


# An alias for the union of the length distributions.
# TODO (jdebache): use explicit `type` alias (https://peps.python.org/pep-0613/) when `yapf` starts supporting Python 3.12.
LengthDistribution = UniformLengthDistribution | NormalLengthDistribution


class UniformTaskIdDistribution(BaseModel):
    """
    A task ID distribution that is uniform between a minimum and maximum task ID.
    """

    min_id: int
    max_id: int


class ConstantTaskIdDistribution(BaseModel):
    """
    A task ID distribution that is a constant task ID, e.g. all samples have the same task ID.
    """

    task_id: int


# An alias for the union of the task ID distributions.
# TODO (jdebache): use explicit `type` alias (https://peps.python.org/pep-0613/) when `yapf` starts supporting Python 3.12.
TaskIdDistribution = UniformTaskIdDistribution | ConstantTaskIdDistribution


class DatasetConfig(BaseModel):
    """Dataset configurations."""
    """Name of the dataset on HuggingFace."""
    name: str
    """Config name of the dataset if existing."""
    config_name: Optional[str] = None
    """Split of the dataset. Typical values: train, validation, test. Setting to None will include all splits."""
    split: Optional[str]
    """The dataset dictionary used for the input sentence."""
    input_key: Optional[str] = None
    """The dataset dictionary key used for the prompt of the input sentence. Must not be set when prompt is set."""
    image_key: Optional[str] = None
    """The dataset dictionary key used for the images."""
    prompt_key: Optional[str] = None
    """The prompt sentence to be added to the input sentence. Must not be set when prompt_key is set."""
    prompt: Optional[str] = None
    """The dataset dictionary key used to derive the output sequence length. Set to None if the dataset does not have a key for output."""
    output_key: Optional[str]

    @model_validator(mode="after")
    def check_prompt(self) -> "DatasetConfig":
        if self.prompt_key and self.prompt:
            raise AssertionError(
                "--prompt-key and --prompt cannot be set at the same time.")
        if (not self.prompt_key) and (not self.prompt):
            raise AssertionError("Either --prompt-key or --prompt must be set.")
        return self

    @property
    def query(self):
        """Generate the query for HuggingFace `datasets.load_dataset()`"""
        if self.config_name:
            return [self.name, self.config_name]
        else:
            return [self.name]

    def get_prompt(self, req: Dict[str, Any]) -> str:
        """Get the prompt sentence from the given request."""
        if self.prompt_key:
            assert self.prompt_key in req, (
                f"Dataset {self.name} does not have key '{self.prompt_key}'. "
                "Please set --prompt-key to one of the available keys: "
                f"{req.keys()}")
            return req[self.prompt_key]
        else:
            return self.prompt

    def get_input(self, req: Dict[str, Any]) -> str:
        """Get the input sentence from the given request."""
        assert self.input_key in req, (
            f"Dataset {self.name} does not have key '{self.input_key}'. "
            "Please set --input-key to one of the available keys: "
            f"{req.keys()}")
        return req[self.input_key]

    def get_images(self, req: Dict[str, Any]) -> List[str | Image.Image]:
        """Get the images from the given request."""
        image_keys = [self.image_key
                      ] + [f"{self.image_key}_{i}" for i in range(1, 8)]
        assert any(key in req for key in image_keys), (
            f"Dataset {self.name} does not have key '{self.image_key}'. "
            "Please set --dataset-image-key to one of the available keys: "
            f"{req.keys()}")
        images = []
        for key in image_keys:
            if key in req and req[key] is not None:
                images.append(req[key])
        return images

    def get_output(self, req: Dict):
        """Get the output sentence from the given request."""
        if self.output_key is None:
            raise RuntimeError(
                "--output-key is not set. Please either:\n"
                "1. Define output length through --output-len-dist.\n"
                f"2. If the dataset {self.name} has key for golden output and "
                "you wish to set output length to the length of the golden "
                "output, set --output-key.")
        assert self.output_key in req, (
            f"Dataset {self.name} does not have key '{self.output_key}'. "
            "Please set --output-key to one of the available keys: "
            f"{req.keys()}")
        return req[self.output_key]
