import logging
import random
import tempfile
from typing import List, Optional, Tuple

from datasets import IterableDataset, load_dataset
from PIL import Image
from transformers import AutoTokenizer
from utils.abstractions import (ConstantTaskIdDistribution, DatasetConfig,
                                LengthDistribution, MultimodalSample,
                                NormalLengthDistribution, TaskIdDistribution,
                                TextSample, UniformLengthDistribution,
                                UniformTaskIdDistribution, Workload,
                                create_workload)


def _generate_lengths_uniform(
    uniform_length_distribution: UniformLengthDistribution,
    num_reqs: int,
    random_source: random.Random,
) -> Tuple[int, ...]:
    return tuple(
        random_source.randint(uniform_length_distribution.min_len,
                              uniform_length_distribution.max_len)
        for _ in range(num_reqs))


def _generate_lengths_normal(
    normal_length_distribution: NormalLengthDistribution,
    num_reqs: int,
    random_source: random.Random,
) -> Tuple[int, ...]:
    return tuple(
        int(
            random_source.gauss(normal_length_distribution.mean,
                                normal_length_distribution.std_dev))
        for _ in range(num_reqs))


def _generate_lengths(
    length_distribution: LengthDistribution,
    num_reqs: int,
    random_source: random.Random,
) -> Tuple[int, ...]:
    match length_distribution:
        case UniformLengthDistribution() as uniform_length_distribution:
            return _generate_lengths_uniform(
                uniform_length_distribution,
                num_reqs,
                random_source,
            )
        case NormalLengthDistribution() as normal_length_distribution:
            return _generate_lengths_normal(
                normal_length_distribution,
                num_reqs,
                random_source,
            )
        case _:
            raise ValueError(
                f"Unsupported length distribution: {length_distribution}")


def _get_samples_from_population(population: Tuple[int, ...], sample_size: int,
                                 random: random.Random) -> Tuple[int, ...]:
    return random.choices(population, k=sample_size)


def _generate_random_tokens(
    input_lengths: Tuple[int, ...],
    tokenizer: AutoTokenizer,
    random_source: random.Random,
) -> Tuple[List[int], ...]:
    population = tuple(range(0, tokenizer.vocab_size))

    def _sample(input_length: int) -> List[int]:
        sampled_ids = _get_samples_from_population(population, input_length,
                                                   random_source)
        # Make sure it does not contain EOS token
        eos_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)
        while set(eos_id).issubset(sampled_ids):
            tmp_id = (eos_id[0] + 1) % tokenizer.vocab_size
            sampled_ids = [
                tmp_id if element == eos_id[0] else element
                for element in sampled_ids
            ]
        return sampled_ids

    return [_sample(input_length) for input_length in input_lengths]


def _generate_task_ids_uniform(
    uniform_task_id_distribution: UniformTaskIdDistribution,
    num_reqs: int,
    random_source: random.Random,
) -> Tuple[int, ...]:
    return tuple(
        random_source.randint(uniform_task_id_distribution.min_id,
                              uniform_task_id_distribution.max_id)
        for _ in range(num_reqs))


def _generate_task_ids_constant(
    constant_task_id_distribution: ConstantTaskIdDistribution,
    num_reqs: int,
) -> Tuple[int, ...]:
    return tuple(constant_task_id_distribution.task_id for _ in range(num_reqs))


def _generate_task_ids(
    task_id_distribution: TaskIdDistribution,
    num_reqs: int,
    random_source: random.Random,
) -> Tuple[int, ...]:
    match task_id_distribution:
        case UniformTaskIdDistribution() as uniform_task_id_distribution:
            return _generate_task_ids_uniform(
                uniform_task_id_distribution,
                num_reqs,
                random_source,
            )
        case ConstantTaskIdDistribution() as constant_task_id_distribution:
            return _generate_task_ids_constant(
                constant_task_id_distribution,
                num_reqs,
            )
        case _:
            raise ValueError(
                f"Unsupported task id distribution: {task_id_distribution}")


def generate_synthetic_text_dataset(
    tokenizer: AutoTokenizer,
    input_lengths_distribution: LengthDistribution,
    output_lengths_distribution: LengthDistribution,
    task_id_distribution: TaskIdDistribution,
    num_reqs: int,
    random_source: random.Random,
) -> Workload:
    input_lens = _generate_lengths(
        input_lengths_distribution,
        num_reqs,
        random_source,
    )
    output_lens = _generate_lengths(
        output_lengths_distribution,
        num_reqs,
        random_source,
    )
    task_ids = _generate_task_ids(
        task_id_distribution,
        num_reqs,
        random_source,
    )

    max_input_len = max(input_lens)
    max_output_len = max(output_lens)

    input_ids = _generate_random_tokens(input_lens, tokenizer, random_source)

    metadata = {
        "num_requests": num_reqs,
        "tokenize_vocabsize": tokenizer.vocab_size,
        "max_input_len": max_input_len,
        "max_output_len": max_output_len,
    }

    samples = [
        TextSample(
            input_len=input_len,
            input_ids=input_ids,
            output_len=output_len,
            task_id=task_id,
        ) for input_len, input_ids, output_len, task_id in zip(
            input_lens,
            input_ids,
            output_lens,
            task_ids,
        )
    ]
    return create_workload(metadata, samples)


def load_dataset_from_hf(dataset_config: DatasetConfig) -> IterableDataset:
    """Load dataset from HuggingFace.

    Args:
        dataset_config: A `DatasetConfig` object that defines the dataset to load.
    Returns:
        Dataset iterator.
    Raises:
        ValueError: When dataset loading fails due to incorrect dataset config setting.
    """
    try:
        return load_dataset(
            *dataset_config.query,
            split=dataset_config.split,
            streaming=True,
            trust_remote_code=True,
        )
    except ValueError as e:
        error_string = str(e)
        if "Config" in error_string:
            error_string += "\n Please add the config name to the dataset config yaml."
        elif "split" in error_string:
            error_string += (
                "\n Please specify supported split in the dataset config yaml.")
        raise ValueError(error_string)


def generate_real_dataset(
    dataset_config: DatasetConfig,
    tokenizer: AutoTokenizer,
    max_input_length: int,
    output_lengths_distribution: Optional[LengthDistribution],
    task_id_distribution: TaskIdDistribution,
    num_reqs: Optional[int],
    random_source: random.Random,
) -> Workload:

    def _is_multimodal_request(req: dict) -> bool:
        return any(key in req for key in ["image", "image_1", "video"])

    def _get_single_output_length(req: dict) -> int:
        if output_lengths_distribution is None:
            output = dataset_config.get_output(req)
            match output:
                case str():
                    return len(tokenizer.encode(output))
                case list():
                    return len(tokenizer.encode(output[0]))
                case _:
                    raise ValueError(f"Invalid output: {output}")
        else:
            return _generate_lengths(output_lengths_distribution, 1,
                                     random_source)[0]

    def _get_single_task_id() -> int:
        return _generate_task_ids(task_id_distribution, 1, random_source)[0]

    def _create_multimodal_sample(req: dict) -> MultimodalSample:
        if "video" in req and req["video"] is not None:
            raise NotImplementedError("Video is not supported yet.")
        text = dataset_config.get_prompt(req)
        images = dataset_config.get_images(req)
        image_paths = []
        for image in images:
            match image:
                case str():
                    image_paths.append(image)
                case Image.Image():
                    with tempfile.NamedTemporaryFile(suffix=".jpg",
                                                     delete=False) as tmp_file:
                        logging.debug(f"Saving image to {tmp_file.name}")
                        image = image.convert("RGB")
                        image.save(tmp_file, "JPEG")
                        filepath = tmp_file.name
                        image_paths.append(filepath)
                case _:
                    raise ValueError(f"Invalid image path: {image}")
        return MultimodalSample(
            prompt=text,
            media_paths=image_paths,
            output_len=_get_single_output_length(req),
            task_id=_get_single_task_id(),
        )

    def _create_text_sample(req: dict) -> Optional[TextSample]:
        prompt = dataset_config.get_prompt(
            req) + " " + dataset_config.get_input(req)
        logging.debug(f"Input sequence: {prompt}")
        line = tokenizer.encode(prompt)
        if max_input_length and len(line) > max_input_length:
            logging.debug(
                f"Input sequence length={len(line)} is larger than the max input length={max_input_length}. Skipping this request."
            )
            return None

        return TextSample(
            input_len=len(line),
            input_ids=line,
            output_len=_get_single_output_length(req),
            task_id=_get_single_task_id(),
        )

    def _create_sample(req: dict) -> TextSample | MultimodalSample | None:
        if _is_multimodal_request(req):
            return _create_multimodal_sample(req)
        else:
            return _create_text_sample(req)

    hf_dataset = load_dataset_from_hf(dataset_config)
    samples = []
    max_input_len = 0
    max_output_len = 0

    for req in hf_dataset:
        sample = _create_sample(req)
        if sample is not None:
            samples.append(sample)
            if isinstance(sample, TextSample):
                max_input_len = max(max_input_len, sample.input_len)
                max_output_len = max(max_output_len, sample.output_len)
        if num_reqs and len(samples) >= num_reqs:
            break

    if len(samples) < num_reqs:
        logging.warning(
            f"Number of requests={len(samples)} is smaller than the num-requests user set={num_reqs}."
        )

    metadata = {
        "num_requests": len(samples),
        "tokenize_vocabsize": tokenizer.vocab_size,
        "max_input_len": max_input_len,
        "max_output_len": max_output_len,
    }

    return create_workload(metadata, samples)
