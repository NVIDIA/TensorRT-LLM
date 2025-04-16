import logging
import random
import re
import tempfile
from typing import Optional

import click
from datasets import load_dataset
from PIL import Image
from pydantic import BaseModel, model_validator
from utils.utils import (get_norm_dist_lengths, multimodal_dataset_dump,
                         print_multimodal_dataset, print_text_dataset,
                         text_dataset_dump)


def validate_output_len_dist(ctx, param, value):
    """Validate the --output-len-dist option."""
    if value is None:
        return value
    m = re.match(r"(\d+),(\d+)", value)
    if m:
        return int(m.group(1)), int(m.group(2))
    else:
        raise AssertionError(
            "Incorrect specification for --output-len-dist. Correct format: --output-len-dist <output_len_mean>,<output_len_stdev>"
        )


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

    @model_validator(mode='after')
    def check_prompt(self) -> 'DatasetConfig':
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

    def get_prompt(self, req):
        """Get the prompt sentence from the given request."""
        if self.prompt_key:
            assert self.prompt_key in req, (
                f"Dataset {self.name} does not have key '{self.prompt_key}'. "
                "Please set --prompt-key to one of the available keys: "
                f"{req.keys()}")
            return req[self.prompt_key]
        else:
            return self.prompt

    def get_input(self, req):
        """Get the input sentence from the given request."""
        assert self.input_key in req, (
            f"Dataset {self.name} does not have key '{self.input_key}'. "
            "Please set --input-key to one of the available keys: "
            f"{req.keys()}")
        return req[self.input_key]

    def get_images(self, req):
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

    def get_output(self, req):
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


def load_dataset_from_hf(dataset_config: DatasetConfig):
    """Load dataset from HuggingFace.

    Args:
        dataset_config: A `DatasetConfig` object that defines the dataset to load.
    Returns:
        Dataset iterator.
    Raises:
        ValueError: When dataset loading fails due to incorrect dataset config setting.
    """
    try:
        dataset = iter(
            load_dataset(*dataset_config.query,
                         split=dataset_config.split,
                         streaming=True,
                         trust_remote_code=True))
    except ValueError as e:
        if "Config" in e:
            e += "\n Please add the config name to the dataset config yaml."
        elif "split" in e:
            e += "\n Please specify supported split in the dataset config yaml."
        raise ValueError(e)

    return dataset


@click.command()
@click.option("--dataset-name",
              required=True,
              type=str,
              help=f"Dataset name in HuggingFace.")
@click.option("--dataset-config-name",
              type=str,
              default=None,
              help=f"Dataset config name in HuggingFace (if exists).")
@click.option("--dataset-split",
              type=str,
              required=True,
              help=f"Split of the dataset to use.")
@click.option("--dataset-input-key",
              type=str,
              help=f"The dataset dictionary key for input.")
@click.option("--dataset-image-key",
              type=str,
              default="image",
              help=f"The dataset dictionary key for images.")
@click.option("--dataset-prompt-key",
              type=str,
              default=None,
              help=f"The dataset dictionary key for prompt (if exists).")
@click.option(
    "--dataset-prompt",
    type=str,
    default=None,
    help=f"The prompt string when there is no prompt key for the dataset.")
@click.option("--dataset-output-key",
              type=str,
              default=None,
              help=f"The dataset dictionary key for output (if exists).")
@click.option(
    "--num-requests",
    type=int,
    default=None,
    help=
    "Number of requests to be generated. Will be capped to min(dataset.num_rows, num_requests)."
)
@click.option(
    "--max-input-len",
    type=int,
    default=None,
    help=
    "Maximum input sequence length for a given request. This will be used to filter out the requests with long input sequence length. Default will include all the requests."
)
@click.option(
    "--output-len-dist",
    type=str,
    default=None,
    callback=validate_output_len_dist,
    help=
    "Output length distribution. Default will be the length of the golden output from the dataset. Format: <output_len_mean>,<output_len_stdev>. E.g. 100,10 will randomize the output length with mean=100 and variance=10."
)
@click.pass_obj
def dataset(root_args, **kwargs):
    """Prepare dataset from real dataset."""
    dataset_config = DatasetConfig(**{
        k[8:]: v
        for k, v in kwargs.items() if k.startswith('dataset_')
    })

    input_ids = []
    input_lens = []
    output_lens = []
    task_ids = []
    req_cnt = 0
    modality = None
    multimodal_texts = []
    multimodal_image_paths = []
    for req in load_dataset_from_hf(dataset_config):
        if any(key in req for key in ['image', 'image_1', 'video']):
            # multimodal input
            if 'video' in req and req['video'] is not None:
                assert "Not supported yet"
            assert kwargs['output_len_dist'] is not None, (
                "Output length distribution must be set for multimodal requests."
            )
            modality = 'image'
            text = dataset_config.get_prompt(req)
            images = dataset_config.get_images(req)
            image_paths = []
            for image in images:
                if image is not None:
                    if isinstance(image, str):
                        image_paths.append(image)
                    elif isinstance(image, Image.Image):
                        with tempfile.NamedTemporaryFile(
                                suffix=".jpg", delete=False) as tmp_file:
                            logging.debug(f"Saving image to {tmp_file.name}")
                            image = image.convert("RGB")
                            image.save(tmp_file, "JPEG")
                            filepath = tmp_file.name
                            image_paths.append(filepath)
                    else:
                        raise ValueError(f"Invalid image path: {image}")
            multimodal_texts.append(text)
            multimodal_image_paths.append(image_paths)
        else:
            # text input
            prompt = dataset_config.get_prompt(
                req) + ' ' + dataset_config.get_input(req)
            logging.debug(f"Input sequence: {prompt}")
            line = root_args.tokenizer.encode(prompt)
            if kwargs['max_input_len'] and len(line) > kwargs['max_input_len']:
                continue
            input_ids.append(line)
            input_lens.append(len(line))

            # output if fetch from golden
            if kwargs['output_len_dist'] is None:
                output_lens.append(
                    len(
                        root_args.tokenizer.encode(
                            dataset_config.get_output(req))))

        # lora task id
        task_id = root_args.task_id
        if root_args.rand_task_id is not None:
            min_id, max_id = root_args.rand_task_id
            task_id = random.randint(min_id, max_id)
        task_ids.append(task_id)

        req_cnt += 1
        if kwargs['num_requests'] and req_cnt >= kwargs['num_requests']:
            break

    if kwargs['num_requests'] and (len(input_ids) if modality is None else len(
            multimodal_texts)) < kwargs['num_requests']:
        logging.warning(
            f"Number of requests={len(input_ids) if modality is None else len(multimodal_texts)} is"
            f" smaller than the num-requests user set={kwargs['num_requests']}."
        )

    # output if randomized
    if kwargs['output_len_dist'] is not None:
        osl_mean, osl_stdev = kwargs['output_len_dist']
        output_lens = get_norm_dist_lengths(
            osl_mean, osl_stdev,
            len(input_ids) if modality is None else len(multimodal_texts),
            root_args.random_seed)
    logging.debug(f"Input lengths: {[len(i) for i in input_ids]}")
    logging.debug(f"Output lengths: {output_lens}")
    if modality is not None:
        logging.debug(f"Modality: {modality}")

    if modality is not None:
        if not root_args.std_out:
            multimodal_dataset_dump(
                multimodal_texts, multimodal_image_paths, output_lens, task_ids,
                {
                    "workload_type": "dataset",
                    "tokenizer": root_args.tokenizer.__class__.__name__,
                    "num_requests": len(task_ids),
                    "max_output_len": max(output_lens)
                }, root_args.output)
        else:
            print_multimodal_dataset(
                multimodal_texts,
                multimodal_image_paths,
                output_lens,
            )
    else:
        if not root_args.std_out:
            text_dataset_dump(
                input_lens, input_ids, output_lens, task_ids, {
                    "workload_type": "dataset",
                    "tokenizer": root_args.tokenizer.__class__.__name__,
                    "num_requests": len(input_ids),
                    "max_input_len": max(input_lens),
                    "max_output_len": max(output_lens)
                }, root_args.output)
        else:
            print_text_dataset(
                input_ids,
                output_lens,
            )
