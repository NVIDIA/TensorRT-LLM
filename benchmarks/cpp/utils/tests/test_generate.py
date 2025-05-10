import random

from abstractions import (ConstantTaskIdDistribution, DatasetConfig,
                          MultimodalSample, TextSample,
                          UniformLengthDistribution)
from generate import generate_real_dataset, generate_synthetic_text_dataset
from transformers import AutoTokenizer


def test_generate_real_dataset():
    dataset_config = DatasetConfig(
        name="cnn_dailymail",
        split="train",
        prompt="Summarize the following article:",
        config_name="3.0.0",
        input_key="article",
        output_key="highlights",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    workload = generate_real_dataset(
        dataset_config,
        tokenizer,
        100,
        None,
        ConstantTaskIdDistribution(task_id=0),
        100,
        random.Random(42),
    )
    assert len(workload.samples) == 100
    for sample in workload.samples:
        assert isinstance(sample, TextSample)
        assert len(sample.input_ids) <= 100
        assert sample.task_id == 0


def test_generate_real_multimodal_dataset():
    dataset_config = DatasetConfig(
        name="fusing/wikiart_captions",
        split="train",
        prompt="Describe the following image:",
        output_key="text",
        image_key="image",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-vision-128k-instruct")
    workload = generate_real_dataset(
        dataset_config,
        tokenizer,
        100,
        None,
        ConstantTaskIdDistribution(task_id=0),
        100,
        random.Random(42),
    )
    assert len(workload.samples) == 100
    for sample in workload.samples:
        assert isinstance(sample, MultimodalSample)
        assert len(sample.prompt) <= 100
        assert sample.task_id == 0


def test_generate_synthetic_dataset_uniform_constant_lengths():
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    workload = generate_synthetic_text_dataset(
        tokenizer,
        UniformLengthDistribution(min_len=10, max_len=10),
        UniformLengthDistribution(min_len=10, max_len=10),
        ConstantTaskIdDistribution(task_id=0),
        100,
        random.Random(42),
    )
    assert len(workload.samples) == 100
    for sample in workload.samples:
        assert isinstance(sample, TextSample)
        assert len(sample.input_ids) == 10
        assert sample.output_len == 10
