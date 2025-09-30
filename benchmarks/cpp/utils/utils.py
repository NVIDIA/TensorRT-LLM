import json
import math
import os
import random
from typing import List, Union

import numpy as np
from pydantic import BaseModel


class TextSample(BaseModel):
    input_len: int
    input_ids: List[int]
    output_len: int
    task_id: int


class MultimodalSample(BaseModel):
    task_id: int
    prompt: str
    media_paths: List[str]
    output_len: int


class Workload(BaseModel):
    metadata: dict
    samples: List[Union[TextSample, MultimodalSample]] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.setup_workload_name()

    def setup_workload_name(self):
        # Keys to ignore
        ignore_keys = ['tokenizer']
        # Create a string by concatenating keys and values with "__"
        workload_name = '__'.join(f'{key}:{value}'
                                  for key, value in self.metadata.items()
                                  if key not in ignore_keys)
        self.metadata.setdefault('workload_name', workload_name)


def text_dataset_dump(input_lens, input_ids, output_lens, task_ids, metadata,
                      output_file):
    samples = []
    for i in range(len(input_ids)):
        samples.append(
            TextSample(input_len=input_lens[i],
                       input_ids=input_ids[i],
                       output_len=output_lens[i],
                       task_id=task_ids[i]))
    workload = Workload(metadata=metadata, samples=samples)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(workload.model_dump(), f)


def multimodal_dataset_dump(multimodal_texts, multimodal_image_paths,
                            output_lens, task_ids, metadata, output_file):
    samples = []
    for i in range(len(multimodal_texts)):
        samples.append(
            MultimodalSample(task_id=task_ids[i],
                             prompt=multimodal_texts[i],
                             media_paths=multimodal_image_paths[i],
                             output_len=output_lens[i]))
    workload = Workload(metadata=metadata, samples=samples)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(workload.model_dump(), f)


def print_text_dataset(input_ids, output_lens, task_ids=None, lora_config=None):
    for i, input_tokens in enumerate(input_ids):
        d = {
            "task_id": i,
            "input_ids": input_tokens,
            "output_tokens": output_lens[i]
        }

        # Add LoRA request if task_ids indicate LoRA usage
        if task_ids is not None and lora_config is not None:
            task_id = task_ids[i]
            if task_id != -1:  # -1 means no LoRA
                d["lora_request"] = {
                    "lora_name":
                    f"lora_{task_id}",
                    "lora_int_id":
                    task_id,
                    "lora_path":
                    os.path.join(lora_config.get("lora_dir", "loras"),
                                 str(task_id))
                }

        print(json.dumps(d, separators=(',', ':'), ensure_ascii=False))


def print_multimodal_dataset(multimodal_texts, multimodal_image_paths,
                             output_lens):
    for i, (text, image_paths) in enumerate(
            zip(multimodal_texts, multimodal_image_paths)):
        d = {
            "task_id": i,
            "prompt": text,
            "media_paths": image_paths,
            "output_tokens": output_lens[i]
        }
        print(json.dumps(d, separators=(',', ':'), ensure_ascii=False))


def get_list_of_delays(delay_dist, mean_time_bet_reqs, num_reqs, random_seed):
    if delay_dist == "constant":
        delays = [mean_time_bet_reqs] * num_reqs
    elif delay_dist == "exponential_dist":
        delays = get_exponential_dist_delays(mean_time_bet_reqs, num_reqs,
                                             random_seed)

    return delays


def get_exponential_dist_delays(mean_time_bet_reqs, num_reqs, random_seed):
    # set seed for determinism
    np.random.seed(random_seed)
    return np.random.exponential(mean_time_bet_reqs, num_reqs).tolist()


def get_norm_dist_lengths(mean, stdev, num_reqs, random_seed):
    # set seed for determinism
    np.random.seed(random_seed)
    numbers_list = np.random.normal(loc=mean, scale=stdev,
                                    size=num_reqs).tolist()
    return [max(1, math.ceil(x)) for x in numbers_list]


def get_unif_dist_lengths(min_len, max_len, num_reqs, random_seed):
    # set seed for determinism
    rng = np.random.default_rng(random_seed)
    numbers = rng.integers(low=min_len, high=max_len + 1, size=num_reqs)
    return numbers.tolist()


def gen_random_tokens(ip_lens, tokenizer, random_seed):

    def get_sample_from_population(population_range, sample_size):
        # random.sample can not sample a value more than once. hence the check
        if sample_size < len(population_range):
            sample = random.sample(population_range, sample_size)
        else:
            sample = random.choices(population_range, k=sample_size)

        return sample

    input_ids = []
    random.seed(random_seed)
    for ip_len in ip_lens:
        start_ids = get_sample_from_population(range(0, tokenizer.vocab_size),
                                               ip_len)
        # Make sure it does not contain EOS token
        eos_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)
        while set(eos_id).issubset(start_ids):
            tmp_id = (eos_id[0] + 1) % tokenizer.vocab_size
            start_ids = [
                tmp_id if element == eos_id[0] else element
                for element in start_ids
            ]
        input_ids.append(start_ids)

    return input_ids
