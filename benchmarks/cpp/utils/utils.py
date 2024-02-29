import json
import math
import random
from typing import List

import numpy as np
from pydantic import BaseModel


class Sample(BaseModel):
    input_ids: List[int]
    output_len: int
    delay: float


class Workload(BaseModel):
    metadata: dict
    samples: List[Sample] = []

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


def dataset_dump(input_ids, output_lens, delays, metadata, output_file):
    samples = []
    for i in range(len(input_ids)):
        samples.append(
            Sample(input_ids=input_ids[i],
                   output_len=output_lens[i],
                   delay=delays[i]))
    workload = Workload(metadata=metadata, samples=samples)
    with open(output_file, 'w') as f:
        json.dump(workload.dict(), f)


def get_req_time_interval(req_rate):
    if req_rate == -1:
        mean_time_bet_reqs = 0
    else:
        mean_time_bet_reqs = 1.0 / req_rate
    return mean_time_bet_reqs


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


def get_norm_dist_tokens(mean, stdev, num_reqs, random_seed):
    # set seed for determinism
    np.random.seed(random_seed)
    numbers_list = np.random.normal(loc=mean, scale=stdev,
                                    size=num_reqs).tolist()
    return [max(1, math.ceil(x)) for x in numbers_list]


def gen_random_tokens(ip_lens, tokenizer, random_seed):
    input_ids = []
    random.seed(random_seed)
    for ip_len in ip_lens:
        start_ids = random.sample(range(0, tokenizer.vocab_size), ip_len)
        # Make sure it does not contain EOS token
        while set(tokenizer.encode(tokenizer.eos_token)).issubset(start_ids):
            start_ids = random.sample(range(0, tokenizer.vocab_size), ip_len)
        input_ids.append(start_ids)

    return input_ids
