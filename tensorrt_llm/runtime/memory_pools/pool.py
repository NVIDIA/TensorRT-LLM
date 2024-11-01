from dataclasses import dataclass


@dataclass
class Pool(object):
    num_kv_heads: int
    num_layers: int
