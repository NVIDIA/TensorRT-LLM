import json
import os
import sys

import safetensors


def dump_stats(ckpt_dir):
    config = None
    with open(os.path.join(ckpt_dir, "config.json")) as c:
        config = json.load(c)
    tp_size = config['mapping']['tp_size']
    pp_size = config['mapping']['pp_size']
    world_size = tp_size * pp_size
    for rank in range(world_size):
        with safetensors.safe_open(os.path.join(ckpt_dir,
                                                f'rank{rank}.safetensors'),
                                   framework='pt',
                                   device='cpu') as f:
            # import pdb; pdb.set_trace()
            for key in f.keys():
                tensor = f.get_tensor(key)
                print(
                    f"rank-{rank}:{key}, shape:{list(tensor.shape)}, max:{tensor.max().item()}, min:{tensor.min().item()}"
                )
    return


dump_stats(sys.argv[1])
