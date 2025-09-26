import argparse

template = '''{{
  "num_experts": {num_experts},
  "k": {k},
  "hidden_size": {hidden_size},
  "inter_size": {inter_size},
  "tp_size": {tp_size},
  "ep_size": {ep_size},
  "world_rank": {world_rank},
  "num_tokens": {num_tokens},
  "act_fn": {act_fn},
  "do_final_scale": 1,
  {dtype_string}
  {routing_string}
  {tactic_string}
  "bias": 0,
  "gemm_to_profile": {gemm_to_profile}
}}'''


def make_dtype_string(dtypes=None):
    if dtypes is None:
        return ""
    if not isinstance(dtypes, list):
        dtypes = [dtypes]
    join_term = '","'  # Include quotes because they should be strings
    return f'"dtypes": ["{join_term.join(dtypes)}"],'


def make_routing_string(name=None, values=None, is_distribution=False):
    if values is None and name is None:
        return ""
    values_field = "expert_distribution" if is_distribution else "selected_experts"
    if values is None:
        return f'"{values_field}": "{name}",'

    values = f'"{values_field}": [{",".join(map(str, values))}],'
    if name is not None:
        values += f' "routing_name": "{name}",'

    return values


def make_tactic_string(tactic_id=None, tactic_id1=None, tactic_id2=None):
    if tactic_id is not None:
        return f'"tactic_id": {tactic_id},'
    if not tactic_id1 and not tactic_id2:
        return f'"tactic_id": "auto",'
    return f'"tactic_id1": {tactic_id1},\n  "tactic_id2": {tactic_id2},'


def populate_benchmark_config(**kwargs):
    return template.format(**kwargs)


# Default Mixtral configurations
num_experts = 8
k = 2
hidden_size = 4096
inter_size = 14336
# tp_size = 8
# ep_size = 1
world_rank = 0
act_fn = 3
dtype_string = make_dtype_string()  # All dtypes
tactic_id1 = '"auto"'
tactic_id2 = '"auto"'
gemms_to_profile = [1, 2, 3]

configs = []
for ep_size in [1, num_experts]:
    for num_tokens in [1, 8, 64, 2048, 16384]:
        tp_size = 8 // ep_size
        if inter_size % (tp_size * 128) != 0:
            continue  # Insufficient alignment
        if num_tokens <= num_experts:
            routing_string = make_routing_string(
                name="balanced",
                is_distribution=False)  # Use the balanced distribution
        else:
            routing_string = make_routing_string(
                name="uniform", is_distribution=True
            )  # Use the default uniform random distribution
        for gemm_to_profile in gemms_to_profile:
            configs.append(
                populate_benchmark_config(num_experts=num_experts,
                                          k=k,
                                          hidden_size=hidden_size,
                                          inter_size=inter_size,
                                          tp_size=tp_size,
                                          ep_size=ep_size,
                                          world_rank=world_rank,
                                          num_tokens=num_tokens,
                                          act_fn=act_fn,
                                          dtype_string=dtype_string,
                                          routing_string=routing_string,
                                          tactic_string=make_tactic_string(
                                              tactic_id1=tactic_id1,
                                              tactic_id2=tactic_id2),
                                          gemm_to_profile=gemm_to_profile))

full_string = "[\n" + ",\n".join(configs) + "\n]"

parser = argparse.ArgumentParser()
parser.add_argument('filename',
                    type=str,
                    help='The name of the file to generate',
                    nargs='?',
                    default="moe-benchmark-file.json")
args = parser.parse_args()

with open(args.filename, "w+") as f:
    f.write(full_string)
