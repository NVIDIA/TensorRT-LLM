import argparse
import json
from pathlib import Path

import jinja2
from parser_utils import kernel_short_name, shortest_common_supersequence

# Parse cmdline
parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=str, required=True)
parser.add_argument("--target", action="append", type=str, required=True)
parser.add_argument("--output", "-o", type=str, required=True)
args = parser.parse_args()
print(args)

with open(args.reference) as f:
    ref_data_all = json.load(f)
if len(ref_data_all) != 1:
    raise ValueError("Ambiguous reference data")
ref_data = ref_data_all[0]
ref_kernel_names = [o["name"] for o in ref_data["timeline"]]
data = []
data.append(
    {
        "series": f"reference: {ref_data['name']}",
        "points": [
            {
                "x": i + 1,
                "name": kernel_short_name(o["name"]),
                "duration": o["duration"] / 1000,
                "end": o["end"] / 1000,
            }
            for i, o in enumerate(ref_data["timeline"])
        ],
    }
)

for target_id, target in enumerate(args.target):
    with open(target) as f:
        tgt_data_all = json.load(f)
    for timeline_id, tgt_data in enumerate(tgt_data_all):
        tgt_kernel_names = [o["name"] for o in tgt_data["timeline"]]
        sup_kernel_names = shortest_common_supersequence(ref_kernel_names, tgt_kernel_names)

        x_sup = []
        j = 0
        for sup_kernel_name in sup_kernel_names:
            if j < len(ref_kernel_names) and sup_kernel_name == ref_kernel_names[j]:
                x_sup.append(j + 1)
                j += 1
            else:
                x_sup.append(None)
        print(f"target {target_id} timeline {timeline_id} {x_sup=}")

        x_tgt = []
        j = 0
        for tgt_kernel_name in tgt_kernel_names:
            while sup_kernel_names[j] != tgt_kernel_name:
                j += 1
            x_tgt.append(x_sup[j])
            j += 1
        if x_tgt[0] is None:
            x_tgt[0] = min(1, min(x for x in x_tgt if x is not None) - 1)
        if x_tgt[-1] is None:
            x_tgt[-1] = max(len(ref_kernel_names), max(x for x in x_tgt if x is not None) + 1)
        top = 0
        while top < len(x_tgt) - 1:
            next_top = top + 1
            while x_tgt[next_top] is None:
                next_top += 1
            for i in range(top + 1, next_top):
                x_tgt[i] = x_tgt[top] + (x_tgt[next_top] - x_tgt[top]) * (i - top) / (
                    next_top - top
                )
            top = next_top
        print(f"target {target_id} timeline {timeline_id} {x_tgt=}")

        data.append(
            {
                "series": f"target{target_id}: {tgt_data['name']}",
                "points": [
                    {
                        "x": x,
                        "name": kernel_short_name(o["name"]),
                        "duration": o["duration"] / 1000,
                        "end": o["end"] / 1000,
                    }
                    for x, o in zip(x_tgt, tgt_data["timeline"])
                ],
            }
        )

loader = jinja2.FileSystemLoader(Path(__file__).parent)
template = jinja2.Environment(loader=loader).get_template("correlation_template.html")
with open(args.output, "w") as f:
    f.write(template.render(rawData=data))
