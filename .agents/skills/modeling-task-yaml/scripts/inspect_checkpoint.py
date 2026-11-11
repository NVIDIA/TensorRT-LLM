#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Summarize the facts about a HuggingFace checkpoint that shape a task.yaml.

Prints one JSON object: architectures, dtype, whether the model is
multimodal, its MTP layer count, its quantization, its weight size, and
whether it ships a chat template. Every field is read from files in the
checkpoint directory; nothing is imported or downloaded.

Usage:
    python3 inspect_checkpoint.py /path/to/checkpoint
"""

import argparse
import json
import sys
from pathlib import Path

MULTIMODAL_CONFIG_KEYS = ("vision_config", "audio_config", "visual", "vision_tower")
PROCESSOR_FILES = ("preprocessor_config.json", "processor_config.json")
MTP_KEYS = ("num_nextn_predict_layers", "num_mtp_layers", "mtp_num_layers")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")


def _find_key(config: dict, keys: tuple[str, ...]):
    """Return (key, value) of the first match, looking in text_config too."""
    for scope in (config, config.get("text_config") or {}):
        for key in keys:
            if key in scope:
                return key, scope[key]
    return None, None


def _quantization(ckpt: Path, config: dict):
    quant = config.get("quantization_config")
    if quant:
        return {
            "source": "config.json:quantization_config",
            "method": quant.get("quant_method") or quant.get("quant_algo"),
            "detail": quant,
        }
    modelopt = ckpt / "hf_quant_config.json"
    if modelopt.is_file():
        detail = json.loads(modelopt.read_text())
        algo = (detail.get("quantization") or {}).get("quant_algo")
        return {"source": "hf_quant_config.json", "method": algo, "detail": detail}
    return None


def _chat_template(ckpt: Path):
    if (ckpt / "chat_template.jinja").is_file():
        return "chat_template.jinja"
    tokenizer_config = ckpt / "tokenizer_config.json"
    if tokenizer_config.is_file():
        if json.loads(tokenizer_config.read_text()).get("chat_template"):
            return "tokenizer_config.json:chat_template"
    return None


def inspect(ckpt: Path) -> dict:
    config_path = ckpt / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"no config.json in {ckpt}")
    config = json.loads(config_path.read_text())

    multimodal_evidence = [k for k in MULTIMODAL_CONFIG_KEYS if k in config]
    multimodal_evidence += [f for f in PROCESSOR_FILES if (ckpt / f).is_file()]

    weight_bytes = sum(
        p.stat().st_size for p in ckpt.iterdir() if p.is_file() and p.suffix in WEIGHT_SUFFIXES
    )

    mtp_key, mtp_layers = _find_key(config, MTP_KEYS)

    return {
        "checkpoint_path": str(ckpt),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "dtype": _find_key(config, ("torch_dtype", "dtype"))[1],
        "multimodal": bool(multimodal_evidence),
        "multimodal_evidence": multimodal_evidence,
        "mtp_layers": mtp_layers,
        "mtp_config_key": mtp_key,
        "quantization": _quantization(ckpt, config),
        "weight_size_gb": round(weight_bytes / 1e9, 1),
        "chat_template": _chat_template(ckpt),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="HF checkpoint directory")
    args = parser.parse_args()
    try:
        summary = inspect(args.checkpoint)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
