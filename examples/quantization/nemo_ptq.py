# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os

import ammo.torch.quantization as atq
import torch
import torch.multiprocessing as mp
from ammo.torch.export import export_model_config
from apex.transformer import parallel_state
from datasets import load_dataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import \
    MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from omegaconf import OmegaConf
from pytorch_lightning.trainer.trainer import Trainer

mp.set_start_method("spawn", force=True)


def get_calib_dataloader(batch_size,
                         max_sequence_length=512,
                         data="cnn_dailymail"):
    if data == "pileval":
        dataset = load_dataset(
            "json",
            data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst",
            split="train")
        text_column = "text"
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        text_column = "article"
    for i in range(len(dataset) // batch_size):
        batch = dataset[i * batch_size:(i + 1) * batch_size][text_column]
        for j in range(len(batch)):
            batch[j] = batch[j][:max_sequence_length]
        yield batch


@hydra_runner(config_path="config", config_name="megatron_gpt_inference")
def main(cfg) -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for the inference.")

    # trainer required for restoring model parallel models
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    # 43B model requires loading from directory instead of nemo file
    # docker's /tmp/ directory has a storage limit
    if os.path.isdir(cfg.gpt_model_file):
        connector = SaveRestoreConnector()
        connector.model_extracted_dir = cfg.gpt_model_file
    else:
        connector = None

    model = MegatronGPTModel.restore_from(
        restore_path=cfg.gpt_model_file,
        trainer=trainer,
        save_restore_connector=connector,
    )
    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.module.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    # Check whether the DDP is initialized
    if parallel_state.is_unitialized():

        def placeholder():
            return

        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(placeholder,
                                                   trainer=model.trainer)
        model.trainer.strategy.setup_environment()

    config = OmegaConf.to_container(cfg.inference)
    model.set_inference_config(config)

    dataloader = get_calib_dataloader(cfg.inference.batch_size,
                                      cfg.inference.max_context_length)
    # =================== Start Quantization ====================
    if cfg.quantization.algorithm == "int8_sq":
        atq_config = atq.INT8_SMOOTHQUANT_CFG
        # disable quantization for the last output layer
        atq_config["quant_cfg"]["*.output_layer.*"] = {
            "enable": False
        }  # type: ignore
    elif cfg.quantization.algorithm == "fp8":
        atq_config = atq.FP8_DEFAULT_CFG
    else:
        raise ValueError(
            f"Unsupported quantization algorithm: {cfg.quantization.algorithm}")

    def forward_loop():
        for i, batch in enumerate(dataloader):
            if i >= cfg.quantization.num_calib_size:
                break
            print(f"Calibrating batch {i}")
            model.predict_step(batch, i)

    atq.quantize(model, atq_config, forward_loop)
    print("Quantization done.")
    # =================== End Quantization ======================

    export_path = cfg.get("model_save_path", os.getcwd())
    export_model_config(
        model,
        "gptnext",
        torch.bfloat16,
        quantization=cfg.quantization.algorithm,
        export_dir=export_path,
    )
    print(f"Quantized model exported to :{export_path}")


if __name__ == "__main__":
    main()
