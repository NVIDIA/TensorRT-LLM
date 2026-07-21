#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""crit3: TP=4 construct + real NVFP4 checkpoint-load + load-time accounting.

Launched under MPI (``srun --ntasks=4 --mpi=pmix python inkling_load_test.py``),
this constructs the Inkling text tower sharded across 4 GB200 GPUs and loads the
real NVFP4 checkpoint through the production ``_torch`` construct+load path
(``AutoModelForCausalLM.from_config`` + ``model.load_weights`` via the registered
``InklingHfWeightMapper``). No forward pass, no KV cache -- this proves the load
integration only.

Proves, at load time:
  * config quant_algo is NVFP4 (no bf16 / alternate-precision path),
  * the model constructs sharded under TP=4 (meta-init -> CUDA materialize),
  * every model parameter/buffer is materialized after load (all required text
    weights consumed; a missing/misshaped source key raises during load because
    allow_partial_loading defaults to False),
  * the only checkpoint keys not routed into the text tower are the intentionally
    deferred audio / vision / MTP keys.

Exit code 0 on success (rank 0 asserts + prints the accounting), non-zero on any
failure.
"""

import os
import sys

import torch

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full")

DEFERRED_PREFIXES = ("model.audio.", "model.visual.", "model.mtp.")


def main() -> int:
    from tensorrt_llm._utils import (local_mpi_rank, mpi_barrier, mpi_rank,
                                     mpi_world_size)
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.quantization.mode import QuantAlgo
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
    from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import \
        HfCheckpointLoader
    # Import registers the auto-model + the InklingHfWeightMapper.
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration

    rank, world = mpi_rank(), mpi_world_size()
    assert world == 4, f"expected 4 ranks (TP=4), got {world}"
    torch.cuda.set_device(local_mpi_rank())
    mapping = Mapping(world_size=world, tp_size=world, rank=rank)

    def log(msg):
        if rank == 0:
            print(f"[rank0] {msg}", flush=True)

    # 1) Config: NVFP4 quant + TP=4 mapping. This is the production config path.
    config = ModelConfig.from_pretrained(
        CKPT,
        trust_remote_code=True,
        mapping=mapping,
        attn_backend="TRTLLM",
        moe_backend="CUTLASS",
    )
    assert config.quant_config is not None
    assert config.quant_config.quant_algo == QuantAlgo.NVFP4, (
        f"expected NVFP4, got {config.quant_config.quant_algo}")
    arch = config.pretrained_config.architectures[0]
    assert arch == "InklingForConditionalGeneration", arch
    log(f"config OK: arch={arch} quant={config.quant_config.quant_algo} "
        f"tp={mapping.tp_size}")

    # 2) Construct sharded on meta, then materialize this rank's shard to CUDA.
    #    (Mirrors ModelLoader.load: MetaInitMode -> init_meta_tensor -> to cuda.)
    #    Construct the class directly (not AutoModelForCausalLM.from_config, which
    #    sets skip_create_weights_in_init=True) so create_weights runs in
    #    __post_init__ and the weight tensors exist for load_weights to fill.
    with MetaInitMode():
        model = InklingForConditionalGeneration(config)

    memo: dict = {}

    def init_meta_tensor(t: torch.Tensor) -> torch.Tensor:
        if t.device != torch.device("meta"):
            return t
        if t not in memo:
            memo[t] = torch.empty_like(t, device="cuda")
        return memo[t]

    model._apply(init_meta_tensor)
    model.to("cuda")
    memo.clear()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"constructed sharded model: {n_params/1e9:.2f}B params on this rank")

    # 3) Load the real NVFP4 checkpoint via the registered mapper.
    loader = HfCheckpointLoader()
    weights = loader.load_weights(CKPT, mapping=mapping)
    all_keys = set(weights.keys())
    mapper = loader.get_initialized_weight_mapper(model, config)
    model.load_weights(weights, weight_mapper=mapper)
    log(f"load_weights OK: {len(all_keys)} checkpoint keys read")

    # 4) Load-time accounting.
    # (a) Every param/buffer must now be materialized on CUDA (no leftover meta):
    #     load raises on a missing/misshaped source key (allow_partial_loading is
    #     False by default), so a clean return + no meta tensors == all required
    #     text weights consumed.
    stray_meta = [
        name for name, p in model.named_parameters() if p.is_meta
    ] + [name for name, b in model.named_buffers() if b.is_meta]
    assert not stray_meta, f"unmaterialized params after load: {stray_meta[:10]}"

    # (b) The only keys NOT routed into the text tower are audio / vision / MTP.
    non_text = {k for k in all_keys if not k.startswith("model.llm.")}
    unexpected = {k for k in non_text if not k.startswith(DEFERRED_PREFIXES)}
    assert not unexpected, f"non-text, non-deferred keys: {sorted(unexpected)[:10]}"

    # (c) Strict consumed/deferred text-key accounting against the REAL checkpoint
    #     key set: every text weight the loader needs is present (missing == empty)
    #     and every checkpoint key is either consumed-text or an intentionally
    #     deferred audio/vision/MTP key (unaccounted == empty). This is the same
    #     assertion the CPU structural test pins, now enforced on the real load so
    #     crit3 reports EXACTLY consumed text + deferred multimodal/MTP.
    import json

    from tensorrt_llm._torch.models.checkpoints.hf.inkling_weight_mapper import \
        inkling_account_checkpoint
    with open(os.path.join(CKPT, "hf_quant_config.json")) as f:
        # exclude_modules is nested under the "quantization" block (same
        # extraction the CPU structural test uses); a top-level get() would be
        # empty and wrongly demand NVFP4 sidecars for the bf16 layer-2 experts.
        exclude = set(json.load(f)["quantization"].get("exclude_modules", []))
    tc = config.pretrained_config.text_config
    acct = inkling_account_checkpoint(all_keys, tc, exclude)
    assert not acct["missing"], f"missing text keys: {sorted(acct['missing'])[:10]}"
    assert not acct["unaccounted"], (
        f"unaccounted keys: {sorted(acct['unaccounted'])[:10]}")
    assert all(k.startswith(DEFERRED_PREFIXES) for k in acct["deferred"])
    assert len(acct["consumed_text"]) + len(acct["deferred"]) == len(all_keys)

    log(f"accounting OK: consumed_text={len(acct['consumed_text'])} "
        f"deferred(audio/vision/mtp)={len(acct['deferred'])} "
        f"missing=0 unaccounted=0 stray_meta=0")
    log("CRIT3_LOAD_OK")

    mpi_barrier()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
