from bisect import bisect_left
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import torch

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

from ..pyexecutor.guided_decoder import GuidedDecoder
from ..pyexecutor.sampler import TorchSampler
from ..pyexecutor.seq_slot_manager import SeqSlotManager
from ..speculative.interface import SpecMetadata
from .dflash import DFlashSpecMetadata, DFlashWorker
from .draft_target import (DraftTargetOneModelSampler,
                           DraftTargetOneModelSpecMetadata,
                           DraftTargetOneModelWorker)
from .eagle3 import (Eagle3OneModelDynamicTreeResourceManager,
                     Eagle3OneModelSampler, Eagle3OneModelSpecMetadata,
                     Eagle3OneModelWorker, Eagle3ResourceManager,
                     Eagle3SpecMetadata, MTPEagleWorker)
from .eagle3_dynamic_tree import Eagle3OneModelDynamicTreeWorker
from .model_drafter import ModelDrafter
from .mtp import MTPHiddenStatesManager, MTPSampler, MTPSpecMetadata, MTPWorker
from .ngram import NGramDrafter, NGramPoolManager
from .pard import PARDSpecMetadata, PARDWorker
from .sa_worker import SASampler, SASpecMetadata, SAWorker
from .save_hidden_state import (SaveHiddenStatesResourceManager,
                                SaveHiddenStatesSpecMetadata)
from .suffix_automaton import SuffixAutomatonManager


def get_spec_metadata(spec_config,
                      model_config,
                      max_num_requests,
                      max_num_tokens,
                      spec_resource_manager=None,
                      is_draft_model=False,
                      max_seq_len=262144):
    use_rejection_sampling = getattr(spec_config, "use_rejection_sampling",
                                     False)
    vocab_size = getattr(model_config, "vocab_size", 0)
    if spec_config.spec_dec_mode.is_mtp_eagle_one_model():
        # MTP Eagle one-model reuses Eagle3 one-model metadata for the
        # unified worker/sampler/slot_ids plumbing, but skips per-layer
        # hidden-state capture: the worker feeds the target model's
        # hidden_states directly into the MTP layer, so we leave
        # layers_to_capture unset and let Eagle3OneModelSpecMetadata default
        # it to an empty tuple. This also keeps post-MLP/MoE fusion enabled
        # on models that gate it on is_layer_capture().
        return Eagle3OneModelSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            num_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            allow_advanced_sampling=spec_config.allow_advanced_sampling,
            use_rejection_sampling=use_rejection_sampling,
            vocab_size=vocab_size,
            spec_resource_manager=spec_resource_manager,
        )
    if spec_config.spec_dec_mode.is_mtp_vanilla():
        return MTPSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            mtp_num_modules=spec_config.max_draft_len,
            max_num_requests=max_num_requests,
            mtp_hidden_states_manager=spec_resource_manager,
            allow_advanced_sampling=spec_config.allow_advanced_sampling,
        )
    if spec_config.spec_dec_mode.is_mtp_eagle():
        return Eagle3SpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            num_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            dtype=model_config.torch_dtype,
            is_draft_model=is_draft_model,
            eagle3_resource_manager=spec_resource_manager,
            layers_to_capture=None,
            is_mtp_eagle=True,
        )
    if spec_config.spec_dec_mode.is_eagle3():
        return Eagle3SpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            num_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            dtype=model_config.torch_dtype,
            is_draft_model=is_draft_model,
            eagle3_resource_manager=spec_resource_manager,
            layers_to_capture=spec_config.eagle3_layers_to_capture,
            is_mtp_eagle=False,
            eagle_choices=spec_config.eagle_choices,
            is_spec_dec_tree=spec_config.eagle_choices is not None
            or spec_config.use_dynamic_tree,
            is_spec_dec_dynamic_tree=spec_config.use_dynamic_tree,
        )
    if spec_config.spec_dec_mode.is_eagle3_one_model():
        return Eagle3OneModelSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            num_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            layers_to_capture=spec_config.eagle3_layers_to_capture,
            allow_advanced_sampling=spec_config.allow_advanced_sampling,
            use_rejection_sampling=use_rejection_sampling,
            vocab_size=vocab_size,
            spec_resource_manager=spec_resource_manager,
            use_dynamic_tree=spec_config.use_dynamic_tree,
            eagle_choices=spec_config.eagle_choices,
        )
    if spec_config.spec_dec_mode.is_pard():
        return PARDSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            allow_advanced_sampling=spec_config.allow_advanced_sampling,
            spec_resource_manager=spec_resource_manager,
        )
    if spec_config.spec_dec_mode.is_dflash():
        target_layer_ids = getattr(spec_config, 'target_layer_ids', None)
        return DFlashSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            allow_advanced_sampling=spec_config.allow_advanced_sampling,
            layers_to_capture=target_layer_ids,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            dtype=model_config.torch_dtype,
        )
    if spec_config.spec_dec_mode.is_draft_target_one_model():
        return DraftTargetOneModelSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            max_num_tokens=max_num_tokens,
            allow_advanced_sampling=spec_config.allow_advanced_sampling,
        )
    if spec_config.spec_dec_mode.is_save_hidden_states():
        return SaveHiddenStatesSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            num_model_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            dtype=model_config.torch_dtype,
            resource_manager=spec_resource_manager,
            layers_to_capture=spec_config.eagle3_layers_to_capture,
        )
    if spec_config.spec_dec_mode.is_sa():
        return SASpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            sa_manager=spec_resource_manager,
            max_matching_ngram_size=spec_config.max_matching_ngram_size,
        )
    if  spec_config.spec_dec_mode.is_draft_target() or \
        spec_config.spec_dec_mode.is_ngram() or \
        spec_config.spec_dec_mode.is_user_provided():
        return SpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
        )
    return None


def get_mtp_hidden_size(model_config) -> int:
    pretrained_config = getattr(model_config, "pretrained_config", model_config)
    hidden_size = getattr(pretrained_config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(model_config, "hidden_size")
    if getattr(pretrained_config, "model_type", None) == "deepseek_v4":
        return hidden_size * getattr(pretrained_config, "hc_mult", 1)
    return hidden_size


def get_spec_resource_manager(model_engine, draft_model_engine=None):
    spec_config = model_engine.spec_config
    if spec_config is None:
        return None
    model_config = model_engine.model.config
    max_num_requests = model_engine.batch_size
    max_seq_len = model_engine.max_seq_len
    max_num_tokens = model_engine.max_num_tokens
    spec_dec_mode = spec_config.spec_dec_mode
    if spec_dec_mode.is_mtp_eagle_one_model():
        sa_manager = None
        sa_cfg = getattr(spec_config, 'sa_config', None)
        if sa_cfg is not None:
            sa_manager = SuffixAutomatonManager(sa_cfg, max_num_requests,
                                                max_seq_len)
        if spec_config.use_relaxed_acceptance_for_thinking or sa_manager is not None:
            # Unified resource manager: the unified worker reads
            # ``relaxed_delta_pool`` from ``Eagle3ResourceManager`` (mirrors the
            # pool ``MTPHiddenStatesManager`` used to provide).
            return Eagle3ResourceManager(
                spec_config,
                model_config.torch_dtype,
                get_mtp_hidden_size(model_config),
                max_num_requests,
                max_seq_len,
                max_num_tokens,
                sa_manager=sa_manager,
            )
        else:
            return None
    if spec_dec_mode.is_mtp_vanilla():
        sa_manager = None
        sa_cfg = getattr(spec_config, 'sa_config', None)
        if sa_cfg is not None:
            sa_manager = SuffixAutomatonManager(sa_cfg, max_num_requests,
                                                max_seq_len)
        return MTPHiddenStatesManager(
            spec_config,
            model_config.torch_dtype,
            get_mtp_hidden_size(model_config),
            max_num_requests,
            sa_manager=sa_manager,
        )
    if spec_dec_mode.is_eagle3_one_model() and getattr(
            spec_config, 'use_dynamic_tree', False):
        return Eagle3OneModelDynamicTreeResourceManager(spec_config,
                                                        max_num_requests)
    if spec_dec_mode.is_eagle3_one_model():
        sa_manager = None
        sa_cfg = getattr(spec_config, 'sa_config', None)
        if sa_cfg is not None:
            sa_manager = SuffixAutomatonManager(sa_cfg, max_num_requests,
                                                max_seq_len)
        return Eagle3ResourceManager(
            spec_config,
            model_config.torch_dtype,
            model_config.hidden_size,
            max_num_requests,
            max_seq_len,
            max_num_tokens,
            sa_manager=sa_manager,
        )
    if spec_dec_mode.is_eagle3() or spec_dec_mode.is_mtp_eagle():
        assert draft_model_engine is not None, "Draft model engine is required for Eagle3 and MTP Eagle two model flow."
        return Eagle3ResourceManager(
            spec_config,
            draft_model_engine.model.config.torch_dtype,
            model_config.hidden_size,
            max_num_requests,
            max_seq_len,
            max_num_tokens,
        )
    if spec_dec_mode.is_save_hidden_states():
        return SaveHiddenStatesResourceManager(
            spec_config,
            model_engine.model.config.torch_dtype,
            model_config.hidden_size,
            max_num_requests,
            max_num_tokens,
        )
    if spec_dec_mode.is_parallel_draft():
        sa_cfg = getattr(spec_config, 'sa_config', None)
        if sa_cfg is not None:
            return SuffixAutomatonManager(sa_cfg, max_num_requests, max_seq_len)
        return None
    if spec_dec_mode.is_ngram():
        return NGramPoolManager(spec_config, max_num_requests)
    if spec_dec_mode.is_sa():
        return SuffixAutomatonManager(spec_config, max_num_requests,
                                      max_seq_len)
    if spec_dec_mode.is_user_provided():
        return spec_config.resource_manager
    return None


def get_spec_decoder(
    sampler_args: TorchSampler.Args,
    spec_config: "DecodingBaseConfig",
):
    if spec_config.spec_dec_mode.is_mtp_eagle_one_model():
        # MTP Eagle one-model now uses the same sampler as Eagle3 one-model.
        return Eagle3OneModelSampler(sampler_args, spec_config=spec_config)
    if spec_config.spec_dec_mode.is_mtp_vanilla():
        return MTPSampler(sampler_args, nextn=spec_config.max_draft_len)
    if spec_config.spec_dec_mode.is_eagle3(
    ) or spec_config.spec_dec_mode.is_mtp_eagle():
        # TorchSampler handles Eagle3 gracefully, by integrating d2t into the sampling process
        return TorchSampler(sampler_args)
    if spec_config.spec_dec_mode.is_eagle3_one_model():
        return Eagle3OneModelSampler(sampler_args, spec_config=spec_config)
    if spec_config.spec_dec_mode.is_parallel_draft():
        return MTPSampler(sampler_args,
                          nextn=spec_config.tokens_per_gen_step - 1)
    if spec_config.spec_dec_mode.is_sa():
        return SASampler(sampler_args, max_draft_len=spec_config.max_draft_len)
    if spec_config.spec_dec_mode.is_draft_target_one_model():
        return DraftTargetOneModelSampler(sampler_args)
    raise ValueError(
        f"Unsupported speculative decoding mode: {spec_config.spec_dec_mode}")


def get_spec_drafter(model_engine,
                     draft_model_engine,
                     sampler,
                     spec_resource_manager,
                     guided_decoder: Optional[GuidedDecoder] = None):
    spec_config = model_engine.spec_config
    if spec_config is None:
        return None

    if spec_config.spec_dec_mode.is_user_provided():
        return spec_config.drafter

    max_num_requests = model_engine.batch_size
    if spec_config.spec_dec_mode.is_draft_target(
    ) or spec_config.spec_dec_mode.is_eagle3(
    ) or spec_config.spec_dec_mode.is_mtp_eagle():
        return ModelDrafter(spec_config,
                            draft_model_engine,
                            spec_config.max_draft_len,
                            spec_config.tokens_per_gen_step - 1,
                            SeqSlotManager(max_num_requests),
                            sampler,
                            spec_resource_manager=spec_resource_manager,
                            guided_decoder=guided_decoder)

    if spec_config.spec_dec_mode.is_ngram():
        return NGramDrafter(spec_config, spec_resource_manager)

    return None


def get_num_spec_layers(spec_config):

    def _mode_matches(predicate_name: str) -> bool:
        predicate = getattr(spec_config.spec_dec_mode, predicate_name, None)
        return predicate is not None and predicate()

    if _mode_matches("is_mtp_eagle_one_model"):
        return 1
    if _mode_matches("is_mtp_vanilla"):
        return spec_config.num_nextn_predict_layers
    if _mode_matches("is_eagle3_one_model"):
        num_eagle_layers = spec_config.num_eagle_layers
        return num_eagle_layers if num_eagle_layers is not None else 1
    return 0


def get_spec_worker(spec_config,
                    model_config,
                    mapping,
                    use_separate_draft_kv_cache: bool = False):
    spec_dec_mode = spec_config.spec_dec_mode
    if spec_dec_mode.is_mtp_vanilla():
        return MTPWorker(spec_config, model_config, use_separate_draft_kv_cache)
    if spec_dec_mode.is_mtp_eagle_one_model():
        return MTPEagleWorker(spec_config, model_config,
                              use_separate_draft_kv_cache)
    if spec_dec_mode.is_eagle3_one_model():
        if getattr(spec_config, 'use_dynamic_tree', False):
            return Eagle3OneModelDynamicTreeWorker(spec_config, mapping,
                                                   use_separate_draft_kv_cache)
        return Eagle3OneModelWorker(
            spec_config,
            mapping=mapping,
            use_separate_draft_kv_cache=use_separate_draft_kv_cache)
    if spec_dec_mode.is_pard():
        return PARDWorker(spec_config, mapping, use_separate_draft_kv_cache)
    if spec_dec_mode.is_dflash():
        return DFlashWorker(spec_config, mapping, use_separate_draft_kv_cache)
    if spec_dec_mode.is_sa():
        return SAWorker(spec_config, model_config)
    if spec_dec_mode.is_draft_target_one_model():
        return DraftTargetOneModelWorker(spec_config, mapping,
                                         use_separate_draft_kv_cache)
    return None


def get_num_extra_kv_tokens(spec_config):
    """
    Implementation detail for one model implementations of speculative decoding. Extra
    KV cache tokens are required.
    """
    if spec_config is None:
        return 0
    if spec_config.spec_dec_mode.use_one_engine():
        return spec_config.max_draft_len - 1
    return 0


def get_draft_kv_cache_manager(spec_config, resource_manager):
    """
    Returns the draft KV cache manager only in one-model speculative decoding
    mode where the target model manages a separate draft KV cache.
    """
    from ..pyexecutor.resource_manager import ResourceManagerType

    if spec_config is None:
        return None
    if not spec_config.spec_dec_mode.use_one_engine():
        return None
    return resource_manager.get_resource_manager(
        ResourceManagerType.DRAFT_KV_CACHE_MANAGER)


def update_spec_config_from_model_config(spec_config, model_config):
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
    if not isinstance(spec_config, MTPDecodingConfig):
        return
    # Read the MTP layer count from the model's pretrained config. This
    # determines the actual MTP layer count in the checkpoint and drives the
    # spec_dec_mode decision (EAGLE vs vanilla MTP). Different checkpoints expose
    # this under different names: DeepSeek-style configs use
    # `num_nextn_predict_layers`, while Qwen3Next-style configs (including
    # Qwen3.5) use `mtp_num_hidden_layers`. Fall back to a single shared MTP /
    # EAGLE layer when neither field is present.
    num_nextn_predict_layers = getattr(model_config, "num_nextn_predict_layers",
                                       None)
    if num_nextn_predict_layers is None:
        num_nextn_predict_layers = getattr(model_config,
                                           "mtp_num_hidden_layers", None)
    if num_nextn_predict_layers is None:
        num_nextn_predict_layers = 1
    spec_config.num_nextn_predict_layers = num_nextn_predict_layers
    is_vanilla = spec_config.spec_dec_mode.is_mtp_vanilla()

    # Resolve max_draft_len when the user didn't set it:
    #   vanilla MTP -> use all checkpoint MTP heads
    #   MTP-Eagle   -> replay the single head once
    if spec_config.max_draft_len is None:
        spec_config.max_draft_len = (spec_config.num_nextn_predict_layers
                                     if is_vanilla else 1)
    elif is_vanilla and spec_config.max_draft_len != spec_config.num_nextn_predict_layers:
        effective_draft_len = min(spec_config.max_draft_len,
                                  spec_config.num_nextn_predict_layers)
        logger.warning(
            f"MTP: max_draft_len ({spec_config.max_draft_len}) does not match "
            f"num_nextn_predict_layers ({spec_config.num_nextn_predict_layers}); "
            f"using max_draft_len={effective_draft_len} draft tokens.")
        spec_config.max_draft_len = effective_draft_len

    spec_config.max_total_draft_tokens = spec_config.max_draft_len


@dataclass
class SpecDecodingTensor:
    """
    Container for speculative decoding tensor parameters.

    Attributes:
        position_offsets: Position offsets for speculative decoding
        packed_mask: Packed attention mask for speculative decoding
        generation_lengths: Optional generation lengths for speculative decoding
    """
    position_offsets: torch.Tensor
    packed_mask: torch.Tensor
    generation_lengths: Optional[torch.Tensor] = None


def get_draft_len_for_batch_size(draft_len_schedule: Dict[int, int],
                                 batch_size: int, max_draft_len: int) -> int:
    """
    Get the appropriate draft length for the given batch size using binary search.

    This is a standalone function that can be used by both the drafter (two-model path)
    and the model engine / spec workers (one-model path).

    New semantics: Keys represent specific batch sizes (transition points).
    Values represent draft_len to use for batch sizes UP TO that key.

    Args:
        draft_len_schedule: Mapping from batch size thresholds to draft lengths.
                            Example: {4: 4, 8: 2, 32: 1} means:
                            - batch size 1-4:   use draft_len=4 (up to key 4)
                            - batch size 5-8:   use draft_len=2 (up to key 8)
                            - batch size 9-32:  use draft_len=1 (up to key 32)
                            - batch size 33+:   use draft_len=0 (speculation disabled, implicit)
        batch_size: Current batch size.
        max_draft_len: Maximum draft length to use if no schedule is provided.

    Returns:
        The draft length to use for this batch size.
    """
    if draft_len_schedule is None:
        return max_draft_len

    # Binary search to find the first threshold >= batch_size
    # draft_len_schedule is already sorted by config validator
    schedule_batch_sizes = list(draft_len_schedule.keys())

    # bisect_left finds where to insert batch_size to keep list sorted
    # This gives us the index of the first key >= batch_size
    idx = bisect_left(schedule_batch_sizes, batch_size)

    if idx < len(schedule_batch_sizes):
        return draft_len_schedule[schedule_batch_sizes[idx]]

    # batch_size > all batch sizes in draft_len_schedule: speculation disabled (implicit)
    return 0
