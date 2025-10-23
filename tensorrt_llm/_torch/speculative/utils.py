from dataclasses import dataclass
from typing import Optional

import torch

from ..pyexecutor.guided_decoder import GuidedDecoder
from ..pyexecutor.sampler import TorchSampler
from ..pyexecutor.seq_slot_manager import SeqSlotManager
from ..speculative.interface import SpecMetadata
from .eagle3 import (Eagle3OneModelSampler, Eagle3OneModelSpecMetadata,
                     Eagle3OneModelWorker, Eagle3ResourceManager,
                     Eagle3SpecMetadata)
from .model_drafter import ModelDrafter
from .mtp import (MTPEagleWorker, MTPHiddenStatesManager, MTPSampler,
                  MTPSpecMetadata, MTPWorker)
from .ngram import NGramDrafter, NGramPoolManager
from .save_hidden_state import SaveHiddenStatesDrafter


def get_spec_metadata(spec_config,
                      model_config,
                      max_num_requests,
                      max_num_tokens,
                      spec_resource_manager=None,
                      is_draft_model=False):
    if spec_config.spec_dec_mode.is_mtp_one_model():
        return MTPSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
            spec_dec_mode=spec_config.spec_dec_mode,
            mtp_num_modules=spec_config.num_nextn_predict_layers,
            max_num_requests=max_num_requests,
            mtp_hidden_states_manager=spec_resource_manager,
        )
    if spec_config.spec_dec_mode.is_mtp_eagle():
        return Eagle3SpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
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
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
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
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            num_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            layers_to_capture=spec_config.eagle3_layers_to_capture,
        )
    if spec_config.spec_dec_mode.is_save_hidden_states():
        if spec_config.eagle3_layers_to_capture is None:
            spec_config.eagle3_layers_to_capture = {
                1, model_config.num_hidden_layers // 2 - 1,
                model_config.num_hidden_layers - 4, -1
            }
        return Eagle3SpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            num_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            dtype=model_config.torch_dtype,
            is_draft_model=is_draft_model,
            eagle3_resource_manager=spec_resource_manager,
            layers_to_capture=spec_config.eagle3_layers_to_capture,
        )
    if  spec_config.spec_dec_mode.is_draft_target() or \
        spec_config.spec_dec_mode.is_ngram() or \
        spec_config.spec_dec_mode.is_user_provided():
        return SpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
        )
    return None


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
        if spec_config.use_relaxed_acceptance_for_thinking:
            return MTPHiddenStatesManager(
                spec_config,
                model_config.torch_dtype,
                model_config.hidden_size,
                max_num_requests,
            )
        else:
            return None
    if spec_dec_mode.is_mtp_one_model():
        return MTPHiddenStatesManager(
            spec_config,
            model_config.torch_dtype,
            model_config.hidden_size,
            max_num_requests,
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
        return Eagle3ResourceManager(
            spec_config,
            model_engine.model.config.torch_dtype,
            model_config.hidden_size,
            max_num_requests,
            max_seq_len,
            max_num_tokens,
        )
    if spec_dec_mode.is_ngram():
        return NGramPoolManager(spec_config, max_num_requests)
    if spec_dec_mode.is_user_provided():
        return spec_config.resource_manager
    return None


def get_spec_decoder(sampler_args: TorchSampler.Args,
                     spec_config: "DecodingBaseConfig"):
    if spec_config.spec_dec_mode.is_mtp_one_model():
        return MTPSampler(sampler_args,
                          nextn=spec_config.num_nextn_predict_layers)
    if spec_config.spec_dec_mode.is_eagle3(
    ) or spec_config.spec_dec_mode.is_mtp_eagle():
        # TorchSampler handles Eagle3 gracefully, by integrating d2t into the sampling process
        return TorchSampler(sampler_args)
    if spec_config.spec_dec_mode.is_eagle3_one_model():
        return Eagle3OneModelSampler(sampler_args)
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
                            spec_config.max_total_draft_tokens,
                            SeqSlotManager(max_num_requests),
                            sampler,
                            spec_resource_manager=spec_resource_manager,
                            guided_decoder=guided_decoder)

    if spec_config.spec_dec_mode.is_ngram():
        return NGramDrafter(spec_config, spec_resource_manager)

    if spec_config.spec_dec_mode.is_save_hidden_states():
        return SaveHiddenStatesDrafter(spec_config, spec_resource_manager)

    return None


def get_num_spec_layers(spec_config):
    if spec_config.spec_dec_mode.is_mtp_one_model():
        return spec_config.num_nextn_predict_layers
    if spec_config.spec_dec_mode.is_eagle3_one_model():
        num_eagle_layers = spec_config.num_eagle_layers
        return num_eagle_layers if num_eagle_layers is not None else 1
    return 0


def get_spec_worker(spec_config, model_config, mapping):
    spec_dec_mode = spec_config.spec_dec_mode
    if spec_dec_mode.is_mtp_vanilla():
        return MTPWorker(spec_config, model_config)
    if spec_dec_mode.is_mtp_eagle_one_model():
        return MTPEagleWorker(spec_config, model_config)
    if spec_dec_mode.is_eagle3_one_model():
        return Eagle3OneModelWorker(spec_config, mapping)
    return None


def get_num_extra_kv_tokens(spec_config):
    """
    Implementation detail for one model implementations of speculative decoding. Extra
    KV cache tokens are required.
    """
    if spec_config is None:
        return 0
    if spec_config.spec_dec_mode.is_eagle3_one_model():
        return spec_config.max_draft_len - 1
    return 0


def update_spec_config_from_model_config(spec_config, model_config):
    if spec_config.spec_dec_mode.is_mtp_one_model():
        # Use `max_draft_len` for several low-level APIs. TODO: Remove this after distinguishing them.
        spec_config.max_draft_len = spec_config.num_nextn_predict_layers
        # Use `num_nextn_predict_layers_from_model_config` to decide decoding mode MTP / MTP_EAGLE.
        spec_config.num_nextn_predict_layers_from_model_config = model_config.num_nextn_predict_layers


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
