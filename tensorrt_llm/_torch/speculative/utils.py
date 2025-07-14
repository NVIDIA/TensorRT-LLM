from typing import Optional

from ..pyexecutor.guided_decoder import GuidedDecoder
from ..pyexecutor.sampler import TorchSampler
from ..pyexecutor.seq_slot_manager import SeqSlotManager
from ..speculative.interface import SpecMetadata
from .eagle3 import (Eagle3OneModelSampler, Eagle3OneModelSpecMetadata,
                     Eagle3OneModelWorker, Eagle3ResourceManager,
                     Eagle3SpecMetadata)
from .external_api import APIDrafter
from .model_drafter import ModelDrafter
from .mtp import (MTPEagleWorker, MTPHiddenStatesManager, MTPSampler,
                  MTPSpecMetadata, MTPWorker)
from .ngram import NGramDrafter, NGramPoolManager


def get_spec_metadata(spec_config,
                      model_config,
                      max_num_requests,
                      max_num_tokens,
                      spec_resource_manager=None,
                      is_draft_model=False):
    if spec_config.spec_dec_mode.is_mtp():
        return MTPSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            spec_dec_mode=spec_config.spec_dec_mode,
            mtp_num_modules=spec_config.num_nextn_predict_layers,
            max_num_requests=max_num_requests,
            mtp_hidden_states_manager=spec_resource_manager,
        )
    if spec_config.spec_dec_mode.is_eagle3():
        return Eagle3SpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            num_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            dtype=model_config.torch_dtype,
            is_draft_model=is_draft_model,
            eagle3_resource_manager=spec_resource_manager,
        )
    if spec_config.spec_dec_mode.is_eagle3_one_model():
        return Eagle3OneModelSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            num_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
        )
    if  spec_config.spec_dec_mode.is_draft_target() or \
        spec_config.spec_dec_mode.is_ngram() or \
        spec_config.spec_dec_mode.is_user_provided() or \
        spec_config.spec_dec_mode.is_external_api():
        return SpecMetadata(
            max_draft_len=spec_config.max_draft_len,
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
    if spec_dec_mode.is_mtp_eagle():
        if spec_config.use_relaxed_acceptance_for_thinking:
            return MTPHiddenStatesManager(
                spec_config,
                model_config.torch_dtype,
                model_config.hidden_size,
                max_num_requests,
            )
        else:
            return None
    if spec_dec_mode.is_mtp():
        return MTPHiddenStatesManager(
            spec_config,
            model_config.torch_dtype,
            model_config.hidden_size,
            max_num_requests,
        )
    if spec_dec_mode.is_eagle3():
        assert draft_model_engine is not None, "Draft model engine is required for Eagle3 two model flow."
        return Eagle3ResourceManager(
            spec_config,
            draft_model_engine.model.config.torch_dtype,
            model_config.hidden_size,
            max_num_requests,
            max_seq_len,
            max_num_tokens,
        )
    if spec_dec_mode.is_ngram():
        return NGramPoolManager(spec_config, max_num_requests)
    if spec_dec_mode.is_user_provided():
        return spec_config.resource_manager
    if spec_dec_mode.is_external_api():
        return None
    return None


def get_spec_decoder(sampler_args: TorchSampler.Args,
                     spec_config: "DecodingBaseConfig"):
    if spec_config.spec_dec_mode.is_mtp():
        return MTPSampler(sampler_args,
                          nextn=spec_config.num_nextn_predict_layers)
    if spec_config.spec_dec_mode.is_eagle3():
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
    ) or spec_config.spec_dec_mode.is_eagle3():
        return ModelDrafter(spec_config,
                            draft_model_engine,
                            spec_config.max_draft_len,
                            SeqSlotManager(max_num_requests),
                            sampler,
                            spec_resource_manager=spec_resource_manager,
                            guided_decoder=guided_decoder)

    if spec_config.spec_dec_mode.is_ngram():
        return NGramDrafter(spec_config, spec_resource_manager)
    if spec_config.spec_dec_mode.is_external_api():
        return APIDrafter(spec_config)
    return None


def get_num_spec_layers(spec_config):
    if spec_config.spec_dec_mode.is_mtp():
        return spec_config.num_nextn_predict_layers
    if spec_config.spec_dec_mode.is_eagle3_one_model():
        return 1
    return 0


def get_spec_worker(spec_config, model_config, mapping):
    spec_dec_mode = spec_config.spec_dec_mode
    if spec_dec_mode.is_mtp_vanilla():
        return MTPWorker(spec_config, model_config)
    if spec_dec_mode.is_mtp_eagle():
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
    if spec_config.spec_dec_mode.is_eagle3_one_model(
    ) or spec_config.spec_dec_mode.is_mtp_eagle():
        return spec_config.max_draft_len - 1
    return 0


def update_spec_config_from_model_config(spec_config, model_config):
    if spec_config.spec_dec_mode.is_mtp():
        # Use `max_draft_len` for several low-level APIs. TODO: Remove this after distinguishing them.
        spec_config.max_draft_len = spec_config.num_nextn_predict_layers
        # Use `num_nextn_predict_layers_from_model_config` to decide decoding mode MTP / MTP_EAGLE.
        spec_config.num_nextn_predict_layers_from_model_config = model_config.num_nextn_predict_layers
