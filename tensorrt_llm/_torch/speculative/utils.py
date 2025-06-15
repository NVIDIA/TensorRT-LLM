from .eagle3 import (Eagle3OneModelDecoder, Eagle3OneModelSpecMetadata,
                     Eagle3OneModelWorker, Eagle3ResourceManager, Eagle3Sampler,
                     Eagle3SpecMetadata)
from .mtp import (MTPEagleWorker, MTPHiddenStatesManager, MTPSampler,
                  MTPSpecMetadata, MTPWorker)
from .ngram import NGramPoolManager


def get_spec_metadata(spec_config,
                      max_num_requests,
                      max_num_tokens,
                      spec_resource_manager=None,
                      is_draft_model=False):
    if spec_config.spec_dec_mode.is_mtp():
        return MTPSpecMetadata(
            max_draft_tokens=spec_config.max_draft_tokens,
            spec_dec_mode=spec_config.spec_dec_mode,
            mtp_num_modules=spec_config.num_nextn_predict_layers,
            max_num_requests=max_num_requests,
            mtp_hidden_states_manager=spec_resource_manager)
    elif spec_config.spec_dec_mode.is_eagle3():
        return Eagle3SpecMetadata(max_draft_tokens=spec_config.max_draft_tokens,
                                  spec_dec_mode=spec_config.spec_dec_mode,
                                  max_num_requests=max_num_requests,
                                  num_layers=spec_config.num_layers,
                                  hidden_size=spec_config.hidden_size,
                                  max_num_tokens=max_num_tokens,
                                  dtype=spec_config.dtype,
                                  is_draft_model=is_draft_model,
                                  eagle3_resource_manager=spec_resource_manager)
    elif spec_config.spec_dec_mode.is_eagle3_one_model():
        return Eagle3OneModelSpecMetadata(
            max_draft_tokens=spec_config.max_draft_tokens,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            num_layers=spec_config.num_layers,
            hidden_size=spec_config.hidden_size,
            max_num_tokens=max_num_tokens)
    else:
        return None


def get_spec_resource_manager(spec_config,
                              model_engine,
                              draft_model_engine=None):
    model_config = model_engine.model.config
    max_num_requests = model_engine.batch_size
    max_seq_len = model_engine.max_seq_len
    max_num_tokens = model_engine.max_num_tokens
    if spec_config.spec_dec_mode.is_mtp_eagle():
        if spec_config.use_relaxed_acceptance_for_thinking:
            return MTPHiddenStatesManager(spec_config, model_config.torch_dtype,
                                          model_config.hidden_size,
                                          max_num_requests)
        else:
            return None
    elif spec_config.spec_dec_mode.is_mtp():
        return MTPHiddenStatesManager(spec_config, model_config.torch_dtype,
                                      model_config.hidden_size,
                                      max_num_requests)
    elif spec_config.spec_dec_mode.is_ngram():
        return NGramPoolManager(spec_config, max_num_requests)
    elif spec_config.spec_dec_mode.is_eagle3():
        assert draft_model_engine is not None, "Draft model engine is required for Eagle3 two model flow."
        draft_model_config = draft_model_engine.model.config
        return Eagle3ResourceManager(spec_config,
                                     draft_model_config.torch_dtype,
                                     model_config.hidden_size, max_num_requests,
                                     max_seq_len, max_num_tokens)
    else:
        return None


def get_spec_decoder(max_seq_len, spec_config):
    if spec_config.spec_dec_mode.is_mtp():
        return MTPSampler(max_seq_len, spec_config)
    elif spec_config.spec_dec_mode.is_eagle3():
        return Eagle3Sampler(max_seq_len)
    elif spec_config.spec_dec_mode.is_eagle3_one_model():
        return Eagle3OneModelDecoder(max_seq_len, spec_config)
    else:
        return None


def get_num_spec_layers(spec_config):
    if spec_config.spec_dec_mode.is_mtp():
        return spec_config.num_nextn_predict_layers
    elif spec_config.spec_dec_mode.is_eagle3_one_model():
        return 1
    else:
        return 0


def get_spec_worker(spec_config, mapping):
    if spec_config.spec_dec_mode.is_mtp():
        return MTPWorker(spec_config)
    elif spec_config.spec_dec_mode.is_mtp_eagle():
        return MTPEagleWorker(spec_config)
    elif spec_config.spec_dec_mode.is_eagle3_one_model():
        return Eagle3OneModelWorker(spec_config, mapping)
    else:
        return None
