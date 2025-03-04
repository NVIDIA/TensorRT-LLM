from .mtp import MTPDecoder, MTPHiddenStatesManager, MTPSpecMetadata


def get_spec_metadata(spec_config,
                      max_num_requests,
                      spec_resource_manager=None):
    if spec_config.spec_dec_name == "MTP":
        return MTPSpecMetadata(
            max_draft_tokens=spec_config.max_draft_tokens,
            spec_dec_mode=spec_config.spec_dec_mode,
            mtp_num_modules=spec_config.num_nextn_predict_layers,
            max_num_requests=max_num_requests,
            mtp_hidden_states_manager=spec_resource_manager)
    else:
        return None


def get_spec_resource_manager(spec_config, model_config, max_num_requests):
    if spec_config.spec_dec_name == "MTP":
        return MTPHiddenStatesManager(spec_config, model_config.torch_dtype,
                                      model_config.hidden_size,
                                      max_num_requests)
    else:
        return None


def get_spec_decoder(max_seq_len, spec_config):
    if spec_config.spec_dec_name == "MTP":
        return MTPDecoder(max_seq_len, spec_config)
    else:
        return None


def get_num_spec_layers(spec_config):
    if spec_config.spec_dec_name == "MTP":
        return spec_config.num_nextn_predict_layers
    else:
        return 0
