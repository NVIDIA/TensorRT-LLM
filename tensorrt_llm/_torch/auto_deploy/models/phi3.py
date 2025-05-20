import re

from transformers import AutoModelForCausalLM, Phi3ForCausalLM

_from_config_original = AutoModelForCausalLM.from_config


def from_config_patched(config, **kwargs):
    if re.search(r"Phi-3-(?:mini|medium)", config._name_or_path):
        kwargs.pop("trust_remote_code", None)
        return Phi3ForCausalLM(config, **kwargs)
    return _from_config_original(config, **kwargs)


AutoModelForCausalLM.from_config = from_config_patched
