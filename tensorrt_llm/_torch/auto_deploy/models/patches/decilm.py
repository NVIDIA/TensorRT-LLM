import re

from transformers import AutoConfig

_orig_from_pretrained = AutoConfig.from_pretrained


def _from_pretrained_patched(pretrained_model_name_or_path, **kwargs):
    print(str(pretrained_model_name_or_path))
    if re.search(r"Llama-3_(?:1|3)-Nemotron-(?:Ultra|Super)", str(pretrained_model_name_or_path)):
        kwargs["attn_implementation"] = "eager"
    return _orig_from_pretrained(pretrained_model_name_or_path, **kwargs)


# TODO: figure out how this can be incorporated into the export patch system
AutoConfig.from_pretrained = _from_pretrained_patched
