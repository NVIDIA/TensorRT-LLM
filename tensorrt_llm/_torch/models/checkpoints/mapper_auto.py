from typing import Optional

from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPER_MAPPING


class CheckpointMapperAuto():

    @staticmethod
    def get(format: str, name: Optional[str] = None) -> "WeightMapperInterface":
        if name is not None:
            try:
                return MODEL_CLASS_MAPPER_MAPPING[name][format]()
            except KeyError:  # no mapper for this model architecture, resort to default
                return MODEL_CLASS_MAPPER_MAPPING[format]()
        else:
            return MODEL_CLASS_MAPPER_MAPPING[format]()
