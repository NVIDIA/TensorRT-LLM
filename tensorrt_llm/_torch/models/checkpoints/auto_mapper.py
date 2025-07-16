from typing import Optional

from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPER_MAPPING


class AutoCheckpointMapper():

    @staticmethod
    def get(format: str, name: Optional[str] = None) -> "BaseWeightMapper":
        if name is not None:
            try:
                return MODEL_CLASS_MAPPER_MAPPING[f'{name}_{format}']()
            except KeyError:  # no mapper for this model architecture, resort to default
                # TODO smor- a potential bug here, if the class isn't added to __init__, it will return the default mapper
                return MODEL_CLASS_MAPPER_MAPPING[format]()
        else:
            return MODEL_CLASS_MAPPER_MAPPING[format]()
