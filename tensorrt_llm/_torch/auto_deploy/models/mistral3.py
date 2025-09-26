"""Auto-deploy model factory for Mistral3 models."""

from typing import Dict, Tuple

import torch

from tensorrt_llm._torch.auto_deploy.custom_ops import attention_interface
from tensorrt_llm._torch.auto_deploy.models import factory, hf


@factory.ModelFactoryRegistry.register("Mistral3VLM")
class Mistral3VLM(hf.AutoModelForImageTextToTextFactory):
    def get_extra_inputs(
        self,
    ) -> Dict[str, Tuple[torch.Tensor, attention_interface.DynamicShapeCallback]]:
        """Return a dictionary of extra inputs for the model.

        Returns:
            A dictionary of extra inputs for the model where the key corresponds to the argument
            name and the value corresponds to a tuple of (example_input, dynamic_shape_callback).
            The dynamic shape callback is a function that returns the dynamic shape of the extra
            input.
        """
        extra_inputs = super().get_extra_inputs()
        # Reuse the same dynamic batch dimension for `image_sizes`.
        batch_dim = extra_inputs["pixel_values"][1]()[0]
        extra_inputs["image_sizes"] = (torch.zeros(0, 2, dtype=torch.long), lambda: {0: batch_dim})

        return extra_inputs

    @staticmethod
    def _strict_forward(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
    ):
        """A strict (args-only) forward pass for the model to functionalize the args.

        It adds ``pixel_values`` and ``image_sizes`` as a positional argument as expected by
        Mistral3Model in addition to the required ``input_ids`` and ``position_ids``.
        """
        return type(model).forward(
            model,
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )

    @property
    def _example_image_dims(self) -> Tuple[int, int]:
        # The pixtral processor requires a minimum image size, which is larger than the default (16, 16)
        # in the parent class.
        # TODO: figure this out on the model config somehow (patch size value, etc.).
        return (64, 64)
