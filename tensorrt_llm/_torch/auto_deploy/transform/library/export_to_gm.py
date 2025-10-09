"""A simple wrapper transform to export a model to a graph module."""

import inspect
from contextlib import contextmanager
from inspect import Parameter, Signature
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from pydantic import Field

from ...export import run_forward_for_capture, torch_export_to_gm
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class ExportToGMConfig(TransformConfig):
    """Configuration for the export to graph module transform."""

    strict: bool = Field(
        description="Whether to export in strict mode. NOTE: we generally export in non-strict mode"
        "for now as it relaxes some assumptions around tracing. Strict mode uses torchdynamo"
        "(symbolic bytecode analysis), which can be brittle since it relies on the exact bytecode"
        "representation of the model see here as well: https://pytorch.org/docs/stable/export.html#non-strict-export",
        default=False,
    )
    clone_state_dict: bool = Field(
        description="Whether to clone the state_dict of the model. This is useful to avoid"
        "modifying the original state_dict of the model.",
        default=False,
    )
    patch_list: Optional[List[str]] = Field(
        description="List of patch names to apply with export. "
        "Default is to apply all registered patches.",
        default=None,
    )


@contextmanager
def capture_forward_kwargs(mod: nn.Module):
    """Context manager to capture the keyword arguments of the forward pass of a module."""
    captured_kwargs = {}

    def _capture_kwargs(mod: nn.Module, args, kwargs) -> None:
        assert not args, "positional arguments are not supported for capture"
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)
        return None

    try:
        # make sure to prepend the hook so that it is called before any other hooks so we can
        # capture the original inputs before other potential hooks have a chance to modify them.
        # NOTE: this simulates the behavior during torch.export.
        handle = mod.register_forward_pre_hook(_capture_kwargs, prepend=True, with_kwargs=True)
        yield captured_kwargs
    finally:
        handle.remove()


@contextmanager
def set_exact_signature(mod: nn.Module, kwargs: Dict[str, Any]):
    """Temporarily set a signature for the forward function corresponding to provided kwargs.

    Args:
        mod: The module to set the signature for.
        kwargs: The keyword arguments to set the signature for.


    Within this context, will have a signature corresponding to only taking the provided kwargs as
    keyword-only parameters (+self if it is a method).
    """
    is_method = inspect.ismethod(mod.forward)
    if is_method:
        forward_func = mod.forward.__func__
    elif inspect.isfunction(mod.forward):
        forward_func = mod.forward
    else:
        raise ValueError(f"Unsupported forward function type: {type(mod.forward)}")

    signature_inspected = inspect.signature(forward_func)

    reset_signature = False
    if hasattr(forward_func, "__signature__"):
        signature_attribute = forward_func.__signature__
        reset_signature = True

    # construct signature object from kwargs
    params_list = []
    if is_method:
        # heuristic to identify the self parameter
        param_keys = list(signature_inspected.parameters.keys())
        self_key = "self" if "self" in param_keys else param_keys[0]
        params_list.append(signature_inspected.parameters[self_key].replace())
    # the rest of the parameters as keyword only
    params_list.extend(
        [Parameter(k, kind=Parameter.KEYWORD_ONLY, annotation=type(v)) for k, v in kwargs.items()]
    )
    forward_func.__signature__ = Signature(parameters=params_list)
    try:
        yield
    finally:
        if reset_signature:
            forward_func.__signature__ = signature_attribute
        else:
            del forward_func.__signature__


@TransformRegistry.register("export_to_gm")
class ExportToGM(BaseTransform):
    """A simple wrapper transform to export a model to a graph module."""

    config: ExportToGMConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return ExportToGMConfig

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        # set the example sequence
        cm.info.set_example_sequence(**factory.get_example_inputs())

        export_infos = factory.get_export_infos(mod)

        # check if any submodules to be exported are children of other submodules that need to be
        # exported. We don't allow for this since this may imply that the submodules are not
        # independent, which would conflict with graph capture logic, i.e., you cannot graph-capture
        # "model" and "model.text_model" for example. However, you can export "model.text_model" and
        # "model.vision_model" separately.
        def _is_child(child: str, parent: str) -> bool:
            """Check if ``child`` is a child of ``parent``."""
            # covers "a.b.c" is a parent of "a.b" or parent being "", i.e., root (a parent of all!)
            return parent == "" or child.startswith(f"{parent}.")

        sub_keys = [info.submodule_name for info in export_infos]
        assert all(not _is_child(k1, k2) for k1 in sub_keys for k2 in sub_keys if k1 != k2), (
            f"Cannot export submodules of already exported submodules, {sub_keys=}"
        )

        for e_info in export_infos:
            sub_mod = mod.get_submodule(e_info.submodule_name)

            # start by capturing the kwargs that are passed to the submodule for export
            with capture_forward_kwargs(sub_mod) as captured_kwargs:
                run_forward_for_capture(
                    mod,
                    args=(),
                    kwargs=cm.named_args,
                    clone=self.config.clone_state_dict,
                    patch_list=self.config.patch_list,
                )

            # construct dynamic shapes based on the captured kwargs and the dynamic shape lookup
            dynamic_shapes = {
                k: e_info.dynamic_shape_lookup[k] if isinstance(v, torch.Tensor) else None
                for k, v in captured_kwargs.items()
            }

            # export the model to a graph module. We temporarily overwrite the signature of the
            # forward function to exactly match the kwargs we pass in. This is to ensure that
            # torch.export's graph capture can correctly handle all inputs. Specifically,
            # torch.export can get confused by keyword arguments that are not explicitly defined in
            # the signature but are captured through generic **kwargs. By overwriting the signature,
            # we ensure each argument is explicitly defined in the signature.
            with set_exact_signature(sub_mod, captured_kwargs):
                sub_gm = torch_export_to_gm(
                    sub_mod,
                    args=(),
                    kwargs=captured_kwargs,
                    dynamic_shapes=dynamic_shapes,
                    clone=self.config.clone_state_dict,
                    strict=self.config.strict,
                    patch_list=self.config.patch_list,
                )

            # post process the sub graph module
            e_info.post_process(sub_mod, sub_gm)

            # set the sub graph module
            if e_info.submodule_name == "":
                mod = sub_gm
            else:
                mod.set_submodule(e_info.submodule_name, sub_gm)

        # this is a clean graph by definition since it was just exported
        info = TransformInfo(
            skipped=False, num_matches=len(sub_keys), is_clean=True, has_valid_shapes=True
        )

        return mod, info
