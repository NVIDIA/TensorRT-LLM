"""Shared model-construction helpers for unit builders."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TypeVar

import torch
from torch import nn
from torch.nn.parameter import is_lazy

_M = TypeVar("_M", bound=nn.Module)


def _customizes_loading(module: nn.Module) -> bool:
    cls = type(module)
    return (
        cls._load_from_state_dict is not nn.Module._load_from_state_dict
        or cls.set_extra_state is not nn.Module.set_extra_state
        or bool(module._load_state_dict_pre_hooks)
        or bool(module._load_state_dict_post_hooks)
    )


def assign_state_dict(module: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:  # noqa: PLR0912
    """Same result as ``module.load_state_dict(state_dict, strict=False, assign=True)``, in one pass.
    Each parameter / persistent buffer looks its key up directly, where ``nn.Module.load_state_dict``
    re-filters the whole state dict for every child module. Keys without a matching tensor are left
    untouched and unmatched entries are ignored (``strict=False``); a shape mismatch raises.
    Falls back to ``load_state_dict`` when any submodule customizes loading (an overridden
    ``_load_from_state_dict`` or ``set_extra_state``, load hooks) or swap-tensors loading is on,
    because those need the stock per-module protocol.
    """
    modules = list(module.named_modules(remove_duplicate=False))
    if torch.__future__.get_swap_module_params_on_conversion() or any(_customizes_loading(m) for _, m in modules):
        if not torch.is_inference_mode_enabled() and any(value.is_inference() for value in state_dict.values()):
            regularized: dict[int, torch.Tensor] = {}
            regular_state = {}
            for key, value in state_dict.items():
                supplied = value
                if value.is_inference():
                    source_id = id(value)
                    regular = regularized.get(source_id)
                    if regular is None:
                        regular = regularized[source_id] = value.clone()
                    supplied = regular
                regular_state[key] = supplied
            state_dict = regular_state
        module.load_state_dict(state_dict, strict=False, assign=True)
        return

    error_msgs: list[str] = []
    regularized_inference: dict[int, torch.Tensor] = {}
    for prefix, submodule in modules:
        key_prefix = f"{prefix}." if prefix else ""
        for name, param in submodule._parameters.items():
            tensor = None if param is None else state_dict.get(key_prefix + name)
            if tensor is None:
                continue
            tensor = _matching_shape(tensor, param, key_prefix + name, error_msgs)
            if tensor is None:
                continue
            if tensor.is_inference() and param.requires_grad and not torch.is_inference_mode_enabled():
                # A retained inference tensor cannot become a trainable Parameter outside
                # inference mode. Preserve repeated references to tied checkpoint values.
                source_id = id(tensor)
                regular = regularized_inference.get(source_id)
                if regular is None:
                    regular = regularized_inference[source_id] = tensor.clone()
                tensor = regular
            if isinstance(tensor, nn.Parameter):
                tensor.requires_grad_(param.requires_grad)
            else:
                tensor = nn.Parameter(tensor, requires_grad=param.requires_grad)
            submodule._parameters[name] = tensor
        for name, buffer in submodule._buffers.items():
            if buffer is None or name in submodule._non_persistent_buffers_set:
                continue
            tensor = state_dict.get(key_prefix + name)
            if tensor is None:
                continue
            tensor = _matching_shape(tensor, buffer, key_prefix + name, error_msgs)
            if tensor is not None:
                submodule._buffers[name] = tensor
    if error_msgs:
        raise RuntimeError(f"Error(s) in loading state_dict for {type(module).__name__}:\n\t" + "\n\t".join(error_msgs))


def _matching_shape(
    tensor: torch.Tensor, current: torch.Tensor, key: str, error_msgs: list[str]
) -> torch.Tensor | None:
    """``tensor`` reshaped the way ``load_state_dict`` accepts it for ``current``, or ``None`` on a mismatch.
    A lazy ``current`` (``UninitializedParameter`` / ``UninitializedBuffer``) has no shape yet and takes
    ``tensor`` as is, as in ``load_state_dict``.
    """
    if is_lazy(current):
        return tensor
    if current.ndim == 0 and tensor.shape == (1,):
        tensor = tensor[0]
    if tensor.shape != current.shape:
        error_msgs.append(
            f"size mismatch for {key}: copying a param with shape {tensor.shape} from checkpoint, "
            f"the shape in current model is {current.shape}."
        )
        return None
    return tensor


def module_tensors(model: nn.Module) -> Iterator[torch.Tensor]:
    """Every parameter and buffer of ``model``, read straight off each module.
    Unlike ``named_parameters`` / ``named_buffers`` there is no de-duplication, so a tensor
    shared between modules is yielded once per owner.
    """
    for module in model.modules():
        yield from (t for t in module._parameters.values() if t is not None)
        yield from (t for t in module._buffers.values() if t is not None)


def to_device(model: _M, device: torch.device) -> _M:
    """``model.to(device)``, skipped when every parameter and buffer already lives on ``device``.
    ``nn.Module.to`` rebuilds every parameter even when nothing moves, which costs a full walk of
    a large model per call.
    """
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if all(t.device == device for t in module_tensors(model)):
        return model
    return model.to(device)


def parse_model_version(version: str | None) -> tuple[int, ...]:
    """Parse a checkpoint's ``model_version`` into comparable numeric components.
    Parsing stops at the first dot-separated component that isn't a plain integer, so
    pre-release tags are dropped (the pre-2.4 era used strings like ``"2.3.rc1"``). A
    shorter numeric prefix already compares as "less than" a longer one sharing its
    leading components (``(2, 3) < (2, 4, 0)``), so dropping the tail is safe. An unset
    or non-numeric version parses to ``()``, which compares below every real version --
    callers get their oldest fallback.
    Tags are not always dot-separated (``"2.4-rc2"`` parses to ``(2,)``). Callers that
    want such a build to compare equal to its own generation should normalize the
    separator first -- see ``detect_params`` in ltx-pipelines.
    """
    if not version:
        return ()
    numeric_parts = []
    for part in version.split("."):
        if not part.isdigit():
            break
        numeric_parts.append(int(part))
    return tuple(numeric_parts)
