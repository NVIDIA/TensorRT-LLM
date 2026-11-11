"""Model hooks that request and release streamed units."""

import sys
from contextlib import ExitStack
from typing import Any

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.streaming_scheduler import StreamingScheduler
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.units import StreamedUnit


class StreamingModel(torch.nn.Module):
    """Delegate model methods while inner layer hooks drive delivery.
    Each triggering module must finish all work that reads its streamed weights on the stream that is
    current when it returns: join any side stream back before returning. A device buffer is reused after
    that stream's work, not after work on other streams.
    ``stack`` holds what the scheduler reads through, such as its checkpoint
    reader session; it closes after the scheduler.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scheduler: StreamingScheduler,
        stack: ExitStack | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self._stack = stack if stack is not None else ExitStack()
        self._hooks: list[Any] = []
        self._closed = False
        for patched in scheduler.units:
            spec = patched.clean
            uid = spec.unit.unit_id
            if not isinstance(spec.unit, StreamedUnit):
                raise ValueError(f"streamed unit {uid!r} has no trigger path")
            path = spec.unit.trigger_path
            module = model.get_submodule(path)
            prefix = f"{path}."

            def before(
                _module: torch.nn.Module, _args: tuple[Any, ...], *, unit_id: str = uid, stem: str = prefix
            ) -> None:
                values = self.scheduler.need(unit_id)
                for key, value in values.items():
                    relative = key.removeprefix(stem)
                    if relative == key:
                        raise ValueError(f"{key!r} is not owned by trigger {stem!r}")
                    parent_name, _, field = relative.rpartition(".")
                    parent = _module.get_submodule(parent_name) if parent_name else _module
                    if field in parent._parameters:
                        setattr(parent, field, torch.nn.Parameter(value, requires_grad=False))
                    elif field in parent._buffers:
                        setattr(parent, field, value)
                    else:
                        raise KeyError(f"{key!r} is neither a parameter nor a buffer")

            def after(_module: torch.nn.Module, _args: tuple[Any, ...], _result: object, *, unit_id: str = uid) -> None:
                original = sys.exc_info()[1]
                try:
                    if self.scheduler.is_entered(unit_id):
                        self.scheduler.done(unit_id)
                except BaseException:
                    if original is None:
                        raise

            self._hooks.append(module.register_forward_pre_hook(before))
            self._hooks.append(module.register_forward_hook(after, always_call=True))

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        try:
            return super().__getattr__(name)
        except AttributeError as error:
            model = super().__getattr__("model")
            if self._closed:
                raise RuntimeError("streaming model is closed") from error
            return getattr(model, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Forward to the wrapped model while streaming hooks are active."""
        if self._closed:
            raise RuntimeError("streaming model is closed")
        return self.model(*args, **kwargs)

    def close(self) -> None:
        """Remove delivery hooks, stop scheduling, release its inputs, and replace bound storage with meta tensors."""
        if self._closed:
            return
        self._closed = True
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        try:
            self.scheduler.close()
        finally:
            self._stack.close()
            for module in self.model.modules():
                for name, value in tuple(module._parameters.items()):
                    if value is not None:
                        module._parameters[name] = torch.nn.Parameter(
                            torch.empty_like(value, device="meta"), requires_grad=False
                        )
                for name, value in tuple(module._buffers.items()):
                    if value is not None:
                        module._buffers[name] = torch.empty_like(value, device="meta")
