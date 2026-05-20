from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch.nn as nn

from .datamodel import CompiledPlan


@dataclass(frozen=True)
class DispatchChoice:
    name: str | None = None
    use_eager: bool = False

    @classmethod
    def eager(cls) -> "DispatchChoice":
        return cls(use_eager=True)

    @classmethod
    def plan(cls, name: str) -> "DispatchChoice":
        return cls(name=name)


class DispatchPolicy(Protocol):
    def select(
        self,
        plans: Sequence[CompiledPlan],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> CompiledPlan | DispatchChoice | str | int | None: ...


class EagerDispatchPolicy:
    def select(
        self,
        plans: Sequence[CompiledPlan],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        return None


class FirstPlanDispatchPolicy:
    def select(
        self,
        plans: Sequence[CompiledPlan],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> CompiledPlan | None:
        return plans[0] if plans else None


class PlanDispatcher(nn.Module):
    def __init__(
        self,
        eager_module: Callable[..., Any],
        plans: Sequence[CompiledPlan] = (),
        dispatch_policy: DispatchPolicy | Callable[..., Any] | None = None,
    ) -> None:
        super().__init__()
        self.eager_module = eager_module
        self.plans = tuple(plans)
        self._plan_modules = nn.ModuleList(
            [plan.module for plan in self.plans if isinstance(plan.module, nn.Module)]
        )
        self.dispatch_policy = dispatch_policy or EagerDispatchPolicy()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        plan = self._select_plan(args, kwargs)
        if plan is None:
            return self.eager_module(*args, **kwargs)
        if plan.module is None:
            raise RuntimeError(f"Compiled plan {plan.name!r} has no callable module")
        return plan.module(*args, **kwargs)

    def _select_plan(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CompiledPlan | None:
        selector = getattr(self.dispatch_policy, "select", self.dispatch_policy)
        choice = selector(self.plans, args, kwargs)
        return self._resolve_choice(choice)

    def _resolve_choice(self, choice: Any) -> CompiledPlan | None:
        if choice is None:
            return None
        if isinstance(choice, CompiledPlan):
            return choice
        if isinstance(choice, DispatchChoice):
            if choice.use_eager:
                return None
            if choice.name is None:
                raise ValueError("DispatchChoice.name is required unless use_eager=True")
            return self._plan_by_name(choice.name)
        if isinstance(choice, str):
            if choice == "eager":
                return None
            return self._plan_by_name(choice)
        if isinstance(choice, int):
            return self.plans[choice]
        raise TypeError(f"Unsupported dispatch choice {choice!r}")

    def _plan_by_name(self, name: str) -> CompiledPlan:
        for plan in self.plans:
            if plan.name == name:
                return plan
        raise KeyError(f"No compiled plan named {name!r}")
