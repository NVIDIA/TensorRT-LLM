from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from threading import Thread

import anyio


def _run_async(coro_func, *args):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return anyio.run(coro_func, *args)

    result: list[object] = []
    error: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(anyio.run(coro_func, *args))
        except BaseException as exc:
            error.append(exc)

    thread = Thread(target=runner)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0] if result else None


class Module(ABC):

    def __init__(self) -> None:
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_closed", False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._close_tree()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._aclose_tree()

    def __del__(self) -> None:
        self._close_tree(suppress_errors=True)

    def __setattr__(self, name: str, value) -> None:
        if name in {"_modules", "_closed"}:
            object.__setattr__(self, name, value)
            return

        modules = self.__dict__.get("_modules")
        if modules is not None:
            if isinstance(value, Module):
                modules[name] = value
            else:
                modules.pop(name, None)

        object.__setattr__(self, name, value)

    @abstractmethod
    def forward(self, content: str) -> str:
        raise NotImplementedError

    async def aforward(self, content: str) -> str:
        return self.forward(content)

    def __call__(self, content: str) -> str:
        return self.forward(content)

    def children(self) -> tuple[Module, ...]:
        return tuple(self.__dict__.get("_modules", {}).values())

    def modules(self):
        yield self
        for child in self.children():
            yield from child.modules()

    async def _aclose_tree(self) -> None:
        if self.__dict__.get("_closed", False):
            return

        for child in self.children():
            await child._aclose_tree()
        await self._aclose_self()
        object.__setattr__(self, "_closed", True)

    def _close_tree(self, *, suppress_errors: bool = False) -> None:
        if self.__dict__.get("_closed", False):
            return

        try:
            _run_async(self._aclose_tree)
        except Exception:
            if not suppress_errors:
                raise

    async def _aclose_self(self) -> None:
        return None


class Sequential(Module):

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        for index, module in enumerate(modules):
            setattr(self, f"layer_{index}", module)

    def forward(self, content: str) -> str:
        return anyio.run(self.aforward, content)

    async def aforward(self, content: str) -> str:
        current = content
        for module in self.children():
            current = await module.aforward(current)
        return current
