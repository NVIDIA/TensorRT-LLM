from __future__ import annotations

from agent_flow import Module, Sequential


class AppendLayer(Module):

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def forward(self, content):
        separator = "\n" if content else ""
        return f"{content}{separator}{self.text}"


class ParentModule(Module):

    def __init__(self) -> None:
        super().__init__()
        self.left = AppendLayer("left")
        self.right = AppendLayer("right")

    def forward(self, content):
        return content


def test_auto_registers_child_modules():
    module = ParentModule()

    children = module.children()

    assert len(children) == 2
    assert children[0] is module.left
    assert children[1] is module.right


def test_reassigning_non_module_unregisters_child():
    module = ParentModule()

    module.left = "not-a-module"

    assert module.children() == (module.right, )


def test_modules_recurses_in_assignment_order():
    module = ParentModule()

    modules = list(module.modules())

    assert modules[0] is module
    assert modules[1] is module.left
    assert modules[2] is module.right


def test_module_does_not_expose_manual_close_api():
    module = AppendLayer("done")

    assert hasattr(module, "__enter__")
    assert hasattr(module, "__aenter__")
    assert not hasattr(module, "close")
    assert not hasattr(module, "aclose")


def test_sequential_runs_children_in_order():
    seq = Sequential(AppendLayer("first"), AppendLayer("second"))

    result = seq("hello")

    assert result == "hello\nfirst\nsecond"
