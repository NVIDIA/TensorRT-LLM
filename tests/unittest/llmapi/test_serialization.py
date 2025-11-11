import torch

from tensorrt_llm import serialization


class TestClass:

    def __init__(self, name: str):
        self.name = name


def test_serialization_allowed_class():
    obj = TestClass("test")
    serialization.register_approved_class(TestClass)
    module = TestClass.__module__
    assert module in serialization.BASE_EXAMPLE_CLASSES
    assert "TestClass" in serialization.BASE_EXAMPLE_CLASSES[module]
    a = serialization.dumps(obj)
    b = serialization.loads(a,
                            approved_imports=serialization.BASE_EXAMPLE_CLASSES)
    assert type(obj) == type(b) and obj.name == b.name


def test_serialization_disallowed_class():
    obj = TestClass("test")
    a = serialization.dumps(obj)
    excep = None
    try:
        serialization.loads(a, approved_imports={})
    except Exception as e:
        excep = e
        print(excep)
    assert isinstance(excep, ValueError) and str(
        excep) == "Import llmapi.test_serialization | TestClass is not allowed"


def test_serialization_basic_object():
    obj = {"test": "test"}
    a = serialization.dumps(obj)
    b = serialization.loads(a,
                            approved_imports=serialization.BASE_EXAMPLE_CLASSES)
    assert obj == b


def test_serialization_complex_object_allowed_class():
    obj = torch.tensor([1, 2, 3])
    a = serialization.dumps(obj)
    b = serialization.loads(a,
                            approved_imports=serialization.BASE_EXAMPLE_CLASSES)
    assert torch.all(obj == b)


def test_serialization_complex_object_partially_allowed_class():
    obj = torch.tensor([1, 2, 3])
    a = serialization.dumps(obj)
    excep = None
    try:
        b = serialization.loads(a,
                                approved_imports={
                                    'torch._utils': ['_rebuild_tensor_v2'],
                                })
    except Exception as e:
        excep = e
    assert isinstance(excep, ValueError) and str(
        excep) == "Import torch.storage | _load_from_bytes is not allowed"


def test_serialization_complex_object_disallowed_class():
    obj = torch.tensor([1, 2, 3])
    a = serialization.dumps(obj)
    excep = None
    try:
        serialization.loads(a)
    except Exception as e:
        excep = e
    assert isinstance(excep, ValueError) and str(
        excep) == "Import torch._utils | _rebuild_tensor_v2 is not allowed"


if __name__ == "__main__":
    test_serialization_allowed_class()
