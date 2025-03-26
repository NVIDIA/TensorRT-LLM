# autoflake: skip_file
import copy
import inspect
import os
import pathlib
from dataclasses import dataclass, fields
from types import MethodType, NoneType
from typing import (Any, Callable, ClassVar, Dict, List, Literal, Optional,
                    Sequence, Tuple, Union, _type_repr)

import docstring_parser
import pydantic.main
import pytest
import torch
import transformers
import yaml

import tensorrt_llm
from tensorrt_llm.executor import GenerationResult
from tensorrt_llm.llmapi import (LLM, CalibConfig, CompletionOutput,
                                 GuidedDecodingParams, QuantConfig,
                                 RequestOutput, SamplingParams)
from tensorrt_llm.llmapi.llm_utils import LlmArgs


def repr_annotation(field_type: type) -> str:
    return _type_repr(field_type).replace("typing.", "")


class StackTrace:
    ''' Keep track of the symbol stack to the current scope. '''
    _instance: ClassVar[Optional["StackTrace"]] = None

    def __init__(self):
        self.stack: List[str] = []

    def push(self, symbol: Optional[str]):
        if symbol is None: return self
        self.stack.append(symbol)
        return self

    def pop(self):
        if self.stack:
            self.stack.pop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pop()

    def get_prefix(self) -> str:
        if not self.stack:
            return ""
        return ".".join(self.stack) + "."

    @staticmethod
    def instance() -> "StackTrace":
        if StackTrace._instance is None:
            StackTrace._instance = StackTrace()
        return StackTrace._instance


@dataclass(slots=True)
class ParamSnapshot:
    annotation: type
    default: Any = None
    name: Optional[str] = None

    @classmethod
    def from_inspect(cls, param: inspect.Parameter):
        return cls(param.annotation, param.default, name=param.name)

    @classmethod
    def from_docstring(cls, param: docstring_parser.common.DocstringParam):
        assert isinstance(param.type_name, str)
        assert isinstance(param.is_optional, bool)
        assert isinstance(param.default, (str, NoneType))
        assert isinstance(param.description, str) and len(param.description) > 1

        annotation = eval(param.type_name)
        if isinstance(annotation, tuple):
            annotation = Union[annotation]
        if param.is_optional:
            annotation = Optional[annotation]

        if param.default is None:
            default = inspect._empty
        else:
            try:
                default = eval(param.default)
            except (NameError, SyntaxError):
                default = param.default

        return cls(annotation, default, name=param.arg_name)

    @classmethod
    def from_dict(cls, d: dict):
        d['annotation'] = eval(d['annotation'])
        if isinstance(d['default'], str):
            try:
                d['default'] = eval(d['default'])
            except (NameError, SyntaxError):
                pass
        return cls(**d)

    def to_dict(self):
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d['annotation'] = repr_annotation(d['annotation'])
        if d['default'] == inspect._empty:
            d['default'] = "inspect._empty"
        return d

    def assert_equal(self, other: 'ParamSnapshot'):
        prefix = StackTrace.instance().get_prefix()
        assert self.annotation == other.annotation, f"{prefix}{self.name} annotation: {self.annotation} != {other.annotation}"
        assert self.default == other.default, f"{prefix}{self.name} default: {self.default} != {other.default}"


@dataclass(slots=True)
class MethodSnapshot:
    parameters: Dict[str, ParamSnapshot]
    return_annotation: type
    name: Optional[str] = None

    @classmethod
    def from_inspect(cls, method: MethodType):
        signature = inspect.signature(method)
        parameters = {}
        for param_name, param in signature.parameters.items():
            if param_name.startswith("_"):
                continue
            parameters[param_name] = ParamSnapshot.from_inspect(param)
        return_annotation = signature.return_annotation
        if isinstance(return_annotation, str):
            if return_annotation == "Self":
                return_annotation = 'None'
            return_annotation = eval(return_annotation)
        return cls(parameters, return_annotation, name=method.__name__)

    @classmethod
    def from_docstring(cls, method: MethodType):
        doc = docstring_parser.parse(method.__doc__)
        parameters = {}
        for param in doc.params:
            if param.args[0] == 'param':
                parameters[param.arg_name] = ParamSnapshot.from_docstring(param)
        if doc.returns is None:
            return_annotation = None
        else:
            return_annotation = eval(doc.returns.type_name)
        return cls(parameters, return_annotation, name=method.__name__)

    @classmethod
    def from_dict(cls, d: dict):
        d['parameters'] = {
            name: ParamSnapshot.from_dict(param)
            for name, param in d['parameters'].items()
        }
        d['return_annotation'] = eval(d['return_annotation'])
        return cls(**d)

    def to_dict(self):
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d['parameters'] = {
            name: param.to_dict()
            for name, param in d['parameters'].items()
        }
        d['return_annotation'] = repr_annotation(d['return_annotation'])
        return d

    def merge(self, other: 'MethodSnapshot'):
        overlapped_keys = set(self.parameters.keys()) & set(
            other.parameters.keys())
        assert not overlapped_keys, f"Overlapped parameters: {overlapped_keys}"
        self.parameters.update(copy.deepcopy(other.parameters))
        assert self.return_annotation == other.return_annotation

    def assert_equal(self, other: 'MethodSnapshot'):
        prefix = StackTrace.instance().get_prefix()

        with StackTrace.instance().push(self.name):
            self_only = set(self.parameters.keys()) - set(
                other.parameters.keys())
            other_only = set(other.parameters.keys()) - set(
                self.parameters.keys())

            self_only = list(
                filter(
                    lambda x: not x.startswith("_") and x not in
                    PYDANTIC_FIELDS, self_only))
            other_only = list(
                filter(
                    lambda x: not x.startswith("_") and x not in
                    PYDANTIC_FIELDS, other_only))

            if self_only or other_only:
                raise AssertionError(
                    f"{prefix}{self.name} has different parameters: "
                    f"adding {self_only}, removing {other_only}")
            else:
                for name, param in self.parameters.items():
                    param.assert_equal(other.parameters[name])
            assert self.return_annotation == other.return_annotation

    def assert_containing(self, other: 'MethodSnapshot'):
        prefix = StackTrace.instance().get_prefix()
        if prefix + self.name == "LLM.__init__":
            return  # LLM.__init__'s arglist is just a subset of the reference which is from LlmArgs

        with StackTrace.instance().push(self.name):
            for name, param in other.parameters.items():
                assert name in self.parameters, (
                    f"{prefix}{self.name} missing parameter '{name}' from reference.\n"
                    f"{prefix}{self.name}'s parameter list is {self.parameters.keys()}"
                )
                self.parameters[name].assert_equal(param)
        assert self.return_annotation == other.return_annotation


PYDANTIC_FIELDS = {
    'model_json_schema',
    'from_orm',
    'json',
    'model_rebuild',
    'construct',
    'model_validate_strings',
    'model_copy',
    'parse_file',
    'validate',
    'model_dump',
    'model_parametrized_name',
    'schema_json',
    'parse_raw',
    'parse_obj',
    'update_forward_refs',
    'model_post_init',
    'model_validate',
    'schema',
    'copy',
    'model_validate_json',
    'dict',
    'model_construct',
    'model_dump_json',
    'model_extra',
    'model_fields_set',
    'model_fields',
}


@dataclass(slots=True)
class ClassSnapshot:
    methods: Dict[str, MethodSnapshot]
    properties: Dict[str, ParamSnapshot]
    name: Optional[str] = None

    @classmethod
    def from_inspect(cls, snapshot_cls: type):
        inst = snapshot_cls.__new__(snapshot_cls)
        name = snapshot_cls.__name__
        methods = {}
        for method_name, method in inspect.getmembers(
                inst, predicate=inspect.ismethod):
            if method_name.startswith("_") and method_name != "__init__":
                continue
            if method_name in PYDANTIC_FIELDS:
                continue
            # deal with pydantic __init__
            if method_name == "__init__" and isinstance(
                    snapshot_cls, type) and issubclass(snapshot_cls,
                                                       pydantic.main.BaseModel):
                # Create a MethodSnapshot for Pydantic model's __init__
                parameters = {}
                for field_name, field in snapshot_cls.model_fields.items():
                    if field_name.startswith("_"):
                        continue
                    parameters[field_name] = ParamSnapshot(
                        annotation=field.annotation,
                        default=field.default or inspect._empty)
                methods[method_name] = MethodSnapshot(parameters=parameters,
                                                      return_annotation=None,
                                                      name="__init__")
            else:
                methods[method_name] = MethodSnapshot.from_inspect(method)
        properties = {}
        for prop_name, prop in inspect.getmembers(
                snapshot_cls, predicate=lambda x: isinstance(x, property)):
            if prop_name.startswith("_"):
                continue
            if prop_name in PYDANTIC_FIELDS:
                continue
            annotation = inspect.signature(prop.fget).return_annotation
            properties[prop_name] = ParamSnapshot(annotation, inspect._empty)
        return cls(methods, properties, name=name)

    @classmethod
    def from_docstring(cls, snapshot_cls: type):
        inst = snapshot_cls.__new__(snapshot_cls)
        methods = {}
        for method_name, method in inspect.getmembers(
                inst, predicate=inspect.ismethod):
            if method_name.startswith("_") and method_name != "__init__":
                continue
            if method_name in PYDANTIC_FIELDS:  # ignore Pydantic methods
                continue
            if method_name == "__init__":
                if isinstance(snapshot_cls, type) and issubclass(
                        snapshot_cls, pydantic.main.BaseModel):
                    parameters = {}
                    for field_name, field in snapshot_cls.model_fields.items():
                        if field_name.startswith("_"):
                            continue
                        parameters[field_name] = ParamSnapshot(
                            annotation=field.annotation,
                            default=field.default or inspect._empty)
                    methods["__init__"] = MethodSnapshot(parameters=parameters,
                                                         return_annotation=None,
                                                         name="__init__")
                else:
                    methods["__init__"] = MethodSnapshot.from_docstring(
                        snapshot_cls)
            else:
                methods[method_name] = MethodSnapshot.from_docstring(method)

        properties = {}
        doc = docstring_parser.parse(snapshot_cls.__doc__)
        for param in doc.params:
            if param.args[0] == 'attribute':
                properties[param.arg_name] = ParamSnapshot.from_docstring(param)
        return cls(methods, properties, name=snapshot_cls.__name__)

    @classmethod
    def from_dict(cls, d: dict):
        d['methods'] = {
            name: MethodSnapshot.from_dict(method)
            for name, method in d['methods'].items()
        }
        d['properties'] = {
            name: ParamSnapshot.from_dict(prop)
            for name, prop in d['properties'].items()
        }
        return cls(**d)

    def to_dict(self):
        d = {}
        d['methods'] = {
            name: method.to_dict()
            for name, method in self.methods.items()
        }
        d['properties'] = {
            name: prop.to_dict()
            for name, prop in self.properties.items()
        }
        return d

    def merge(self, other: 'ClassSnapshot'):
        for name, method in self.methods.items():
            if name in other.methods:
                method.merge(other.methods[name])
        new_methods = {
            name: method
            for name, method in other.methods.items()
            if name not in self.methods
        }
        self.methods.update(copy.deepcopy(new_methods))
        assert self.properties.keys().isdisjoint(other.properties.keys())
        self.properties.update(copy.deepcopy(other.properties))

    def assert_equal(self, other: 'ClassSnapshot'):
        prefix = StackTrace.instance().get_prefix()
        with StackTrace.instance().push(self.name):
            if self.methods.keys() != other.methods.keys():
                diff_keys = set(self.methods.keys()) ^ set(other.methods.keys())
                raise AssertionError(
                    f"{prefix}{self.name} has different methods: {diff_keys}")

            for name, method in self.methods.items():
                if self.name == "LLM" and name == "__init__":
                    # only check the explicit the explicit arglist from LLM.__init__
                    for param in method.parameters.values():
                        if param.name not in other.methods[name].parameters:
                            raise AssertionError(
                                f"{prefix}{self.name} doesn't have a parameter '{param.name}' in reference.\n"
                                f"The reference parameter list is {other.methods[name].parameters.keys()}"
                            )
                        param.assert_equal(
                            other.methods[name].parameters[param.name])
                else:
                    method.assert_equal(other.methods[name])

            if self.properties.keys() != other.properties.keys():
                diff_keys = set(self.properties.keys()) ^ set(
                    other.properties.keys())
                raise AssertionError(
                    f"{prefix}{self.name} has different properties: {diff_keys}"
                )

            for name, prop in self.properties.items():
                prop.assert_equal(other.properties[name])

    def assert_containing(self, other: 'ClassSnapshot'):
        with StackTrace.instance().push(self.name):
            for name, method in other.methods.items():
                assert name in self.methods
                self.methods[name].assert_containing(method)
            for name, prop in other.properties.items():
                assert name in self.properties
                self.properties[name].assert_equal(prop)


class ApiStabilityTestHarness:
    TEST_CLASS = None
    REFERENCE_COMMITTED_DIR = f"{os.path.dirname(__file__)}/references_committed"
    REFERENCE_DIR = f"{os.path.dirname(__file__)}/references"
    REFERENCE_FILE = None

    @classmethod
    def setup_class(cls):
        with open(f"{cls.REFERENCE_DIR}/{cls.REFERENCE_FILE}") as f:
            cls.reference = ClassSnapshot.from_dict(yaml.safe_load(f))
        if os.path.exists(
                f"{cls.REFERENCE_COMMITTED_DIR}/{cls.REFERENCE_FILE}"):
            with open(
                    f"{cls.REFERENCE_COMMITTED_DIR}/{cls.REFERENCE_FILE}") as f:
                cls.reference_committed = ClassSnapshot.from_dict(
                    yaml.safe_load(f))
            cls.reference.merge(cls.reference_committed)
        else:
            cls.reference_committed = None
        cls.error_msg = f"API validation failed because you changed {cls.TEST_CLASS.__name__}'s APIs, please ask for reviews from the code owners."
        cls.error_msg_committed = f"API validation failed because you changed {cls.TEST_CLASS.__name__}'s committed APIs, please ask for approval."

    def create_snapshot_from_inspect(self):
        return ClassSnapshot.from_inspect(self.TEST_CLASS)

    def test_signature(self):
        snapshot = self.create_snapshot_from_inspect()
        if self.reference_committed is not None:
            try:
                snapshot.assert_containing(self.reference_committed)
            except AssertionError as e:
                raise AssertionError(self.error_msg_committed) from e
        try:
            snapshot.assert_equal(self.reference)
        except AssertionError as e:
            raise AssertionError(self.error_msg) from e

    def create_snapshot_from_docstring(self):
        return ClassSnapshot.from_docstring(self.TEST_CLASS)

    def test_docstring(self):
        snapshot = self.create_snapshot_from_docstring()
        if self.reference_committed is not None:
            try:
                snapshot.assert_containing(self.reference_committed)
            except AssertionError as e:
                raise AssertionError(self.error_msg_committed) from e
        try:
            snapshot.assert_equal(self.reference)
        except AssertionError as e:
            raise AssertionError(self.error_msg) from e
