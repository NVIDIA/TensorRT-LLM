# autoflake: skip_file
import inspect
import os
import pathlib
from dataclasses import dataclass, fields
from types import MethodType, NoneType
from typing import (Any, Callable, Dict, List, Literal, Optional, Sequence,
                    Tuple, Union, _type_repr)

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


@dataclass(slots=True)
class ParamSnapshot:
    name: str
    annotation: type
    default: Any = None

    @classmethod
    def from_inspect(cls, param: inspect.Parameter):
        return cls(param.name, param.annotation, param.default)

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

        return cls(param.arg_name, annotation, default)

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
        assert self.name == other.name
        assert self.annotation == other.annotation
        assert self.default == other.default


@dataclass(slots=True)
class MethodSnapshot:
    name: str
    parameters: Dict[str, ParamSnapshot]
    return_annotation: type

    @classmethod
    def from_inspect(cls, name: str, method: MethodType):
        signature = inspect.signature(method)
        parameters = {}
        for param_name, param in signature.parameters.items():
            if param_name.startswith("_"):
                continue
            parameters[param_name] = ParamSnapshot.from_inspect(param)
        return_annotation = signature.return_annotation
        if isinstance(return_annotation, str):
            return_annotation = eval(return_annotation)
        return cls(name, parameters, return_annotation)

    @classmethod
    def from_docstring(cls, name: str, method: MethodType):
        doc = docstring_parser.parse(method.__doc__)
        parameters = {}
        for param in doc.params:
            if param.args[0] == 'param':
                parameters[param.arg_name] = ParamSnapshot.from_docstring(param)
        if doc.returns is None:
            return_annotation = None
        else:
            return_annotation = eval(doc.returns.type_name)
        return cls(name, parameters, return_annotation)

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

    def assert_equal(self, other: 'MethodSnapshot'):
        assert self.name == other.name
        assert self.parameters.keys() == other.parameters.keys()
        for name, param in self.parameters.items():
            param.assert_equal(other.parameters[name])
        assert self.return_annotation == other.return_annotation


@dataclass(slots=True)
class ClassSnapshot:
    methods: Dict[str, MethodSnapshot]
    properties: Dict[str, ParamSnapshot]

    @classmethod
    def from_inspect(cls, snapshot_cls: type):
        inst = snapshot_cls.__new__(snapshot_cls)
        methods = {}
        for method_name, method in inspect.getmembers(
                inst, predicate=inspect.ismethod):
            if method_name.startswith("_") and method_name != "__init__":
                continue
            methods[method_name] = MethodSnapshot.from_inspect(
                method_name, method)
        properties = {}
        for prop_name, prop in inspect.getmembers(
                snapshot_cls, predicate=lambda x: isinstance(x, property)):
            if prop_name.startswith("_"):
                continue
            annotation = inspect.signature(prop.fget).return_annotation
            properties[prop_name] = ParamSnapshot(prop_name, annotation,
                                                  inspect._empty)
        return cls(methods, properties)

    @classmethod
    def from_docstring(cls, snapshot_cls: type):
        inst = snapshot_cls.__new__(snapshot_cls)
        methods = {}
        for method_name, method in inspect.getmembers(
                inst, predicate=inspect.ismethod):
            if method_name.startswith("_") and method_name != "__init__":
                continue
            if method_name == "__init__":
                methods["__init__"] = MethodSnapshot.from_docstring(
                    "__init__", snapshot_cls)
            else:
                methods[method_name] = MethodSnapshot.from_docstring(
                    method_name, method)
        properties = {}
        doc = docstring_parser.parse(snapshot_cls.__doc__)
        for param in doc.params:
            if param.args[0] == 'attribute':
                properties[param.arg_name] = ParamSnapshot.from_docstring(param)
        return cls(methods, properties)

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

    def assert_equal(self, other: 'ClassSnapshot'):
        assert self.methods.keys() == other.methods.keys()
        for name, method in self.methods.items():
            method.assert_equal(other.methods[name])
        assert self.properties.keys() == other.properties.keys()
        for name, prop in self.properties.items():
            prop.assert_equal(other.properties[name])


class ApiStabilityTestHarness:
    TEST_CLASS = None
    REFERENCE_DIR = f"{os.path.dirname(__file__)}/references"
    REFERENCE_FILE = None

    @classmethod
    def reference_path(cls):
        return f"{cls.REFERENCE_DIR}/{cls.REFERENCE_FILE}"

    @classmethod
    def setup_class(cls):
        with open(cls.reference_path()) as f:
            cls.reference = ClassSnapshot.from_dict(yaml.safe_load(f))
        cls.error_msg = (
            f"API stability validation failed. "
            f"This is probably because you changed {cls.TEST_CLASS.__name__}'s APIs, please ask for reviews from the code owners."
        )

    def test_signature(self):
        snapshot = ClassSnapshot.from_inspect(self.TEST_CLASS)
        try:
            snapshot.assert_equal(self.reference)
        except AssertionError as e:
            raise AssertionError(self.error_msg) from e

    def test_docstring(self):
        snapshot = ClassSnapshot.from_docstring(self.TEST_CLASS)
        try:
            snapshot.assert_equal(self.reference)
        except AssertionError as e:
            raise AssertionError(self.error_msg) from e
