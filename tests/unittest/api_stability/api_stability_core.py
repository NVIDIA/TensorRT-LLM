# autoflake: skip_file
import copy
import inspect
import os
import pathlib
from dataclasses import _HAS_DEFAULT_FACTORY_CLASS, dataclass, fields
from pprint import pprint
from types import MethodType, NoneType
from typing import (Any, Callable, ClassVar, Dict, List, Literal, Optional,
                    Sequence, Tuple, Union, _type_repr)

import docstring_parser
import pydantic.main
import pytest
import torch
import transformers
import yaml
from pydantic import BaseModel

import tensorrt_llm
from tensorrt_llm import LLM
# Import BaseCheckpointLoader for YAML processing
from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import \
    BaseCheckpointLoader
from tensorrt_llm.executor import GenerationResult
from tensorrt_llm.executor.result import TokenLogprobs
from tensorrt_llm.llmapi import (CalibConfig, CompletionOutput,
                                 GuidedDecodingParams, QuantConfig,
                                 RequestOutput, SamplingParams)
from tensorrt_llm.llmapi.llm_args import SamplerType
from tensorrt_llm.llmapi.llm_utils import LlmArgs
from tensorrt_llm.logger import Singleton


def repr_annotation(field_type: type) -> str:
    return _type_repr(field_type).replace("typing.", "")


class StackTrace(metaclass=Singleton):
    ''' Keep track of the symbol stack to the current scope. '''

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

    def get_name(self) -> str:
        if not self.stack:
            return ""
        return self.stack[-1]

    def get_qual_name(self) -> str:
        if not self.stack:
            return ""
        return ".".join(self.stack)


@dataclass(slots=True)
class ParamSnapshot:
    annotation: type
    default: Any = None
    status: Optional[str] = None

    def __post_init__(self):
        # Unify default value of None and inspect._empty
        if self.default is inspect._empty:
            self.default = None

    @classmethod
    def from_inspect(cls, param: inspect.Parameter):
        return cls(param.annotation, param.default)

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

        return cls(annotation, default)

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
        qual_name = StackTrace().get_qual_name()
        assert self.annotation == other.annotation, f"{qual_name} annotation: {self.annotation} != {other.annotation}"
        if not isinstance(self.default, _HAS_DEFAULT_FACTORY_CLASS):
            assert self.default == other.default, f"{qual_name} default: {self.default} != {other.default}"


@dataclass(slots=True)
class MethodSnapshot:
    parameters: Dict[str, ParamSnapshot]
    return_annotation: type
    status: Optional[str] = None

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
            return_annotation = eval(return_annotation)
        return cls(parameters, return_annotation)

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
        return cls(parameters, return_annotation)

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
        qual_name = StackTrace().get_qual_name()

        self_only = set(self.parameters.keys()) - set(other.parameters.keys())
        other_only = set(other.parameters.keys()) - set(self.parameters.keys())

        self_only = list(
            filter(lambda x: not x.startswith("_") and x not in PYDANTIC_FIELDS,
                   self_only))
        other_only = list(
            filter(lambda x: not x.startswith("_") and x not in PYDANTIC_FIELDS,
                   other_only))

        if self_only or other_only:
            raise AssertionError(f"{qual_name} has different parameters: "
                                 f"adding {self_only}, removing {other_only}")
        else:
            for name, param in self.parameters.items():
                with StackTrace().push(name):
                    param.assert_equal(other.parameters[name])
        assert self.return_annotation == other.return_annotation

    def assert_containing(self, other: 'MethodSnapshot'):
        qual_name = StackTrace().get_qual_name()
        if qual_name == "LLM.__init__":
            return  # LLM.__init__'s arglist is just a subset of the reference which is from LlmArgs

        for name, param in other.parameters.items():
            assert name in self.parameters, (
                f"{qual_name} missing parameter '{name}' from reference.\n"
                f"{qual_name}'s parameter list is {self.parameters.keys()}")
            with StackTrace().push(name):
                self.parameters[name].assert_equal(param)
        assert self.return_annotation == other.return_annotation


class _DummyModel(BaseModel):
    pass


# get all members of the Pydantic model
PYDANTIC_FIELDS = set(dir(_DummyModel)) - {"__init__"}


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
            if method_name in PYDANTIC_FIELDS:
                continue
            # deal with pydantic __init__
            if method_name == "__init__" and isinstance(
                    snapshot_cls, type) and issubclass(snapshot_cls,
                                                       pydantic.main.BaseModel):
                # Create a MethodSnapshot for Pydantic model's __init__,
                # the parameters are the fields of the model
                parameters = {}
                for field_name, field in snapshot_cls.model_fields.items():
                    if field_name.startswith("_"):
                        continue
                    parameters[field_name] = ParamSnapshot(
                        annotation=field.annotation,
                        default=field.default or inspect._empty)
                methods[method_name] = MethodSnapshot(parameters=parameters,
                                                      return_annotation=None)
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
        return cls(methods, properties)

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
                                                         return_annotation=None)
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
        qual_name = StackTrace().get_qual_name()
        if self.methods.keys() != other.methods.keys():
            diff_keys = set(self.methods.keys()) ^ set(other.methods.keys())
            raise AssertionError(
                f"{qual_name} has different methods: {diff_keys}")

        for name, method in self.methods.items():
            # LLM.__init__'s arglist is just a subset of the reference which is from LlmArgs, thus we need to
            # handle it separately
            if qual_name == "LLM" and name == "__init__":
                # only check the explicit the explicit arglist from LLM.__init__
                for param_name, param in method.parameters.items():
                    if param_name not in other.methods[name].parameters:
                        raise AssertionError(
                            f"{qual_name} doesn't have a parameter '{param_name}' in reference.\n"
                            f"The reference parameter list is {other.methods[name].parameters.keys()}"
                        )
                    with StackTrace().push(param_name):
                        param.assert_equal(
                            other.methods[name].parameters[param_name])
            else:
                with StackTrace().push(name):
                    method.assert_equal(other.methods[name])

        if self.properties.keys() != other.properties.keys():
            diff_keys = set(self.properties.keys()) ^ set(
                other.properties.keys())
            this_diff_keys = set(self.properties.keys()) - set(
                other.properties.keys())
            other_diff_keys = set(other.properties.keys()) - set(
                self.properties.keys())
            raise AssertionError(
                f"{qual_name} has different properties: {diff_keys}\n"
                f"This class has extra properties: {this_diff_keys}\n"
                f"The reference has extra properties: {other_diff_keys}")

        for name, prop in self.properties.items():
            with StackTrace().push(name):
                prop.assert_equal(other.properties[name])

    def assert_containing(self, other: 'ClassSnapshot'):
        for name, method in other.methods.items():
            with StackTrace().push(name):
                assert name in self.methods
                self.methods[name].assert_containing(method)
        for name, prop in other.properties.items():
            with StackTrace().push(name):
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
            cls.non_committed_reference = copy.deepcopy(cls.reference)
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

    def create_snapshot_from_docstring(self):
        return ClassSnapshot.from_docstring(self.TEST_CLASS)

    def test_signature(self):
        with StackTrace().push(self.TEST_CLASS.__name__):
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

    def test_docstring(self):
        with StackTrace().push(self.TEST_CLASS.__name__):
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

    def test_api_status(self):
        """ Check that the API status (prototype | beta) matches the llm.yaml.
        Note that, only the non-committed APIs are checked, the committed APIs
        are treated as stable.
        """

        # Only check the API status for llm.yaml
        if self.REFERENCE_FILE != "llm.yaml":
            return

        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

        actual_fields = TorchLlmArgs.model_fields
        reference_data = self.non_committed_reference.to_dict()
        committed_data = self.reference_committed.to_dict()

        def get_actual_status(field_name):
            if field_name in actual_fields:
                field = actual_fields[field_name]
                return field.json_schema_extra.get(
                    'status') if field.json_schema_extra else None
            return None

        def check_status(field_name, reference_status, context=""):
            # Deprecated fields are not checked
            if reference_status == "deprecated":
                return

            actual_status = get_actual_status(field_name)
            if actual_status is None:
                raise AssertionError(
                    f"context: {self.TEST_CLASS} {context}\n"
                    f"Status is not set for the non-committed '{field_name}', "
                    "please update the field with Field(..., status='<status>') in llm_args.py, "
                    "status could be either 'beta' or 'prototype'.")

            if reference_status is None:
                raise AssertionError(
                    f"context: {self.TEST_CLASS} {context}\n"
                    f"Status is not set for '{field_name}' in reference/llm.yaml, "
                    "please update the field with `status: <status>`, "
                    "status could be either 'beta' or 'prototype'.")

            if actual_status != reference_status:
                raise AssertionError(
                    f"Status mismatch for '{field_name}': "
                    f"actual='{actual_status}', reference='{reference_status}'")

        from tensorrt_llm.llmapi.utils import get_api_status

        # Check non-committed methods and properties
        for method_name, method_data in reference_data.get('methods',
                                                           {}).items():

            # step 1: check the method status
            method = getattr(self.TEST_CLASS, method_name)
            if method_name in committed_data.get('methods', {}):
                if method_name != "__init__":
                    continue
                # Both committed and non-committed methods have __init__ with different parameters
            if method_name != "__init__":
                method_status = get_api_status(method)
                if method_status is None:
                    raise AssertionError(
                        f"Status is not set for the non-committed {method_name}, "
                        "please update the method with @set_api_status(<status>), "
                        "status could be either 'beta' or 'prototype'.")
                if method_status != method_data.get('status'):
                    raise AssertionError(
                        f"Status mismatch for {method_name}: "
                        f"actual='{method_status}', reference='{method_data.get('status')}'"
                    )

            # step 2: check the method parameters
            # Only check the LLM.__init__'s parameters, for other methods, just check the method status
            # TODO[Superjomn]: support other methods
            if method_name == "__init__":
                for param_name, param_data in method_data.get('parameters',
                                                              {}).items():
                    print(f"param_name: {param_name}, param_data: {param_data}")
                    check_status(
                        param_name, param_data.get('status'),
                        f"parameter '{param_name}' in method '{method_name}': ")
