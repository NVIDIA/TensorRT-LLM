# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import operator

from ._common import default_net
from .logger import logger


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class Module(object):
    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self._network_outputs = {}

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        current_net = default_net()
        if not current_net._module_call_stack.module_names_set():
            logger.debug("Initializing top level module")
            current_net._module_call_stack.set_module_names(self)
        unique_name = current_net._module_call_stack.get_mod_name(self)
        with current_net._module_call_stack.call_stack_mgr() as stack:
            stack.append(unique_name)
            start_layer_idx = current_net.trt_network.num_layers
            output = self.forward(*args, **kwargs)
            end_layer_idx = current_net.trt_network.num_layers
            current_net._module_call_stack.set_layer_range(
                self, range(start_layer_idx, end_layer_idx)
            )
            return output

    def __getattr__(self, name):
        parameters = self.__dict__.get("_parameters")
        if parameters is not None and name in parameters:
            return parameters[name]

        modules = self.__dict__.get("_modules")
        if modules is not None and name in modules:
            return modules[name]

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __setattr__(self, name, value) -> None:
        from .parameter import Parameter

        # Improved module setattr to handle one edge case:
        # attribute could be first set to None and later reset to Parameter / Module class

        try:
            super().__getattribute__(name)

        except AttributeError:
            # if base class doesn't have the attribute, no matter we init or reset:
            # - keep Parameter and Module attrs in this Module class
            # - leave all other attrs in base class
            if isinstance(value, Parameter):
                self.__dict__.get("_parameters")[name] = value
            elif isinstance(value, Module):
                self.__dict__.get("_modules")[name] = value
            else:
                super().__setattr__(name, value)

        else:
            # if base class has the attribute, reset as follows:
            # - when reset as Parameter or Module attr, remove from base & add to this Module class
            # - other types reset and remain in base class
            if isinstance(value, Parameter):
                super().__delattr__(name)
                self.__dict__.get("_parameters")[name] = value
            elif isinstance(value, Module):
                super().__delattr__(name)
                self.__dict__.get("_modules")[name] = value
            else:
                super().__setattr__(name, value)

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m

    def named_modules_with_parent(self, memo=None, prefix="", parent=None, remove_duplicate=True):
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self, parent

            if parent:
                # Use the up-to-date module from the parent, to allow replacing
                # layers while iterating this generator.
                module_name = prefix.rsplit(".", 1)[-1]
                module = getattr(parent, module_name)
                if module is None:
                    return
            else:
                module = self

            for child_name, child_module in module._modules.items():
                if child_module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + child_name
                for m in child_module.named_modules_with_parent(
                    memo, submodule_prefix, module, remove_duplicate
                ):
                    yield m

    def named_children(self):
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def _named_members(self, get_members_fn, prefix="", recurse=True):
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v

    def parameters(self, recurse=True):
        for name, param in self.named_parameters():
            yield param

    def named_parameters(self, prefix="", recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items(), prefix=prefix, recurse=recurse
        )
        for elem in gen:
            yield elem

    def children(self):
        for _, module in self.named_children():
            yield module

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def _get_name(self):
        return self.__class__.__name__

    def register_parameter(self, name, param):
        if param is None:
            self._parameters[name] = None
        else:
            self._parameters[name] = param

    def register_network_output(self, name, value):
        self._network_outputs[name] = value

    def named_network_outputs(self):
        for name, module in self.named_modules():
            for n, output in module._network_outputs.items():
                yield name + ("." if name else "") + n, output

    def update_parameters(self, torch_module):
        m = {k: v for k, v in self.named_parameters()}
        tm = {k: v for k, v in torch_module.named_parameters()}

        assert sorted(m.keys()) == sorted(tm.keys()), (
            "The parameter names of the TensorRT LLM module must be the same with the torch module"
        )

        for k, v in self.named_parameters():
            v.value = tm[k].detach().cpu().numpy()

    def _get_name(self):
        return self.__class__.__name__

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        main_str = self._get_name() + "("
        if child_lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(child_lines) + "\n"
        main_str += ")"
        return main_str


class ModuleList(Module):
    def __init__(self, modules) -> None:
        super(ModuleList, self).__init__()
        offset = len(self)
        for i, module in enumerate(modules):
            self._modules[str(offset + i)] = module

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules."""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx, module) -> None:
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __len__(self):
        return len(self._modules)

    def __repr__(self):
        """Return a custom repr for ModuleList that compresses repeated module representations."""
        list_of_reprs = [repr(item) for item in self]
        if len(list_of_reprs) == 0:
            return self._get_name() + "()"

        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1
                continue

            start_end_indices.append([i, i])
            repeated_blocks.append(r)

        lines = []
        main_str = self._get_name() + "("
        for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
            local_repr = f"({start_id}): {b}"  # default repr

            if start_id != end_id:
                n = end_id - start_id + 1
                local_repr = f"({start_id}-{end_id}): {n} x {b}"

            local_repr = _addindent(local_repr, 2)
            lines.append(local_repr)

        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str
