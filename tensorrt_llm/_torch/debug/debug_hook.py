import enum
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn

import tensorrt_llm


class DumpStyle(enum.IntEnum):
    BINARY = 1
    TEXT = 2


class DebugFeatures(enum.IntEnum):
    DUMPTENSOR = 1


# A context container which contains the running states, such as the layer structures,
# log folder, hooks to run, is pre_forward or after forward, etc.
class DebuggerContext:

    def __init__(self, dest_folder=None):
        self.pre_forward_actions = {}
        self.after_forward_actions = {}

        self.layer_names = []
        self.layer_inner_counter = []

        self.module_forward_hook_handle = None
        self.module_forward_pre_hook_handle = None

        self.forward_hook_handles = {}  # module to handlers
        self.forward_pre_hook_handles = {}
        self.log_folder = dest_folder
        self.is_forward_pre = True
        self.dump_style: DumpStyle = DumpStyle.BINARY
        self._init_log_folder()

    def set_log_folder(self, log_folder):
        self.log_folder = log_folder
        self._init_log_folder()

    def _init_log_folder(self):
        if self.log_folder is None:
            import os
            pwd = os.getcwd()
            self.log_folder = os.path.join(pwd, "data_dump")

        rank = tensorrt_llm.mpi_rank()

        from pathlib import Path
        p = Path(self.log_folder) / f"rank{rank}"
        self.log_folder = p.absolute()
        p.mkdir(parents=True, exist_ok=True)

    def get_log_folder(self):
        return self.log_folder

    def check_in_pre_forward(self):
        return self.is_forward_pre

    def mark_in_pre_forward(self, is_pre_forward):
        self.is_forward_pre = is_pre_forward

    def clear_state(self):
        self.pre_forward_actions.clear()
        self.after_forward_actions.clear()
        self.layer_names.clear()
        self.layer_inner_counter.clear()

        if self.module_forward_hook_handle is not None:
            self.module_forward_hook_handle.remove()
        if self.module_forward_pre_hook_handle is not None:
            self.module_forward_pre_hook_handle.remove()

        self.module_forward_hook_handle = None
        self.module_forward_pre_hook_handle = None

        for _, handler in self.forward_hook_handles.items():
            handler.remove()

        for _, handler in self.forward_pre_hook_handles.items():
            handler.remove()

        self.forward_hook_handles.clear()
        self.forward_pre_hook_handles.clear()

    def register_pre_forward_action(self, filter, action):
        self.pre_forward_actions[filter] = action

    def get_pre_forward_action(self):
        return self.pre_forward_actions

    def register_after_forward_action(self, filter, action):
        self.after_forward_actions[filter] = action

    def get_after_forward_action(self):
        return self.after_forward_actions

    def get_current_modules_tree(self):
        return self.layer_names

    def get_module_indices_tree(self):
        return self.layer_inner_counter

    def get_current_model_loop_index(self):
        return self.layer_inner_counter[0] + 1 if len(
            self.layer_inner_counter) >= 1 else 0

    def do_actions(self, module, tensors, actions):
        for k, a in actions.items():
            if k.filter(module, tensors):
                a(module, tensors, self)


class Filter:

    def __init__(self):
        pass

    def filter(self, module: nn.Module, debug_ctx: DebuggerContext):
        raise NotImplementedError("Need to implement filter interface.")


debug_ctx = None


def get_current_debug_ctx():
    global debug_ctx
    return debug_ctx


def set_current_debug_ctx(ctx):
    global debug_ctx
    debug_ctx = ctx


# The hook is registered to module with torch.nn.modules.module.register_module_forward_pre_hook.
# This hook will be executed before module's forward is called.
# It will record module tree into debugCtx and call debugCtx's do_actions function
# to execute all hooks registered to debugCtx on current module.
def module_pre_forward(module: nn.Module, input: torch.Tensor):
    debug_ctx = get_current_debug_ctx()
    debug_ctx.mark_in_pre_forward(True)
    name = module.name if hasattr(module, "name") else module.__class__.__name__
    debug_ctx.get_current_modules_tree().append(name)
    if len(debug_ctx.get_module_indices_tree()) == 0:
        debug_ctx.get_module_indices_tree().append(0)

    if len(debug_ctx.get_current_modules_tree()) >= len(
            debug_ctx.get_module_indices_tree()):
        debug_ctx.get_module_indices_tree().append(0)

    debug_ctx.get_module_indices_tree()[
        len(debug_ctx.get_current_modules_tree()) -
        1] = debug_ctx.get_module_indices_tree()[
            len(debug_ctx.get_current_modules_tree()) - 1] + 1
    debug_ctx.do_actions(module, input, debug_ctx.get_pre_forward_action())
    return input


# The hook is registered to module with torch.nn.modules.module.register_module_forward_hook.
# This hook will be executed before module's forward is called.
# It will record module tree into debugCtx and call debugCtx's do_actions function
# to execute all hooks registered to debugCtx on current module.
def module_after_forward(module: nn.Module, input: torch.Tensor,
                         output: torch.Tensor):
    debug_ctx = get_current_debug_ctx()
    debug_ctx.mark_in_pre_forward(False)
    debug_ctx.do_actions(module, [input, output],
                         debug_ctx.get_after_forward_action())
    debug_ctx.get_current_modules_tree().pop(-1)
    debug_ctx.get_module_indices_tree().pop(-1)
    return output


# The function style to interface to enable debugger on all classes inherit from nn.Module.
# Example:
#       from tensorrt_llm._torch.debug.debug_hook import enable_debug_all_modules
#       model_config = ModelConfig(pretrained_config=llama_config,
#                                   attn_backend=backend)
#       llama = LlamaForCausalLM(model_config).to(dtype).to(device)
#       llama.load_weights(hf_llama.state_dict())
#       with torch.inference_mode():
#           enable_debug_all_modules(r"tensor_dump"):
#           attn_metadata.prepare()
#           logits = llama.forward(input_ids=input_ids,
#                                   position_ids=position_ids,
#                                   attn_metadata=attn_metadata)
#
# Note: this method need user to disable debug by calling disable_debug_all_modules
def enable_debug_all_modules(dest_folder: Optional[str] = None, ):
    assert get_current_debug_ctx() is None, ""
    debug_ctx = DebuggerContext(dest_folder)
    set_current_debug_ctx(debug_ctx)

    debug_ctx.get_current_modules_tree.clear()
    debug_ctx.get_module_indices_tree().clear()
    # parse env

    if debug_ctx.module_forward_pre_hook_handle is None:
        debug_ctx.module_forward_pre_hook_handle = torch.nn.modules.module.register_module_forward_pre_hook(
            module_pre_forward)
    if debug_ctx.module_forward_hook_handle is None:
        debug_ctx.module_forward_hook_handle = torch.nn.modules.module.register_module_forward_hook(
            module_after_forward)

    # TODO: Use ENV to enable debug features
    features: DebugFeatures = DebugFeatures.DUMPTENSOR
    if features | DebugFeatures.DUMPTENSOR:
        register_tensor_dump_hook(debug_ctx)


def disable_debug_all_modules():
    assert get_current_debug_ctx() is not None, ""
    debug_ctx.clear_state()
    set_current_debug_ctx(None)


# The context manager style interface to enable debugger on all classes inherit from nn.Module.
# Example:
#       from tensorrt_llm._torch.debug.debug_hook import debugger_addon_all_modules
#       model_config = ModelConfig(pretrained_config=llama_config,
#                                   attn_backend=backend)
#       llama = LlamaForCausalLM(model_config).to(dtype).to(device)
#       llama.load_weights(hf_llama.state_dict())
#        with torch.inference_mode() and debugger_addon_all_modules(r"tensor_dump"):
#            attn_metadata.prepare()
#            logits = llama.forward(input_ids=input_ids,
#                                   position_ids=position_ids,
#                                   attn_metadata=attn_metadata)
@contextmanager
def debugger_addon_all_modules(dest_folder: Optional[str] = None):
    try:
        enable_debug(dest_folder, filter)
    finally:
        disable_debug()


# The hook is registered to module with module.register_forward_pre_hook.
# This hook will be executed before module's forward is called.
# It will record module tree into debugCtx and call debugCtx's do_actions function
# to execute all hooks registered to debugCtx on current module.
def pre_forward(module: nn.Module, args, kwargs):
    name = module.name if hasattr(module, "name") else module.__class__.__name__
    debug_ctx = get_current_debug_ctx()
    assert debug_ctx is not None, "DebugContext instance shall not be None."
    debug_ctx.mark_in_pre_forward(True)
    debug_ctx.get_current_modules_tree().append(name)
    if len(debug_ctx.get_module_indices_tree()) == 0:
        debug_ctx.get_module_indices_tree().append(0)

    if len(debug_ctx.get_current_modules_tree()) >= len(
            debug_ctx.get_module_indices_tree()):
        debug_ctx.get_module_indices_tree().append(0)

    debug_ctx.get_module_indices_tree()[
        len(debug_ctx.get_current_modules_tree()) -
        1] = debug_ctx.get_module_indices_tree()[
            len(debug_ctx.get_current_modules_tree()) - 1] + 1
    debug_ctx.do_actions(module, args, debug_ctx.get_pre_forward_action())
    return None


# The hook is registered to module with module.register_forward_hook.
# This hook will be executed after module's forward is called.
# It will remove module from debugCtx and call debugCtx's do_actions function
# to execute all hooks registered to debugCtx on current module.
def after_forward(module: nn.Module, args, kwargs, output):
    debug_ctx = get_current_debug_ctx()
    debug_ctx.mark_in_pre_forward(False)
    debug_ctx.do_actions(module, [args, output],
                         debug_ctx.get_after_forward_action())
    name = module.name if hasattr(module, "name") else module.__class__.__name__
    old_name = debug_ctx.get_current_modules_tree().pop(-1)
    assert name == old_name, "module mismatch"

    debug_ctx.get_module_indices_tree().pop(-1)
    debug_ctx.tensor_counter = 0
    return None


# The function style to interface to enable debugger on model.
# If filter is provided, it will be used to filter out satisfied module to register hook.
# If filter is not provided, all modules will be registered with hooks.
# Example:
#       from tensorrt_llm._torch.debug.debug_hook import enable_debug
#       model_config = ModelConfig(pretrained_config=llama_config,
#                                   attn_backend=backend)
#       llama = LlamaForCausalLM(model_config).to(dtype).to(device)
#       llama.load_weights(hf_llama.state_dict())
#       with torch.inference_mode():
#           enable_debug(llama, r"tensor_dump"):
#           attn_metadata.prepare()
#           logits = llama.forward(input_ids=input_ids,
#                                   position_ids=position_ids,
#                                   attn_metadata=attn_metadata)
#
# Note: this method need user to disable debug by calling disable_debug
def enable_debug(model, dest_folder: Optional[str] = None, filter=None):
    debug_ctx = get_current_debug_ctx()
    assert debug_ctx is None, "DebugContext shall be None when enable debugger context."
    debug_ctx = DebuggerContext(dest_folder)
    set_current_debug_ctx(debug_ctx)

    debug_ctx.get_current_modules_tree().clear()
    debug_ctx.get_module_indices_tree().clear()
    for name, submodule in model.named_modules():
        if name == "":
            continue

        if submodule not in debug_ctx.forward_hook_handles:
            do_hook = filter(submodule) if filter is not None else True
            if do_hook:
                debug_ctx.forward_hook_handles[
                    submodule] = submodule.register_forward_hook(
                        after_forward, with_kwargs=True, always_call=True)

        if submodule not in debug_ctx.forward_pre_hook_handles:
            do_hook = filter(submodule) if filter is not None else True
            if do_hook:
                debug_ctx.forward_pre_hook_handles[
                    submodule] = submodule.register_forward_pre_hook(
                        pre_forward, with_kwargs=True)

    # TODO: Use ENV to enable debug features
    features: DebugFeatures = DebugFeatures.DUMPTENSOR
    if features | DebugFeatures.DUMPTENSOR:
        register_tensor_dump_hook(debug_ctx)


# The function style to interface to disable debugger on model.
def disable_debug():
    debug_ctx = get_current_debug_ctx()
    assert debug_ctx is not None, "DebugContext shall be None when enable debugger context."
    debug_ctx.clear_state()
    for _, handler in debug_ctx.forward_hook_handles.items():
        handler.remove()

    for _, handler in debug_ctx.forward_pre_hook_handles.items():
        handler.remove()

    debug_ctx.forward_hook_handles.clear()
    debug_ctx.forward_pre_hook_handles.clear()
    set_current_debug_ctx(None)


# The context manager style interface to enable debugger on model.
# If filter is provided, it will be used to filter out satisfied module to register hook.
# If filter is not provided, all modules will be registered with hooks.
# Example:
#       from tensorrt_llm._torch.debug.debug_hook import debugger_addon
#       model_config = ModelConfig(pretrained_config=llama_config,
#                                   attn_backend=backend)
#       llama = LlamaForCausalLM(model_config).to(dtype).to(device)
#       llama.load_weights(hf_llama.state_dict())
#        with torch.inference_mode() and debugger_addon(llama, r"tensor_dump"):
#            attn_metadata.prepare()
#            logits = llama.forward(input_ids=input_ids,
#                                   position_ids=position_ids,
#                                   attn_metadata=attn_metadata)
@contextmanager
def debugger_addon(model, dest_folder: Optional[str] = None, filter=None):
    try:
        enable_debug(model, dest_folder, filter)
        yield model
    finally:
        disable_debug()


input_tensor_names = []
tensor_counter = 0


def get_forward_arg_names(module: nn.Module):
    if hasattr(module, "forward"):
        forward_func = module.forward
        import inspect
        args = inspect.getfullargspec(forward_func).args
        return args

    return None


# Below is one hook for dump tensors.
# Normally, if you want implement one hook, you need to implement one filter by
# inheriting from base class Filter and one function which defines what to do,
# such as dump data, modify data, inject actions, etc.
class DumpTensorFilter(Filter):

    def __init__(self):
        pass

    def filter(self, module: nn.Module, debug_ctx: DebuggerContext):
        return True


def dump_tensor(module: nn.Module, data_tensor, debug_ctx: DebuggerContext):
    global input_tensor_names
    global tensor_counter

    input_tensor_names = get_forward_arg_names(module)
    if input_tensor_names is not None:
        input_tensor_names = input_tensor_names[1:]

    def get_dump_file_path(tensor):
        assert debug_ctx.get_log_folder(
        ) is not None, "Log folder shall be initialized by DebugContext."
        global tensor_counter

        name_parts = []
        for idx in range(len(debug_ctx.get_current_modules_tree())):
            inner_idx = f"{debug_ctx.get_module_indices_tree()[idx]}"
            layer_name = debug_ctx.get_current_modules_tree()[idx]
            name_parts.append(".".join([inner_idx, layer_name]))
        module_path = "-".join(name_parts)

        tensor_type = "input" if debug_ctx.check_in_pre_forward() else "output"
        if hasattr(tensor, "name") and tensor.name is not None:
            tensor_name = f"{tensor_type}.{tensor.name}.pt"
        elif tensor_counter < len(input_tensor_names):
            tensor_name = f"{tensor_type}.{input_tensor_names[tensor_counter]}.pt"
        else:
            tensor_name = f"{tensor_type}.{tensor_counter}.pt"

        tensor_counter += 1
        module_path = "-".join([module_path, tensor_name])
        from pathlib import Path
        p = Path(debug_ctx.get_log_folder()) / module_path
        return p.absolute()

    def dump_tensor_data(t):
        file_path = get_dump_file_path(t)
        # print(f"Saving tensor data to {file_path}")
        torch.save(t, file_path)

    def dump(t):
        if isinstance(t, torch.Tensor):
            dump_tensor_data(t)
        elif isinstance(t, tuple) or isinstance(t, list):
            for _t in t:
                dump(_t)

    tensor_counter = 0
    dump(data_tensor)
    tensor_counter = 0
    input_tensor_names.clear()


def register_tensor_dump_hook(debug_ctx):
    assert debug_ctx is not None, ""
    debug_ctx.register_pre_forward_action(DumpTensorFilter(), dump_tensor)
    debug_ctx.register_after_forward_action(DumpTensorFilter(), dump_tensor)
