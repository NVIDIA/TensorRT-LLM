# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import tensorrt as trt

from ._common import default_net
from ._utils import mpi_rank
from .logger import logger


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
            start_layer_id = current_net.trt_network.num_layers
            output = self.forward(*args, **kwargs)
            end_layer_id = current_net.trt_network.num_layers
            current_net._module_call_stack.set_layer_range(
                self, range(start_layer_id, end_layer_id))
            return output

    def __getattr__(self, name):
        parameters = self.__dict__.get('_parameters')
        if parameters is not None and name in parameters:
            return parameters[name]

        modules = self.__dict__.get('_modules')
        if modules is not None and name in modules:
            return modules[name]

        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

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
                self.__dict__.get('_parameters')[name] = value
            elif isinstance(value, Module):
                self.__dict__.get('_modules')[name] = value
            else:
                super().__setattr__(name, value)

        else:
            # if base class has the attribute, reset as follows:
            # - when reset as Parameter or Module attr, remove from base & add to this Module class
            # - other types reset and remain in base class
            if isinstance(value, Parameter):
                super().__delattr__(name)
                self.__dict__.get('_parameters')[name] = value
            elif isinstance(value, Module):
                super().__delattr__(name)
                self.__dict__.get('_modules')[name] = value
            else:
                super().__setattr__(name, value)

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix,
                                              remove_duplicate):
                    yield m

    def named_children(self):
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix,
                                                                      self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def parameter(self, recurse=True):
        for name, param in self.named_parameters():
            yield param

    def named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(lambda module: module._parameters.items(),
                                  prefix=prefix,
                                  recurse=recurse)
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
                yield name + ('.' if name else '') + n, output

    def update_parameters(self, torch_module):
        m = {k: v for k, v in self.named_parameters()}
        tm = {k: v for k, v in torch_module.named_parameters()}

        assert sorted(m.keys()) == sorted(
            tm.keys()
        ), 'The parameter names of the tensorrt-llm module must be the same with the torch module'

        for k, v in self.named_parameters():
            v.value = tm[k].detach().cpu().numpy()


class TopLevelModuleMixin:
    '''
        The Module class are reused between building blocks (like Attention, MLP) and the top level model (like LLaMAForCausalLM)
        While there are some functions, like the loading hf/ft weights, or build/load trt engines are only supported by the top level model, not the building blocks.
        So top level model class like: LLaMAForCausalLM shall inherit this class.
    '''

    def __init__(self) -> None:
        super().__init__()
        self._trt_engine = None
        self._builder_config = None
        self.config = None

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, cfg: 'PretrainedConfig'):
        # import here to avoid circular importing
        from .models.modeling_utils import PretrainedConfig
        assert isinstance(cfg,
                          PretrainedConfig), f"Got unexpected type {type(cfg)}"
        self._config = cfg

    @classmethod
    def from_hugging_face(cls, hf_model_dir: str, **kwargs):
        '''create and object and load weights from hugging face'''
        raise NotImplementedError("Subclass shall override this")

    @classmethod
    def from_faster_transformer(cls, ft_model_dir: str, **kwargs):
        '''create and object and load weights from FasterTransformer'''
        raise NotImplementedError("Subclass shall override this")

    def quantize(self, *args, **kwargs):
        '''Quantize a given model object and return the quantized Module object
        '''
        raise NotImplementedError("Will add in future release")

    def to_trt(self,
               batch_size: int,
               input_len: int,
               output_len: int,
               plugin_config: 'PluginConfig' = None,
               **kwargs) -> Tuple[trt.IHostMemory, 'BuilderConfig']:
        '''Build TRT engine from the Module using given size limits
            Parameters:
                batch_size: the max batch size can be used in the runtime by one generate call
                input_len: the max input length of one input sequence
                output_len: the max output length
                plugin_config: PluginConfig
                    When the plugin_config is None, to_trt() will call default_plugin_config() to build the engine.
                other optional kwargs:
                    Other optional fields can be accepted that affects the build behavior, set these if you need finer control:
                    - max_beam_width: int, default 1,  the max beam search width
                    - max_num_tokens: int, default batch_size * input_len, max number of tokens one engine forward pass can do,
                    - timing_cache: str, default None, a file contains the previous timing cache
                    - strongly_typed: bool, default False, a bool flag to indicate if or not use the strong type mode of TRT
                    - builder_opt: int, default None, TRT builder opt level
                    - gather_all_logits: bool, default False, whether or not to gather all the logits.
                        When this is False, the engine does output all logits, and thus needs additional input Tensor to the engine
                        for each request to gather the last generated token for that request

        '''
        from . import profiler
        from .builder import Builder
        from .network import net_guard
        profiler.start("Network construction and build engine")
        assert self.config is not None, "Module.config not set"
        assert isinstance(self, Module), "to_trt use self.named_parameters()"

        builder = Builder()
        builder_config = builder.create_builder_config(
            # model attribute section
            name=self.config.architecture,
            precision=self.config.dtype,
            tensor_parallel=self.config.mapping.tp_size,
            pipeline_parallel=self.config.mapping.pp_size,
            num_layers=self.config.num_hidden_layers,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            hidden_size=self.config.hidden_size,
            vocab_size=self.config.vocab_size,
            hidden_act=self.config.hidden_act,
            max_position_embeddings=self.config.max_position_embeddings,
            quant_mode=self.config.quant_mode,
            # trt build config section
            timing_cache=kwargs.get("timing_cache", None),
            max_batch_size=batch_size,
            max_input_len=input_len,
            max_output_len=output_len,
            max_beam_width=kwargs.get("max_beam_width", 1),
            max_num_tokens=kwargs.get("max_num_tokens", batch_size * input_len),
            int8=False,  #TODO: support int8, see examples/llama/build.py
            strongly_typed=kwargs.get("strongly_typed", False),
            opt_level=kwargs.get("builder_opt", None),
            max_prompt_embedding_table_size=getattr(
                self, 'max_prompt_embedding_table_size', 0),
            gather_all_token_logits=kwargs.get('gather_all_token_logits',
                                               False),
            lora_target_modules=None,  #TODO: support lora
        )
        network = builder.create_network()
        # use default if user don't provide one
        network.plugin_config = plugin_config if plugin_config is not None else self.default_plugin_config(
            **kwargs)
        with net_guard(network):
            # Prepare
            network.set_named_parameters(self.named_parameters())

            # Forward
            inputs = self.prepare_inputs(
                batch_size,
                input_len,
                output_len,
                True,
                kwargs.get('max_beam_width', 1),
                max_num_tokens=kwargs.get("max_num_tokens",
                                          batch_size * input_len),
                prompt_embedding_table_size=getattr(
                    self, 'max_prompt_embedding_table_size', 0),
                gather_all_token_logits=kwargs.get('gather_all_token_logits',
                                                   False),
                lora_target_modules=None)  #TODO: support lora
            self(*inputs)
        engine = builder.build_engine(network, builder_config)
        self._trt_engine = engine
        self._builder_config = builder_config
        profiler.stop("Network construction and build engine")
        return engine, builder_config

    def _generate(self,
                  input_text: Union[List[str], "torch.Tensor"],
                  max_new_tokens: Optional[str] = None,
                  sampling_config: Optional["SamplingConfig"] = None,
                  tokenizer_dir: Optional[str] = None,
                  engine_dir: Optional[str] = None,
                  lora_uids: Optional[list] = None,
                  prompt_tasks: Optional[List[int]] = None,
                  streaming: bool = False,
                  stopping_criteria: Optional["StoppingCriteria"] = None,
                  logits_processor: Optional["LogitsProcessor"] = None,
                  **kwargs) -> Iterable[Tuple[str, str]]:
        '''
        Note: this is private for test purpose only, use higher level LLM class for generation.
        Parameters:
            input_text_or_ids:
            max_new_tokens: when it's None, use the output_len specified in to_trt
            tokenizer_dir: the directory contains the tokenizer config
            engine_dir: the directory contains the engine, optional. When it's none, user much call to_trt to compile the engine firstly.
            lora_uids: list of integers with length as batch size, each integer is lora id for one input sequence
                When lora_dir and lora_uid is not None, the use_lora() function must be called before to_trt() function builds the engine.
            prompt_tasks: Optional[List[int]]
                prompt tuning tasks for each input sequence
            TODO: add all other features supported by current runtime APIs.
        '''
        from .mapping import Mapping
        from .runtime import (GenerationSession, LoraManager, ModelRunner,
                              SamplingConfig, model_runner)

        # create and cache tokenizer and runner
        if getattr(self, 'tokenizer', None) is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                      legacy=False,
                                                      padding_side='left',
                                                      truncation_side='left',
                                                      trust_remote_code=True,
                                                      use_fast=True)
            self.tokenizer = tokenizer

        if getattr(self, 'runner', None) is None:
            if engine_dir is not None:  # read from the engine_dir, use ModelRunner
                assert Path(engine_dir).exists(
                ), f"Invalid engine_dir argument, the path does not exist {engine_dir}"
                model_config, other_config = model_runner.read_config(
                    Path(engine_dir) / "config.json")
                self.runner = ModelRunner.from_dir(
                    engine_dir=engine_dir,
                    lora_dir=getattr(self, 'lora_dir', None),
                    lora_ckpt_source=getattr(self, "lora_ckpt_source", 'hf'))
            else:
                assert hasattr(
                    self, '_trt_engine'
                ), f"must call to_trt(...) function firstly to build model to trt engine"
                assert isinstance(
                    self._trt_engine, trt.IHostMemory
                ), f"Engine type is unexpected, got {type(self._trt_engine)}"
                assert hasattr(
                    self, '_builder_config'
                ), f"Internal error: to_trt() shall set self._builder_config"
                model_config, other_config = model_runner._builder_to_model_config(
                    self._builder_config.to_dict())
                world_size = other_config.get('world_size')
                tp_size = other_config.get('tp_size')
                pp_size = other_config.get('pp_size')
                max_batch_size = other_config.get('max_batch_size')
                max_input_len = other_config.get('max_input_len')
                max_output_len = other_config.get('max_output_len')
                max_beam_width = other_config.get('max_beam_width')
                rank = mpi_rank()
                runtime_mapping = Mapping(world_size=world_size,
                                          rank=rank,
                                          tp_size=tp_size,
                                          pp_size=pp_size)
                session = GenerationSession(model_config, self._trt_engine,
                                            runtime_mapping)
                lora_manager = None
                if session.use_lora_plugin:
                    assert getattr(self.lora_dir, None) is not None, \
                        "lora_dir should not be None for engine built with lora_plugin enabled."
                    lora_manager = LoraManager()
                    lora_manager.load_from_ckpt(
                        model_dir=self.lora_dir,
                        model_config=model_config,
                        runtime_mapping=runtime_mapping,
                        ckpt_source=self.lora_ckpt_source)
                self.runner = ModelRunner(session, max_batch_size,
                                          max_input_len, max_output_len,
                                          max_beam_width, lora_manager)
        assert self.runner is not None

        tokenizer = self.tokenizer
        if sampling_config is None:
            sampling_config = SamplingConfig(
                end_id=tokenizer.eos_token_id,
                pad_id=tokenizer.eos_token_id
                if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
                max_new_tokens=max_new_tokens)
            sampling_config.output_sequence_lengths = True
            sampling_config.return_dict = True
        import torch
        assert not isinstance(
            input_text, torch.Tensor), "Only input string is supported for now"

        def generate_on_batch(batch_input_ids: List[torch.Tensor],
                              batch_input_text: List[str]):
            batch_input_ids = [
                torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
            ]  # List[torch.Tensor(seq)]

            assert len(batch_input_ids) <= other_config['max_batch_size'], \
                f"Can not run batch size larger than {other_config['max_batch_size']}, got {len(batch_input_ids)}"
            # generate
            outputs = self.runner.generate(batch_input_ids,
                                           sampling_config,
                                           prompt_table_path=getattr(
                                               self, "prompt_table_path", None),
                                           prompt_tasks=prompt_tasks,
                                           lora_uids=lora_uids,
                                           streaming=streaming,
                                           stopping_criteria=stopping_criteria,
                                           logits_processor=logits_processor)

            # parse and print output
            output_ids = outputs['output_ids']
            sequence_lengths = outputs['sequence_lengths']

            batch_size, num_beams, max_len = output_ids.size()
            input_lengths = [x.size(0) for x in batch_input_ids]
            assert num_beams == 1, "Support beam search later"

            batched_output = []
            for batch_idx in range(batch_size):
                for beam in range(num_beams):
                    inputs = output_ids[batch_idx][
                        0][:input_lengths[batch_idx]].tolist()
                    input_text_echo = tokenizer.decode(
                        inputs)  # output_ids shall contain the input ids

                    output_begin = input_lengths[batch_idx]
                    output_end = sequence_lengths[batch_idx][beam]
                    outputs = output_ids[batch_idx][beam][
                        output_begin:output_end].tolist()

                    output_text = tokenizer.decode(outputs)
                    assert input_text_echo == "<s> " + batch_input_text[
                        batch_idx], f"Got {input_text_echo}, expect: {batch_input_text[batch_idx]}"
                    batched_output.append(
                        (batch_input_text[batch_idx], output_text))
            return batched_output

        # prepare inputs
        batch_input_ids = []
        batch_input_text = []
        for inp in input_text:
            # we'd like the avoid the padding and needs to know each seq's length, so tokenize them one by one
            input_ids = tokenizer.encode(
                inp, truncation=True, max_length=other_config['max_input_len'])
            batch_input_ids.append(input_ids)
            batch_input_text.append(inp)
            #TODO: handling batching better, use batch size for demo purpose for now.
            GEN_BATCH_SIZE = 1
            if len(batch_input_ids) >= GEN_BATCH_SIZE:
                batch_io_pairs = generate_on_batch(batch_input_ids,
                                                   batch_input_text)
                # return text to user
                for i, o in batch_io_pairs:
                    yield i, o
                # clear buffer to next batch
                batch_input_ids = []
                batch_input_text = []

    def save(self, engine_dir):
        ''' Save the engine and build config to given directory
        '''

        from .builder import Builder
        engine_dir = Path(engine_dir)
        if not engine_dir.exists():
            engine_dir.mkdir()
        config_path = engine_dir / 'config.json'

        def get_engine_name(model, dtype, tp_size, pp_size, rank):
            if pp_size == 1:
                return '{}_{}_tp{}_rank{}.engine'.format(
                    model, dtype, tp_size, rank)
            return '{}_{}_tp{}_pp{}_rank{}.engine'.format(
                model, dtype, tp_size, pp_size, rank)

        #TODO: implement multi gpus names
        engine_path = engine_dir / get_engine_name(
            self._builder_config.name, self.config.dtype,
            self.config.mapping.tp_size, self.config.mapping.pp_size,
            self.config.mapping.rank)
        builder = Builder()
        builder.save_config(self._builder_config, config_path)
        with open(engine_path, 'wb') as f:
            f.write(self._trt_engine)

    def load_trt(self, engine: trt.IHostMemory, **kwargs):
        '''Load trt engine for this model
        '''
        raise NotImplementedError

    def use_lora(self, lora_dir: str, lora_ckpt_source: str):
        '''Load lora weights and config from the give dir to the module. lora_format should be one of 'hf' or 'nemo'.
           lora_dir: the directory contains the lora weights
        '''
        # TODO: this is build time API, so pack the lora data together as engine
        self.lora_dir = lora_dir
        self.lora_ckpt_source = lora_ckpt_source
        raise NotImplementedError  # Fill more details later

    def use_prompt_tuning(self, max_prompt_embedding_table_size: str,
                          prompt_table_path: str):
        '''Enable p tuning when build the TRT engine, call this before to_trt
        '''
        # TODO: this is build time API, so pack the p-tuning table data together as engine,
        #  otherwise, if the build and runtime path has different p tuning table path, it will fail.
        self.prompt_table_path = prompt_table_path
        # TODO: change the embedding layer member after this.
        self.max_prompt_embedding_table_size = max_prompt_embedding_table_size
        raise NotImplementedError  # Fill more details later

    def use_streaming_llm(self, sink_token_length: int):
        '''Enable Streaming-LLM feature
        '''
        raise NotImplementedError

    def config_moe(self, moe_top_k: int, moe_tp_mode, moe_renorm_mode):
        '''Configure the moe tuning parameters, the model must a MoE model, otherwise, this fails.
        '''
        raise NotImplementedError

    def default_plugin_config(self, **kwargs) -> 'PluginConfig':
        '''Return the default plugin config for this model, when the plugin_config value is not given in to_trt() call.
           If user needs to set different plugin configs, can start from the return object and change it.
        '''
        from .plugin.plugin import ContextFMHAType, PluginConfig
        plugin_config = PluginConfig()
        plugin_config.set_gpt_attention_plugin()
        plugin_config.set_gemm_plugin()
        plugin_config.set_rmsnorm_plugin()

        # Quantization plugins.
        plugin_config.set_context_fmha(ContextFMHAType.enabled_with_fp32_acc)
        if self.mapping.world_size > 1:
            plugin_config.set_nccl_plugin()
        plugin_config.enable_remove_input_padding()
        #TODO: tune this to get best perf with minimum code change
        return plugin_config


class ModuleList(Module):

    def __init__(self, modules) -> None:
        super(ModuleList, self).__init__()
        offset = len(self)
        for i, module in enumerate(modules):
            self._modules[str(offset + i)] = module

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
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
