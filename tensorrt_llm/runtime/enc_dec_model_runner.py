import json
import time
from pathlib import Path

# isort: off
import torch
import tensorrt as trt

from ..logger import logger
from .._utils import torch_to_numpy, trt_dtype_to_torch, mpi_world_size, mpi_rank
from ..plugin.plugin import CustomAllReduceHelper
from .generation import ModelConfig, SamplingConfig, LoraManager, GenerationSession
from ..mapping import Mapping
from .session import Session
from ..models.modeling_utils import get_kv_cache_type_from_legacy


def get_engine_name(rank):
    return 'rank{}.engine'.format(rank)


def read_config(config_path: Path):
    with open(config_path, "r") as f:
        config = json.load(f)

    builder_config = config['build_config']
    plugin_config = builder_config['plugin_config']
    pretrained_config = config['pretrained_config']
    lora_config = builder_config['lora_config']
    use_gpt_attention_plugin = plugin_config["gpt_attention_plugin"]
    remove_input_padding = plugin_config["remove_input_padding"]
    use_lora_plugin = plugin_config["lora_plugin"]
    tp_size = pretrained_config['mapping']['tp_size']
    pp_size = pretrained_config['mapping']['pp_size']
    gpus_per_node = pretrained_config['mapping']['gpus_per_node']
    world_size = tp_size * pp_size
    assert world_size == mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({mpi_world_size()})'
    num_heads = pretrained_config["num_attention_heads"]
    hidden_size = pretrained_config["hidden_size"]
    head_size = pretrained_config["head_size"]
    vocab_size = pretrained_config["vocab_size"]
    max_batch_size = builder_config["max_batch_size"]
    max_beam_width = builder_config["max_beam_width"]
    num_layers = pretrained_config["num_hidden_layers"]
    num_kv_heads = pretrained_config.get('num_kv_heads', num_heads)

    assert (num_heads % tp_size) == 0
    num_heads = num_heads // tp_size
    hidden_size = hidden_size // tp_size
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

    cross_attention = pretrained_config["architecture"] == "DecoderModel"
    skip_cross_kv = pretrained_config.get('skip_cross_kv', False)
    has_position_embedding = pretrained_config["has_position_embedding"]
    has_token_type_embedding = hasattr(pretrained_config, "type_vocab_size")
    dtype = pretrained_config["dtype"]

    paged_kv_cache = plugin_config['paged_kv_cache']
    tokens_per_block = plugin_config['tokens_per_block']

    gather_context_logits = builder_config.get('gather_context_logits', False)
    gather_generation_logits = builder_config.get('gather_generation_logits',
                                                  False)
    max_prompt_embedding_table_size = builder_config.get(
        'max_prompt_embedding_table_size', 0)

    kv_cache_type = get_kv_cache_type_from_legacy(True, paged_kv_cache)
    language_adapter_config = pretrained_config.get("language_adapter_config",
                                                    None)

    model_config = ModelConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_size=head_size,
        max_batch_size=max_batch_size,
        max_beam_width=max_beam_width,
        vocab_size=vocab_size,
        num_layers=num_layers,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        kv_cache_type=kv_cache_type,
        tokens_per_block=tokens_per_block,
        cross_attention=cross_attention,
        has_position_embedding=has_position_embedding,
        has_token_type_embedding=has_token_type_embedding,
        dtype=dtype,
        gather_context_logits=gather_context_logits,
        gather_generation_logits=gather_generation_logits,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        lora_plugin=use_lora_plugin,
        lora_target_modules=lora_config.get('lora_target_modules'),
        trtllm_modules_to_hf_modules=lora_config.get(
            'trtllm_modules_to_hf_modules'),
        skip_cross_kv=skip_cross_kv,
        language_adapter_config=language_adapter_config)

    return model_config, tp_size, pp_size, gpus_per_node, dtype


class EncDecModelRunner:

    def __init__(self,
                 engine_name,
                 engine_dir,
                 lora_dir=None,
                 lora_task_uids=None,
                 debug_mode=False,
                 skip_encoder=False,
                 stream: torch.cuda.Stream = None,
                 enable_context_fmha_fp32_acc: bool = None):
        # in multi-node setup, it's important to set_device at the very beginning so .to('cuda') refers to current device
        # accordingly, all input & output tensors should be moved to current device
        # otherwise, it's default to 'cuda:0'
        self.runtime_rank = mpi_rank()
        device_id = self.runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.device = torch.cuda.current_device()
        self.skip_encoder = skip_encoder
        self.lora_task_uids = lora_task_uids
        self.enable_context_fmha_fp32_acc = enable_context_fmha_fp32_acc

        # when enc-dec runs by itself, stream can be None and we create new stream here
        # when enc-dec has to run as a component in a bigger workflow (e.g., multimodal), earlier components in the workflow may have results in its stream, which we should pass that stream in to avoid unnecessary stream sync
        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        engine_dir = Path(engine_dir)

        def engine_setup(component):
            # model config
            config_path = engine_dir / component / "config.json"
            logger.info(f"Using config path {config_path}")
            model_config, tp_size, pp_size, gpus_per_node, dtype = read_config(
                config_path)

            # MGMN config
            world_size = tp_size * pp_size
            runtime_rank = mpi_rank()
            assert runtime_rank < world_size, "Runtime GPU rank exceeds MPI world size. Did you launch more MPI processes than required?"
            runtime_mapping = Mapping(world_size,
                                      runtime_rank,
                                      tp_size=tp_size,
                                      pp_size=pp_size,
                                      gpus_per_node=gpus_per_node)

            # load engine
            engine_fname = get_engine_name(runtime_rank)
            with open(engine_dir / component / engine_fname, "rb") as f:
                engine_buffer = f.read()

            return model_config, runtime_mapping, engine_buffer

        # Note: encoder and decoder doesn't necessarily have the same TP & PP config

        if not skip_encoder:
            self.encoder_model_config, self.encoder_runtime_mapping, encoder_engine_buffer = engine_setup(
                component='encoder')

            self.nccl_comm = None
            if self.encoder_runtime_mapping.has_pp():
                # for Pipeline Parallelism in encoder
                self.nccl_comm = torch.classes.trtllm.NcclCommunicatorOp(
                    self.encoder_runtime_mapping.world_size,
                    self.encoder_runtime_mapping.rank)

            # session setup
            self.encoder_session = Session.from_serialized_engine(
                encoder_engine_buffer)

            # encoder lora manager setup
            if self.encoder_model_config.lora_plugin:
                self.encoder_lora_manager = LoraManager(
                    mapping=self.encoder_runtime_mapping,
                    model_config=self.encoder_model_config,
                )
                # TODO: this is only for bart
                self.encoder_lora_manager.load_from_hf(
                    model_dirs=lora_dir,
                    model_config=self.encoder_model_config,
                    component='encoder',
                )
            else:
                self.encoder_lora_manager = None
        else:
            self.encoder_model_config, self.encoder_runtime_mapping, encoder_engine_buffer = None, None, None
            self.nccl_comm, self.encoder_session = None, None

        self.decoder_model_config, self.decoder_runtime_mapping, decoder_engine_buffer = engine_setup(
            component='decoder')
        self.decoder_session = GenerationSession(self.decoder_model_config,
                                                 decoder_engine_buffer,
                                                 self.decoder_runtime_mapping,
                                                 debug_mode=debug_mode)

        # decoder lora manager setup
        if self.decoder_model_config.lora_plugin:
            self.decoder_lora_manager = LoraManager(
                mapping=self.decoder_runtime_mapping,
                model_config=self.decoder_model_config,
            )
            # TODO: this is only for bart
            self.decoder_lora_manager.load_from_hf(
                model_dirs=lora_dir,
                model_config=self.decoder_model_config,
                component='decoder',
            )
        else:
            self.decoder_lora_manager = None

    @classmethod
    def from_engine(cls,
                    engine_name,
                    engine_dir,
                    lora_dir=None,
                    lora_task_uids=None,
                    debug_mode=False,
                    skip_encoder=False,
                    stream=None,
                    enable_context_fmha_fp32_acc=None):
        return cls(engine_name,
                   engine_dir,
                   lora_dir,
                   lora_task_uids,
                   debug_mode=debug_mode,
                   skip_encoder=skip_encoder,
                   stream=stream,
                   enable_context_fmha_fp32_acc=enable_context_fmha_fp32_acc)

    def process_input(self,
                      input_ids,
                      remove_input_padding=False,
                      pad_token_id=0,
                      prompt_tasks=None,
                      language_adapter_routings=None):
        if remove_input_padding:
            # in remove padding mode --> flatten input, calculate actual length and max length
            # Note: 1st token should never be removed, even if it is pad_token_id
            first_ids = input_ids[:, 0]
            input_ids = input_ids[:, 1:]
            input_lengths = 1 + (input_ids != pad_token_id).sum(dim=1).type(
                torch.IntTensor).to(self.device)  # [batch_size]
            new_ids = []
            for i in range(len(input_ids)):
                row = input_ids[i, :]
                row = row[row != pad_token_id]
                new_ids.append(
                    torch.cat(
                        (torch.IntTensor([first_ids[i]]).to(self.device), row)))
            input_ids = torch.cat(new_ids)  # [num_tokens]
            if prompt_tasks is not None:
                prompt_tasks = prompt_tasks[:input_ids.shape[0]]
        else:
            # in padding mode --> keep input, just calculate actual length and max length
            # Note: 1st token should always count, even if it is pad_token_id. e.g., decoder start id in enc-dec models could be a single pad_token_id, we should count
            input_lengths = torch.tensor(
                1 + (input_ids[:, 1:] != pad_token_id).sum(dim=1).type(
                    torch.IntTensor).to(self.device),
                dtype=torch.int32,
                device=self.device)
        max_input_length = torch.max(input_lengths).item()
        if language_adapter_routings is not None:
            language_adapter_routings = language_adapter_routings.to(
                self.device)
        return input_ids, input_lengths, max_input_length, prompt_tasks, language_adapter_routings

    def encoder_run(self,
                    input_ids,
                    input_lengths,
                    max_input_length,
                    position_ids=None,
                    token_type_ids=None,
                    debug_mode=False,
                    prompt_embedding_table=None,
                    prompt_tasks=None,
                    prompt_vocab_size=None,
                    attention_mask=None,
                    language_adapter_routings=None):

        # each engine has hidden_dim/TP, don't forget to multiply TP
        hidden_size = self.encoder_model_config.hidden_size * self.encoder_runtime_mapping.tp_size
        if input_ids.dim() == 1:
            hidden_states_shape = (input_ids.shape[0], hidden_size
                                   )  # [num_tokens,D]
        else:
            hidden_states_shape = (input_ids.shape[0], input_ids.shape[1],
                                   hidden_size)  # [BS,seqlen,D]
        hidden_states_dtype = lambda name: trt_dtype_to_torch(
            self.encoder_session.engine.get_tensor_dtype(name))

        # input tensors. only first PP rank has id input, others are hidden_states input
        inputs = {}
        if self.encoder_runtime_mapping.is_first_pp_rank():
            inputs['input_ids'] = input_ids.contiguous()
            if self.encoder_model_config.has_position_embedding:
                if position_ids is None:
                    if self.encoder_model_config.remove_input_padding:
                        position_ids = [
                            torch.arange(sample_length,
                                         dtype=torch.int32,
                                         device=input_ids.device)
                            for sample_length in torch_to_numpy(input_lengths)
                        ]
                        position_ids = torch.cat(position_ids)
                    else:
                        bsz, seq_len = input_ids.shape[:2]
                        position_ids = torch.arange(
                            seq_len, dtype=torch.int32,
                            device=input_ids.device).expand(bsz, -1)
                inputs['position_ids'] = position_ids.contiguous()
            if self.encoder_model_config.has_token_type_embedding:
                inputs['token_type_ids'] = token_type_ids.contiguous()

            if self.encoder_model_config.max_prompt_embedding_table_size > 0:
                inputs[
                    'prompt_embedding_table'] = prompt_embedding_table.contiguous(
                    )
                inputs['tasks'] = prompt_tasks.contiguous()
                inputs['prompt_vocab_size'] = prompt_vocab_size.contiguous()
        else:
            # just need a placeholder, engine will call NCCL to recv and fill data from previous rank
            inputs['hidden_states_input'] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype('hidden_states_input'),
                device=self.device).contiguous()
        if attention_mask is not None and not self.encoder_model_config.gpt_attention_plugin:
            inputs['attention_mask'] = attention_mask.contiguous()

        inputs['input_lengths'] = input_lengths
        # use shape info to pass max length info in remove padding mode
        inputs['max_input_length'] = torch.empty(
            (max_input_length, ),
            dtype=hidden_states_dtype('max_input_length'),
            device=self.device).contiguous()

        if self.encoder_runtime_mapping.tp_size > 1:
            ipc_buffers, all_reduce_workspace = CustomAllReduceHelper.allocate_workspace(
                self.encoder_runtime_mapping,
                CustomAllReduceHelper.max_workspace_size_auto(
                    self.encoder_runtime_mapping.tp_size))
            inputs['all_reduce_workspace'] = all_reduce_workspace

        if self.encoder_model_config.lora_plugin:
            inputs.update(
                self.encoder_lora_manager.input_buffers(
                    self.lora_task_uids,
                    self.encoder_runtime_mapping,
                    self.encoder_model_config.num_layers,
                ))
            batch_size = input_lengths.size(0)
            inputs['host_request_types'] = torch.IntTensor([0] *
                                                           batch_size).to('cpu')
            if self.encoder_model_config.remove_input_padding:
                inputs['host_context_lengths'] = input_lengths.to('cpu')
        if language_adapter_routings is not None:
            inputs['language_adapter_routings'] = language_adapter_routings
        # Note: runtime.Session's run() method will set input/output tensor address, here we only need to provide tensor shape
        self.encoder_session.set_shapes(inputs)

        # output tensors. only last PP rank final encoder output, others are intermediate hidden_states output. Need broadcast later
        outputs = {}
        if self.encoder_runtime_mapping.is_last_pp_rank():
            outputs['encoder_output'] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype('encoder_output'),
                device=self.device).contiguous()
        else:
            outputs['hidden_states_output'] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype('hidden_states_output'),
                device=self.device).contiguous()

        # -------------------------------------------
        if debug_mode:
            engine = self.encoder_session.engine
            context = self.encoder_session.context
            # setup debugging buffer for the encoder
            for i in range(self.encoder_session.engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(
                        name
                ) == trt.TensorIOMode.OUTPUT and name not in outputs.keys():
                    dtype = engine.get_tensor_dtype(name)
                    shape = context.get_tensor_shape(name)
                    outputs[name] = torch.zeros(tuple(shape),
                                                dtype=trt_dtype_to_torch(dtype),
                                                device=self.device)
                    context.set_tensor_address(name, outputs[name].data_ptr())
        # -------------------------------------------

        # TRT session run
        # Note: need cuda stream ID, not a torch Stream
        ok = self.encoder_session.run(inputs, outputs, self.stream.cuda_stream)
        assert ok, "Runtime execution failed"
        self.stream.synchronize()

        # Tensor Parallelism is handled by model/engine definition
        # But we need to broadcast among PP group at the end of encoder's Pipeline Parallelism
        # After this, all ranks should recv the encoder output, and world might be re-configured using decoder's TP-PP config
        def pp_communicate_encoder_output(encoder_output):
            if self.encoder_runtime_mapping.is_last_pp_rank():
                for pp_rank in self.encoder_runtime_mapping.pp_group:
                    if pp_rank != self.encoder_runtime_mapping.rank:
                        self.nccl_comm.send(encoder_output, pp_rank)
                return encoder_output
            else:
                self.nccl_comm.recv(encoder_output,
                                    self.encoder_runtime_mapping.pp_group[-1])
                return encoder_output

        if self.encoder_runtime_mapping.has_pp():
            # use hidden_states output buffer to receive output as the shapes are same
            encoder_output_buf = outputs[
                'encoder_output'] if self.encoder_runtime_mapping.is_last_pp_rank(
                ) else outputs['hidden_states_output']
            encoder_output = pp_communicate_encoder_output(encoder_output_buf)
        else:
            encoder_output = outputs['encoder_output']

        return encoder_output

    def generate(self,
                 encoder_input_ids,
                 decoder_input_ids,
                 max_new_tokens,
                 num_beams=1,
                 pad_token_id=None,
                 eos_token_id=None,
                 bos_token_id=None,
                 debug_mode=False,
                 return_dict=False,
                 prompt_embedding_table=None,
                 prompt_tasks=None,
                 prompt_vocab_size=None,
                 attention_mask=None,
                 time_encoder=False,
                 return_encoder_output=False,
                 encoder_language_adapter_routings=None,
                 decoder_language_adapter_routings=None):
        ## ensure all externally provided tensors are on the correct device.
        encoder_input_ids = encoder_input_ids.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)

        if attention_mask is not None:
            attention_mask = torch.tensor(attention_mask,
                                          dtype=torch.int32,
                                          device=self.device)

        ## encoder run
        encoder_remove_input_padding = self.encoder_model_config.remove_input_padding if self.encoder_model_config else self.decoder_model_config.remove_input_padding

        encoder_input_ids, encoder_input_lengths, encoder_max_input_length, prompt_tasks, encoder_language_adapter_routings = self.process_input(
            encoder_input_ids, encoder_remove_input_padding, pad_token_id,
            prompt_tasks, encoder_language_adapter_routings)

        if not self.skip_encoder:
            logger.info(f"Rank {self.runtime_rank} Running encoder engine ...")
            if time_encoder:
                tik = time.time()
            encoder_output = self.encoder_run(
                encoder_input_ids,
                encoder_input_lengths,
                encoder_max_input_length,
                debug_mode=debug_mode,
                prompt_embedding_table=prompt_embedding_table,
                prompt_tasks=prompt_tasks,
                prompt_vocab_size=prompt_vocab_size,
                attention_mask=attention_mask,
                language_adapter_routings=encoder_language_adapter_routings)
            if time_encoder:
                tok = time.time()
                print(f"TRT-LLM Encoder time {(tok-tik)*1000}ms")
        else:
            encoder_output = prompt_embedding_table
            if encoder_input_ids.dim() > 1:
                encoder_output = encoder_output.unsqueeze(0)

        ## decoder run
        logger.info(f"Rank {self.runtime_rank} Running decoder engine ...")
        decoder_input_ids, decoder_input_lengths, decoder_max_input_length, _, decoder_language_adapter_routings = self.process_input(
            decoder_input_ids, self.decoder_model_config.remove_input_padding,
            pad_token_id, None, decoder_language_adapter_routings)
        # `cross_attention_mask` in context phase [batch_size, query_len, encoder_input_len]
        # where query_len happens to be 1 in current cases, but not necessarily always, and
        # `cross_attention_mask` in generation phase [batch_size, 1, encoder_input_len] where
        # the query_len is always 1 since we have kv cache. But we use
        # cross_attention_mask[:, step, :] during generation
        cross_attention_mask = None
        if attention_mask is not None:
            cross_attention_mask = torch.tensor(attention_mask,
                                                dtype=torch.int32,
                                                device=self.device).reshape(
                                                    attention_mask.shape[0], 1,
                                                    attention_mask.shape[1])
            cross_attention_mask = cross_attention_mask.repeat(
                [1, decoder_max_input_length + max_new_tokens, 1])

        # generation config
        sampling_config = SamplingConfig(end_id=eos_token_id,
                                         pad_id=pad_token_id,
                                         num_beams=num_beams,
                                         min_length=1,
                                         return_dict=return_dict)
        sampling_config.update(output_cum_log_probs=return_dict,
                               output_log_probs=return_dict)

        # decoder autoregressive generation
        self.decoder_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            num_beams,
            max_attention_window_size=None,
            encoder_max_input_length=encoder_max_input_length,
            lora_manager=self.decoder_lora_manager,
            lora_uids=self.lora_task_uids,
            enable_context_fmha_fp32_acc=self.enable_context_fmha_fp32_acc)

        output = self.decoder_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_output,
            encoder_input_lengths=encoder_input_lengths,
            return_dict=return_dict,
            cross_attention_mask=cross_attention_mask,
            language_adapter_routings=decoder_language_adapter_routings)

        if return_dict and return_encoder_output:
            output['encoder_output'] = encoder_output

        return output
