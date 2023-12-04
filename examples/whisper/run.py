import argparse
import json
import time
from pathlib import Path

# isort: off
import torch
import tensorrt as trt
# isort: on
from transformers import AutoConfig, AutoTokenizer, WhisperForConditionalGeneration

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._utils import trt_dtype_to_torch
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip

import whisper


def print_tensor(tensor_name, tensor, num_elements=10):
    print(
        f'{tensor_name}: mean={tensor.abs().mean().item():.3f}, sum={tensor.abs().sum().item():.3f}, max={tensor.abs().max().item():.3f}'
    )
    # Pass num_elements=-1 will print the whole tensor
    if num_elements < 0:
        num_elements = torch.numel(tensor)
    print(f'{tensor.flatten()[:num_elements]}')
    print("Tensor Shape: ", tensor.size())
    print("")


def read_config(config_path: Path):
    with open(config_path, "r") as f:
        config = json.load(f)
    use_gpt_attention_plugin = config["plugin_config"]["gpt_attention_plugin"]
    remove_input_padding = config["plugin_config"]["remove_input_padding"]
    tp_size = config['builder_config']['tensor_parallel']
    pp_size = config['builder_config']['pipeline_parallel']
    gpus_per_node = config['builder_config']['gpus_per_node']
    world_size = tp_size * pp_size
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config["builder_config"]["num_heads"]
    hidden_size = config["builder_config"]["hidden_size"]
    head_size = config["builder_config"]["head_size"]
    vocab_size = config["builder_config"]["vocab_size"]
    num_layers = config["builder_config"]["num_layers"]
    num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)

    assert (num_heads % tp_size) == 0
    num_heads = num_heads // tp_size
    hidden_size = hidden_size // tp_size
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

    cross_attention = config["builder_config"]["cross_attention"]
    has_position_embedding = config["builder_config"]["has_position_embedding"]
    has_token_type_embedding = config["builder_config"][
        "has_token_type_embedding"]
    use_custom_all_reduce = config['plugin_config'].get('use_custom_all_reduce',
                                                        False)
    dtype = config["builder_config"]["precision"]

    model_config = ModelConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_size=head_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        cross_attention=cross_attention,
        has_position_embedding=has_position_embedding,
        has_token_type_embedding=has_token_type_embedding,
        use_custom_all_reduce=use_custom_all_reduce,
        dtype=dtype)

    return model_config, tp_size, pp_size, gpus_per_node, dtype


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument('--max_kv_cache_length',
                        type=int,
                        default=None,
                        help='The max kv cache length. \
              If the final sequence length exceeds the kv cache length, we will enable cyclic kv cache. \
              If it is set to None, we will use the max sequence length.')
    parser.add_argument("--log_level", type=str, default="error")
    parser.add_argument("--engine_dir", "-i", type=str, default="trt_engines")
    parser.add_argument("--engine_name", type=str, default="whisper_tiny")
    parser.add_argument("--model_name",
                        type=str,
                        help="HuggingFace model name",
                        default="openai/whisper-tiny.en")
    parser.add_argument("--num_beams",
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument("--debug_mode",
                        help="Whether or not to turn on the debug mode",
                        action='store_true')
    parser.add_argument("--compare_hf_fp32",
                        help="Compare results with HuggingFace FP32",
                        action='store_true')
    return parser.parse_args()


class TRTLLMEncDecModel:

    def __init__(self, engine_name, engine_dir, debug_mode=False):
        # in multi-node setup, it's important to set_device at the very beginning so .to('cuda') refers to current device
        # accordingly, all input & output tensors should be moved to current device
        # otherwise, it's default to 'cuda:0'
        self.runtime_rank = tensorrt_llm.mpi_rank()
        device_id = self.runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.device = torch.cuda.current_device()

        engine_dir = Path(engine_dir)

        def engine_setup(component):
            # model config
            config_path = engine_dir / component / "config.json"
            model_config, tp_size, pp_size, gpus_per_node, dtype = read_config(
                config_path)

            # MGMN config
            world_size = tp_size * pp_size
            runtime_rank = tensorrt_llm.mpi_rank()
            assert runtime_rank < world_size, "Runtime GPU rank exceeds MPI world size. Did you launch more MPI processes than required?"
            runtime_mapping = tensorrt_llm.Mapping(world_size,
                                                   runtime_rank,
                                                   tp_size=tp_size,
                                                   pp_size=pp_size,
                                                   gpus_per_node=gpus_per_node)

            # load engine
            print(engine_name, dtype, tp_size, pp_size, runtime_rank)
        
            engine_fname = get_engine_name(engine_name, dtype, tp_size, pp_size, runtime_rank)
            with open(engine_dir / component / engine_fname, "rb") as f:
                engine_buffer = f.read()

            return model_config, runtime_mapping, engine_buffer

        # Note: encoder and decoder doesn't necessarily have the same TP & PP config
        self.encoder_model_config, self.encoder_runtime_mapping, encoder_engine_buffer = engine_setup(
            component='encoder')

        self.decoder_model_config, self.decoder_runtime_mapping, decoder_engine_buffer = engine_setup(
            component='decoder')

        # for Pipeline Parallelism in encoder
        self.nccl_comm = torch.classes.FasterTransformer.NcclCommunicatorOp(
            self.encoder_runtime_mapping.tp_size,
            self.encoder_runtime_mapping.pp_size,
            self.encoder_runtime_mapping.rank)

        # session setup
        self.encoder_session = tensorrt_llm.runtime.Session.from_serialized_engine(
            encoder_engine_buffer)
        self.decoder_session = tensorrt_llm.runtime.GenerationSession(
            self.decoder_model_config,
            decoder_engine_buffer,
            self.decoder_runtime_mapping,
            debug_mode=debug_mode)
        self.stream = torch.cuda.current_stream().cuda_stream

    @classmethod
    def from_engine(cls, engine_name, engine_dir, debug_mode=False):
        return cls(engine_name, engine_dir, debug_mode=debug_mode)
    
    def process_input(self,
                      input_ids,
                      remove_input_padding=False,
                      pad_token_id=0):
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
            input_ids = torch.cat(new_ids).unsqueeze(dim=0)  # [1, num_tokens]
        else:
            # in padding mode --> keep input, just calculate actual length and max length
            # Note: 1st token should always count, even if it is pad_token_id. e.g., decoder start id in enc-dec models could be a single pad_token_id, we should count
            input_lengths = torch.tensor(
                1 + (input_ids[:, 1:] != pad_token_id).sum(dim=1).type(
                    torch.IntTensor).to(self.device),
                dtype=torch.int32,
                device=self.device)
        max_input_length = torch.max(input_lengths).item()
        return input_ids, input_lengths, max_input_length

    def encoder_run(self,
                    input_features,
                    position_ids=None,
                    token_type_ids=None,
                    debug_mode=False):

        # each engine has hidden_dim/TP, don't forget to multiply TP
        hidden_size = self.encoder_model_config.hidden_size * self.encoder_runtime_mapping.tp_size
        hidden_states_shape = (input_features.shape[0], input_features.shape[2] // 2,
                               hidden_size)  # [1,num_tokens,D] or [BS,seqlen,D]
        hidden_states_dtype = lambda name: trt_dtype_to_torch(
            self.encoder_session.engine.get_tensor_dtype(name))

        # input tensors. only first PP rank has id input, others are hidden_states input
        inputs = {
            'input_features': input_features,
        }
        if self.encoder_model_config.has_position_embedding:
            inputs['position_ids'] = position_ids
        if self.encoder_model_config.has_token_type_embedding:
            inputs['token_type_ids'] = token_type_ids
        for k, v in inputs.items():
            self.encoder_session.context.set_input_shape(k, v.shape)
        
        batch_size = input_features.shape[0]
        # output tensors. only last PP rank final encoder output, others are intermediate hidden_states output. Need broadcast later
        outputs = {
            'encoder_output':
            torch.empty((batch_size, 1500, 384),
                        dtype=trt_dtype_to_torch(
                            self.encoder_session.engine.get_tensor_dtype(
                                'encoder_output')),
                        device='cuda')
        }

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
        # Note: runtime.Session's run() method will set input/output tensor address, here we only need to provide tensor shape
        self.encoder_session.set_shapes(inputs)
        ok = self.encoder_session.run(inputs, outputs, self.stream)
        assert ok, "Runtime execution failed"
        torch.cuda.synchronize()

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

        encoder_output = outputs['encoder_output']

        # -------------------------------------------
        if debug_mode:
            torch.cuda.synchronize()
            # use print_tensor() to print the tensors registered in the encoder network
            print("--------------------------------------")
            print("Debug output for Encoder")
            print("--------------------------------------")
            print("Registered output tensors are: ", outputs.keys())
            print_tensor('encoder_output', encoder_output)
            print("--------------------------------------")
        # -------------------------------------------

        return encoder_output

    def generate(
        self,
        input_features,
        decoder_input_ids,
        max_new_tokens,
        num_beams=1,
        pad_token_id=None,
        eos_token_id=None,
        bos_token_id=None,
        debug_mode=False,
    ):
        ## ensure all externally provided tensors are on the correct device.
        input_features = input_features.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)

        ## encoder run
        logger.info(f"Rank {self.runtime_rank} Running encoder engine ...")
        encoder_output = self.encoder_run(input_features, debug_mode=debug_mode)

        ## decoder run
        logger.info(f"Rank {self.runtime_rank} Running decoder engine ...")
        decoder_input_ids, decoder_input_lengths, decoder_max_input_length = self.process_input(
            decoder_input_ids, self.decoder_model_config.remove_input_padding,
            pad_token_id)

        # generation config
        sampling_config = SamplingConfig(end_id=eos_token_id,
                                         pad_id=pad_token_id,
                                         num_beams=num_beams,
                                         min_length=1)

        # decoder autoregressive generation
        self.decoder_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            num_beams,
            max_kv_cache_length=None,
            encoder_max_input_length=1500)
        torch.cuda.synchronize()

        encoder_input_lengths = torch.IntTensor([1500]).to('cuda:0')
        output_ids = self.decoder_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_output,
            encoder_input_lengths=encoder_input_lengths,
        )
        torch.cuda.synchronize()

        return encoder_output, output_ids
        # return encoder_output


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_arguments()
    logger.set_level(args.log_level)

    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    max_new_tokens = args.max_new_tokens
    # encoder_input = torch.ones(1,80,3000, 1).float().cuda()
    audio = whisper.load_audio("0.wav")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    input_features = whisper.log_mel_spectrogram(audio).cuda().unsqueeze(dim=0).unsqueeze(dim=-1)

    if tensorrt_llm.mpi_rank() == 0:
        print("--------------------------------------")
        print("input features: ", input_features.shape)
        print("--------------------------------------")

    model_config = AutoConfig.from_pretrained(args.model_name)

    # start_id for decoder (could add more input_ids as forced_decoder_ids)
    decoder_input_ids = torch.IntTensor([[
        model_config.forced_decoder_ids[0][0],
        model_config.forced_decoder_ids[0][1],
        model_config.decoder_start_token_id]]).to('cuda')

    # decoder_input_ids = decoder_input_ids.repeat((input_ids.shape[0], 1))

    # simple comparison with HF on FP32
    if args.compare_hf_fp32:
        if tensorrt_llm.mpi_rank() == 0:
            if "t5" in args.model_name:
                hf_model = WhisperForConditionalGeneration.from_pretrained(
                    args.model_name).to('cuda')
            else:
                pass

            tik = time.time()
            hf_output_ids = hf_model.generate(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=args.num_beams,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True)
            torch.cuda.synchronize()
            tok = time.time()

            output_ids = hf_output_ids.squeeze(dim=1)
            hf_output_text = tokenizer.batch_decode(output_ids,
                                                    skip_special_tokens=True)
            decoder_input_lengths = (decoder_input_ids !=
                                     tokenizer.pad_token_id).sum(dim=1)
            output_gen_lengths = (output_ids != tokenizer.eos_token_id).sum(
                dim=1) - decoder_input_lengths
            print("--------------------------------------")
            print("HF output_ids: ", output_ids)
            print("HF output text: ", hf_output_text)
            print("HF output generated lengths: ", output_gen_lengths)
            print(f"HF E2E time {(tok-tik)*1000}ms")
            print("--------------------------------------")

    # TRT-LLM runtime
    tllm_model = TRTLLMEncDecModel.from_engine(args.engine_name,
                                               args.engine_dir,
                                               debug_mode=args.debug_mode)
    tik = time.time()
    encoder_output, tllm_output_ids = tllm_model.generate(
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=args.num_beams,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        debug_mode=args.debug_mode,
    )
    tok = time.time()

    inference_dtype = tllm_model.encoder_model_config.dtype

    if tensorrt_llm.mpi_rank() == 0:
        output_ids = tllm_output_ids[:, 0, :]
        output_text = tokenizer.batch_decode(output_ids,
                                             skip_special_tokens=True)
        decoder_input_lengths = (decoder_input_ids !=
                                 tokenizer.pad_token_id).sum(dim=1)
        output_gen_lengths = (output_ids != tokenizer.eos_token_id).sum(
            dim=1) - decoder_input_lengths
        print("--------------------------------------")
        print("TRT-LLM output_ids: ", output_ids)
        print("TRT-LLM output text: ", output_text)
        print("TRT-LLM output generated lengths: ", output_gen_lengths)
        print(f"TRT-LLM E2E time {(tok-tik)*1000}ms")
        print("Precision:", inference_dtype)
        print("--------------------------------------")


        # only encoder test
        print("--------------------------------------")
        # print(encoder_output['encoder_output'].shape)
        # print(encoder_output['encoder_output'][0,0,:5])
        print("--------------------------------------")

        # simple accuracy check
        if args.compare_hf_fp32:
            from difflib import SequenceMatcher
            match_rate = SequenceMatcher(None, "\n".join(output_text),
                                         "\n".join(hf_output_text)).ratio()
            print(output_text)
            print(hf_output_text)
            if inference_dtype != "float32":
                print("")
                print(
                    f"[CAVEAT] Comparing TRT-LLM {inference_dtype} results with HF float32 results. Close match are not expected!"
                )
            assert match_rate > 0.9, f"Incorrect results! Match rate {match_rate}"
            print(
                f"TRT-LLM results match HF FP32 results with literal match rate {match_rate}"
            )
