import os.path
from argparse import ArgumentParser
from dataclasses import dataclass, fields
from subprocess import run
from sys import stderr, stdout
from typing import List, Literal, Union

split = os.path.split
join = os.path.join
dirname = os.path.dirname


@dataclass
class Arguments:
    download: bool = False
    dtype: Literal['float16', 'float32', 'bfloat16'] = 'float16'

    hf_repo_name: Literal[
        'facebook/bart-large-cnn', 't5-small',
        'language_adapter-enc_dec_language_adapter'] = 'facebook/bart-large-cnn'

    model_cache: str = '/llm-models'

    tp: int = 1
    pp: int = 1

    beams: str = '1'
    gpus_per_node: int = 4
    debug: bool = False

    rm_pad: bool = True
    gemm: bool = True

    max_new_tokens: int = 64

    @property
    def beams_tuple(self):
        return eval(f'tuple([{self.beams}])')

    @property
    def max_beam(self):
        return max(self.beams_tuple)

    @property
    def ckpt(self):
        return self.hf_repo_name.split('/')[-1]

    @property
    def base_dir(self):
        return dirname(dirname(__file__))

    @property
    def data_dir(self):
        return join(self.base_dir, 'data/enc_dec')

    @property
    def models_dir(self):
        return join(self.base_dir, 'models/enc_dec')

    @property
    def hf_models_dir(self):
        return join(self.model_cache, self.ckpt)

    @property
    def trt_models_dir(self):
        return join(self.models_dir, 'trt_models', self.ckpt)

    @property
    def engines_dir(self):
        return join(self.models_dir, 'trt_engines', self.ckpt,
                    f'{self.tp * self.pp}-gpu', self.dtype)

    @property
    def model_type(self):
        return self.ckpt.split('-')[0]

    def __post_init__(self):
        parser = ArgumentParser()
        for k in fields(self):
            k = k.name
            v = getattr(self, k)
            if isinstance(v, bool):
                parser.add_argument(f'--{k}', action='store_true')
            else:
                parser.add_argument(f'--{k}', default=v, type=type(v))

        args = parser.parse_args()
        for k, v in args._get_kwargs():
            setattr(self, k, v)


@dataclass
class RunCMDMixin:
    args: Arguments

    def command(self) -> Union[str, List[str]]:
        raise NotImplementedError

    def run(self):
        cmd = self.command()
        if cmd:
            cmd = ' '.join(cmd) if isinstance(cmd, list) else cmd
            print('+ ' + cmd)
            run(cmd, shell='bash', stdout=stdout, stderr=stderr, check=True)


class DownloadHF(RunCMDMixin):

    def command(self):
        args = self.args
        return [
            'git', 'clone', f'https://huggingface.co/{args.hf_repo_name}',
            args.hf_models_dir
        ] if args.download and args.model_type != 'language_adapter' else ''


class Convert(RunCMDMixin):

    def command(self):
        args = self.args
        return [
            f'python examples/models/core/enc_dec/convert_checkpoint.py',
            f'--model_type {args.model_type}',
            f'--model_dir {args.hf_models_dir}',
            f'--output_dir {args.trt_models_dir}',
            f'--tp_size {args.tp} --pp_size {args.pp}'
        ]


class Build(RunCMDMixin):

    def command(self):
        args = self.args
        engine_dir = args.engines_dir
        weight_dir = args.trt_models_dir
        encoder_build = [
            f"trtllm-build --checkpoint_dir {join(weight_dir, 'encoder')}",
            f"--output_dir {join(engine_dir, 'encoder')}",
            f'--paged_kv_cache disable',
            f'--max_beam_width {args.max_beam}',
            f'--max_batch_size 8',
            f'--max_input_len 512',
            f'--gemm_plugin {args.dtype}',
            f'--bert_attention_plugin {args.dtype}',
            f'--gpt_attention_plugin {args.dtype}',
            f'--remove_input_padding enable',
        ]

        decoder_build = [
            f"trtllm-build --checkpoint_dir {join(weight_dir, 'decoder')}",
            f"--output_dir {join(engine_dir, 'decoder')}",
            f'--paged_kv_cache enable',
            f'--max_beam_width {args.max_beam}',
            f'--max_batch_size 8',
            f'--max_seq_len 201',
            f'--max_encoder_input_len 512',
            f'--gemm_plugin {args.dtype}',
            f'--bert_attention_plugin {args.dtype}',
            f'--gpt_attention_plugin {args.dtype}',
            f'--remove_input_padding enable',
            '--max_input_len 1',
        ]

        # t5 model with relative attention cannot use context_fmha
        encoder_build.append(f'--context_fmha disable')
        decoder_build.append(f'--context_fmha disable')

        # language adapter plugin leverages MOE plugin for static expert selection
        if args.model_type == 'language_adapter':
            encoder_build.append(f'--moe_plugin auto')
            decoder_build.append(f'--moe_plugin auto')
        else:
            encoder_build.append(f'--moe_plugin disable')
            decoder_build.append(f'--moe_plugin disable')

        encoder_build = ' '.join(encoder_build)
        decoder_build = ' '.join(decoder_build)
        ret = ' && '.join((encoder_build, decoder_build))
        return ret


if __name__ == "__main__":
    # TODO: add support for more models / setup
    args = Arguments()
    DownloadHF(args).run()
    Convert(args).run()
    Build(args).run()
