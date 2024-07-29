from build_enc_dec_engines import Arguments, RunCMDMixin


class Run(RunCMDMixin):

    def command(self):
        args = self.args
        world_size = args.tp * args.pp
        mpi_run = f'mpirun --allow-run-as-root -np {world_size} ' if world_size > 1 else ''
        ret = (
            f'python3 examples/enc_dec/run.py --engine_dir {args.engines_dir}',
            f'--engine_name {args.ckpt}',
            f'--model_name "{args.hf_models_dir}"',
            f'--max_new_tokens={args.max_new_tokens}',
            f'--num_beams={args.beams}',
            f'--compare_hf_fp32',
            f'--output_npy={args.data_dir}',
            "--debug_mode" if args.debug else "",
        )
        ret = mpi_run + ' '.join(ret)
        return ret


if __name__ == '__main__':
    args = Arguments()
    Run(args).run()
