from build_enc_dec_engines import Arguments, RunCMDMixin


class Run(RunCMDMixin):

    def command(self):
        args = self.args
        world_size = args.tp * args.pp
        mpi_run = f'mpirun --allow-run-as-root -np {world_size}' if world_size > 1 else ''
        ret = []
        for beam in args.beams_tuple:
            ret.append((
                mpi_run,
                f'python3 examples/models/core/enc_dec/run.py --engine_dir {args.engines_dir}',
                f'--engine_name {args.ckpt}',
                f'--model_name "{args.hf_models_dir}"',
                f'--max_new_tokens={args.max_new_tokens}',
                f'--num_beams={beam}',
                f'--compare_hf_fp32',
                f'--output_npy={args.data_dir}',
                "--debug_mode" if args.debug else "",
            ))
        ret = [' '.join(x) for x in ret]
        ret = ' && '.join(ret)
        return ret


if __name__ == '__main__':
    args = Arguments()
    Run(args).run()
