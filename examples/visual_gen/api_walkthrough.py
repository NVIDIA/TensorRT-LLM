### :title API walkthrough
### :order 0
from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm.visual_gen.args import CompilationConfig


def main():
    # 1. List supported models registered with the pipeline registry.
    print("\n=== Supported models ===")
    for hf_id in VisualGen.supported_models():
        print(f"  - {hf_id}")

    # 2. Inspect default pipeline_config knobs for the chosen model. These
    #    are per-architecture runtime knobs (e.g. Lightricks/LTX-2's
    #    ``text_encoder_path``); Wan-AI/Wan2.1-T2V-1.3B-Diffusers registers
    #    none, so the dict is empty.
    pipeline_defaults = VisualGen.pipeline_config("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    print("\n=== Pipeline config defaults for Wan-AI/Wan2.1-T2V-1.3B-Diffusers ===")
    print(f"  {pipeline_defaults or '(none)'}")

    # 3. Build VisualGenArgs. ``pipeline_config`` carries the per-architecture
    #    knobs from step 2 (here we just forward the registered defaults;
    #    real callers would override entries like ``text_encoder_path``).
    #    ``compilation_config.skip_warmup`` skips the post-load warmup pass.
    visual_gen = VisualGen(
        model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        args=VisualGenArgs(
            pipeline_config=pipeline_defaults,
            compilation_config=CompilationConfig(skip_warmup=True),
        ),
    )

    # 4. Discover model-specific ``extra_params`` accepted by the loaded
    #    pipeline. Wan-AI/Wan2.1-T2V-1.3B-Diffusers declares none;
    #    Wan-AI/Wan2.2-T2V-A14B-Diffusers surfaces ``guidance_scale_2`` and
    #    ``boundary_ratio`` here.
    specs = visual_gen.extra_param_specs
    print("\n=== Extra param specs (extra_params keys) ===")
    for name, spec in specs.items():
        print(f"  - {name}: {spec}")
    if not specs:
        print("  (none for this model)")

    # 5. Take the pipeline's resolved defaults (height/width/steps/etc.)
    #    and override fields. ``default_params`` already pre-populates
    #    ``params.extra_params`` with each declared spec's default, so the
    #    override below shows how a caller would set a model-specific knob
    #    -- no-op on Wan-AI/Wan2.1-T2V-1.3B-Diffusers, but the wiring is
    #    the same on Wan-AI/Wan2.2-T2V-A14B-Diffusers where
    #    ``extra_params["guidance_scale_2"]`` is honored.
    params = visual_gen.default_params
    # Wan requires num_frames of the form 4k+1; 1.25x the model default (81)
    # is 101.25, so we round to the nearest valid value, 101 (= 4*25 + 1).
    params.num_frames = 101
    for name, spec in specs.items():
        params.extra_params[name] = spec.default

    print("\n=== Request params ===")
    print(params.model_dump_json(indent=2))

    output = visual_gen.generate(inputs="A cute cat playing piano in a sunny room", params=params)

    # 6. Persist to disk. ``save`` infers the container from the file
    #    extension (.avi/.mp4) and uses the frame_rate carried on the
    #    output.
    saved = output.save("api_walkthrough_output.avi")
    print(f"\nSaved: {saved}")


if __name__ == "__main__":
    main()
