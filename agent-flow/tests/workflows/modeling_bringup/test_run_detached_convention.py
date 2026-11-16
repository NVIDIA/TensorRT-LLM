import agent_flow.workflows.modeling_bringup.prompts as mb_prompts


def test_coder_prompt_requires_run_detached_for_long_commands():
    bundle = mb_prompts.build_modeling_bringup_prompts()
    assert "run_detached" in bundle.coder
    assert "detached" in bundle.coder.lower()


def test_run_detached_convention_present_in_slurm_mode_too():
    bundle = mb_prompts.build_modeling_bringup_prompts(include_slurm_environment=True)
    assert "run_detached" in bundle.coder
