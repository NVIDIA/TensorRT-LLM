def suggest_spec_config(max_batch_size: int) -> "DecodingBaseConfig":
    """
    Suggests a reasonable draft model free speculation scheme.
    Used when the user specifies spec_mode == AUTO.

    For now, we always use an ngram scheme that gets disabled at
    BS>=32.
    """
    from tensorrt_llm.llmapi.llm_args import NGramDecodingConfig
    return NGramDecodingConfig(
        max_draft_len=5 if max_batch_size <= 4 else 3,
        max_matching_ngram_size=3 if max_batch_size <= 4 else 5,
        max_concurrency=32,
        is_keep_all=True,
        is_use_oldest=True,
        is_public_pool=True,
    )
