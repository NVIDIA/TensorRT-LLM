def compression_manager_class(algorithm: str):
    """Return the compression-manager class registered for ``algorithm``.

    Mirrors the dispatch in ``_util.create_kv_cache_compression_manager``
    without building a manager, for callers that only need class-level
    declarations (e.g. ``physically_evicts_cached_tokens`` in the attention
    RoPE gate). Returns None for unknown algorithms; the factory is the one
    that rejects them.
    """
    if algorithm == "triattention":
        from .triattention import TriAttention

        return TriAttention
    return None
