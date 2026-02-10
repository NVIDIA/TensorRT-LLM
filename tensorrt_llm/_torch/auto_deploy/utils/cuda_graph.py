from contextlib import contextmanager


class CudaGraphState:
    """A singleton class used to broadcast the state during cuda graph capture.

    Also holds shared batch info that's updated by host_prepare functions
    and read by ops during CUDA graph replay. This is needed because
    batch_info_host.tolist() returns capture-time values during replay.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if it doesn't exist
            cls._instance = super().__new__(cls)
        return cls._instance

    # Indicates the warm-up phase of cuda graph capture when
    # the graph is executed with representative inputs.
    WARM_UP: bool = False

    # Shared batch info - updated by host_prepare, read by ops during graph replay
    # These values are set by any registered host_prepare function that processes
    # batch_info_host, and can be read by ops to get current (not capture-time) values.
    _batch_info_ready: bool = False
    _num_prefill: int = 0
    _num_prefill_tokens: int = 0
    _num_decode: int = 0

    def begin_warm_up():
        if CudaGraphState.WARM_UP:
            raise ValueError("Already in a warm-up state")
        CudaGraphState.WARM_UP = True

    def end_warm_up():
        if not CudaGraphState.WARM_UP:
            raise ValueError("Not in warm-up state")
        CudaGraphState.WARM_UP = False

    def in_warm_up() -> bool:
        return CudaGraphState.WARM_UP

    @classmethod
    def set_batch_info(cls, num_prefill: int, num_prefill_tokens: int, num_decode: int) -> None:
        """Set shared batch info (called by host_prepare functions)."""
        cls._num_prefill = num_prefill
        cls._num_prefill_tokens = num_prefill_tokens
        cls._num_decode = num_decode
        cls._batch_info_ready = True

    @classmethod
    def get_batch_info(cls):
        """Get shared batch info (called by ops during graph replay).

        Returns:
            Tuple of (num_prefill, num_prefill_tokens, num_decode) if available,
            or None if host_prepare hasn't been called yet.
        """
        if cls._batch_info_ready:
            return cls._num_prefill, cls._num_prefill_tokens, cls._num_decode
        return None

    @classmethod
    def reset_batch_info(cls) -> None:
        """Reset batch info state (called at end of forward or on error)."""
        cls._batch_info_ready = False


cuda_graph_state = CudaGraphState


@contextmanager
def CudaGraphWarmUpPhase():
    cuda_graph_state.begin_warm_up()
    try:
        yield
    finally:
        cuda_graph_state.end_warm_up()
