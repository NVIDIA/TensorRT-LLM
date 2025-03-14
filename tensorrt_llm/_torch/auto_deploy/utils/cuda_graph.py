from contextlib import contextmanager


class CudaGraphState:
    """A singleton class used to broadcast the state during cuda graph capture."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if it doesn't exist
            cls._instance = super().__new__(cls)
        return cls._instance

    # Indicates the warm-up phase of cuda graph capture when
    # the graph is executed with representative inputs.
    WARM_UP: bool = False

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


cuda_graph_state = CudaGraphState


@contextmanager
def CudaGraphWarmUpPhase():
    cuda_graph_state.begin_warm_up()
    try:
        yield
    finally:
        cuda_graph_state.end_warm_up()
