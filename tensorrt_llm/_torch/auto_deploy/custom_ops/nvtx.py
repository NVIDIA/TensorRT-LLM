"""Contains custom NVTX operations for profiling."""

import nvtx
from torch.library import custom_op, register_fake


class NVTXState:
    """A singleton class used to maintain the nvtx ranges."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if it doesn't exist
            cls._instance = super().__new__(cls)
        return cls._instance

    ranges: dict[str, nvtx._lib.lib.RangeId] = {}

    def add_range(self, name: str, range_id: nvtx._lib.lib.RangeId) -> None:
        """Add a range to the state.

        Args:
            name: The name of the range
            range_id: The NVTX range ID to store
        """
        assert name not in self.ranges, f"Range {name} already exists"
        self.ranges[name] = range_id

    def get_range(self, name: str) -> nvtx._lib.lib.RangeId:
        """Get a range from the state.

        Args:
            name: The name of the range to retrieve

        Returns:
            The NVTX range ID for the given name
        """
        assert name in self.ranges, f"Range {name} does not exist"
        id = self.ranges[name]
        del self.ranges[name]
        return id

    def reset(self) -> None:
        """Clear all stored ranges."""
        self.ranges.clear()


nvtx_state = NVTXState()


@custom_op("nvtx_ops::start_range", mutates_args=())
def start_range(name: str) -> None:
    """Start an NVTX range with the given name."""
    nvtx_state.add_range(name, nvtx.start_range(name))


@custom_op("nvtx_ops::end_range", mutates_args=())
def end_range(name: str) -> None:
    """End the current NVTX range."""
    nvtx.end_range(nvtx_state.get_range(name))


# Register fake implementation for tracing
@register_fake("nvtx_ops::start_range")
def start_range_fake(name: str) -> None:
    """Fake implementation for tracing."""
    pass


@register_fake("nvtx_ops::end_range")
def end_range_fake(name: str) -> None:
    """Fake implementation for tracing."""
    pass
