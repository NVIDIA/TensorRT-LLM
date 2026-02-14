"""Transform for multi-stream execution of MoE layers that have shared experts and routed experts."""

from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


# Previously, CudaStreamManager and the custom ops that use the cuda streams and events were
# placed in custom_ops folder. However doing so resulted in CudaStreamManager
# being created only in the parent process, but we need each rank to have its own CudaStreamManager that
# manages the cuda streams and events for that rank. Placing the logic to instantiate
# CudaStreamManager and the custom ops that use the cuda streams and events at the transform level ensures that
# each rank has its own CudaStreamManager since each rank applies the transform independently.
class _Singleton(type):
    _instances: Dict[type, Any] = {}
    _lock = RLock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:  # double-checked locking
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# A singleton that holds the pointers to the cuda streams and events.
# Each device has its own cuda streams and events.
class CudaStreamManager(metaclass=_Singleton):
    AUX_STREAM_NAME = "aux"
    MAIN_STREAM_NAME = "main"
    devices: List[torch.device] = []
    events: Dict[torch.device, Dict[str, Any]] = {}
    streams: Dict[torch.device, Dict[str, Any]] = {}
    # Per-device save slot for the caller's stream.  ``begin_aux_stream_passthrough``
    # saves the real current stream here so that ``end_aux_stream_passthrough`` can
    # restore it — this is critical during CUDA graph capture where the capture stream
    # differs from ``torch.cuda.default_stream()``.
    _caller_streams: Dict[int, Any] = {}

    def __init__(self) -> None:
        # In case __init__ ever gets called twice, guard against re-init
        if hasattr(self, "streams"):
            return

        self._lock = RLock()
        self.add_device(torch.cuda.current_device())

    def add_device(self, device: int) -> None:
        if device not in self.devices:
            self.devices.append(device)
            with torch.cuda.device(device):
                self.events[device] = {
                    self.AUX_STREAM_NAME: torch.cuda.Event(),
                    self.MAIN_STREAM_NAME: torch.cuda.Event(),
                }
                self.streams[device] = {
                    self.AUX_STREAM_NAME: torch.cuda.Stream(),
                    self.MAIN_STREAM_NAME: torch.cuda.default_stream(),
                }
        else:
            ad_logger.warning(f"CudaStreamManager: Device {device} already added")

    def get_stream(self, device: int, stream_name: str) -> torch.cuda.Stream:
        return self.streams[device][stream_name]

    def get_event(self, device: int, event_name: str) -> torch.cuda.Event:
        return self.events[device][event_name]


# Every device will have a singleton instance of CudaStreamManager.
cuda_stream_manager = CudaStreamManager()


@torch.library.custom_op("auto_deploy::record_event", mutates_args=())
def record_event(device: int, stream_name: str) -> None:
    event = cuda_stream_manager.get_event(device, stream_name)
    event.record()


@torch.library.custom_op("auto_deploy::wait_event", mutates_args=())
def wait_event(device: int, stream_name: str) -> None:
    event = cuda_stream_manager.get_event(device, stream_name)
    event.wait()


@torch._dynamo.disable
def record_event_passthrough(
    x: torch.Tensor,
    *,
    device: int = -1,
) -> torch.Tensor:
    """Record a CUDA event on the main stream and return the input unchanged.

    Inserted after the gating/routing computation to mark a synchronization
    point.  The aux stream waits for this event before starting the MoE
    computation, enabling overlap between the shared expert (main stream)
    and routed experts (aux stream).
    """
    if device < 0:
        device = torch.cuda.current_device()
    torch.ops.auto_deploy.record_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
    return x


@torch._dynamo.disable
def aux_stream_wrapper(
    fn: Callable,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> torch.Tensor:
    stream_name = cuda_stream_manager.AUX_STREAM_NAME
    device = kwargs.pop("device", torch.cuda.current_device())
    with torch.cuda.stream(cuda_stream_manager.get_stream(device, stream_name)):
        torch.ops.auto_deploy.wait_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
        output = fn(*args, **kwargs)
        torch.ops.auto_deploy.record_event(device, cuda_stream_manager.AUX_STREAM_NAME)
    torch.ops.auto_deploy.wait_event(device, cuda_stream_manager.AUX_STREAM_NAME)
    return output


@torch._dynamo.disable
def begin_aux_stream_passthrough(
    x: torch.Tensor,
    *,
    device: int = -1,
) -> torch.Tensor:
    """Record a CUDA event on the main stream, switch to aux, and wait for it.

    After this function returns the thread-local current stream is the
    auxiliary stream.  All subsequent GPU ops dispatched by the FX graph
    interpreter will be recorded on aux until ``end_aux_stream_passthrough``
    switches back to main.
    """
    if device < 0:
        device = torch.cuda.current_device()
    # Save the *actual* current stream so ``end_aux`` can restore it.
    # During CUDA graph capture the current stream is the capture stream,
    # which is NOT ``torch.cuda.default_stream()``.
    caller_stream = torch.cuda.current_stream(device)
    cuda_stream_manager._caller_streams[device] = caller_stream
    # Record where the caller's stream has reached so aux knows when data is ready.
    main_event = cuda_stream_manager.get_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
    main_event.record(caller_stream)
    # Switch the thread-local current stream to aux.
    aux_stream = cuda_stream_manager.get_stream(device, cuda_stream_manager.AUX_STREAM_NAME)
    torch.cuda.set_stream(aux_stream)
    # Make aux wait for the main-stream event before executing any work.
    aux_stream.wait_event(main_event)
    return x


@torch._dynamo.disable
def end_aux_stream_passthrough(
    x: torch.Tensor,
    *,
    device: int = -1,
) -> torch.Tensor:
    """Record a CUDA event on the aux stream and switch back to the caller's stream.

    This does **not** make the caller's stream wait for aux.  The caller must
    insert ``wait_aux_stream_passthrough`` at the point where both branches
    need to be synchronised (typically right before the ``add`` that merges
    shared-expert and routed-expert outputs).
    """
    if device < 0:
        device = torch.cuda.current_device()
    # Record the aux-stream progress so the caller's stream can wait for it later.
    aux_event = cuda_stream_manager.get_event(device, cuda_stream_manager.AUX_STREAM_NAME)
    aux_event.record()
    # Restore the caller's stream saved by ``begin_aux_stream_passthrough``.
    # This is critical during CUDA graph capture where the capture stream
    # differs from ``torch.cuda.default_stream()``.
    caller_stream = cuda_stream_manager._caller_streams.pop(device, None)
    if caller_stream is not None:
        torch.cuda.set_stream(caller_stream)
    else:
        torch.cuda.set_stream(
            cuda_stream_manager.get_stream(device, cuda_stream_manager.MAIN_STREAM_NAME)
        )
    return x


@torch._dynamo.disable
def wait_aux_stream_passthrough(
    x: torch.Tensor,
    *,
    device: int = -1,
) -> torch.Tensor:
    """Make the current stream wait for the auxiliary stream's last recorded event.

    This is a GPU-side wait (non-blocking on the CPU).  Insert this right
    before the ``add`` that merges shared-expert output (computed on aux)
    with routed-expert output (computed on main).

    Uses ``torch.cuda.current_stream()`` rather than the stored default stream
    so that the correct stream is waited on during CUDA graph capture.
    """
    if device < 0:
        device = torch.cuda.current_device()
    aux_event = cuda_stream_manager.get_event(device, cuda_stream_manager.AUX_STREAM_NAME)
    torch.cuda.current_stream(device).wait_event(aux_event)
    return x


def _make_aux_stream_impl(base_overload: Callable) -> Callable:
    """Build an implementation that runs *base_overload* on the auxiliary CUDA stream."""

    def _impl(*args, **kwargs):
        device = torch.cuda.current_device()
        with torch.cuda.stream(
            cuda_stream_manager.get_stream(device, cuda_stream_manager.AUX_STREAM_NAME)
        ):
            torch.ops.auto_deploy.wait_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
            output = base_overload(*args, **kwargs)
            torch.ops.auto_deploy.record_event(device, cuda_stream_manager.AUX_STREAM_NAME)
        torch.ops.auto_deploy.wait_event(device, cuda_stream_manager.AUX_STREAM_NAME)
        return output

    return _impl


def _find_merge_add(moe_node: Node) -> Optional[Node]:
    """Walk forward from a MoE op through users to find the ``aten.add.Tensor`` merge node.

    The merge ``add`` is the node where shared-expert output and routed-expert
    output are combined.  The search is a breadth-first traversal of the user
    graph starting from the MoE node.
    """
    visited: Set[Node] = set()
    queue = list(moe_node.users.keys())
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue
        visited.add(n)
        if is_op(n, torch.ops.aten.add.Tensor):
            return n
        queue.extend(n.users.keys())
    return None


def _get_ancestors(node: Node) -> Set[Node]:
    """Return the set of all nodes reachable by walking backwards from *node*."""
    ancestors: Set[Node] = set()
    queue = list(node.all_input_nodes)
    while queue:
        n = queue.pop()
        if n in ancestors:
            continue
        ancestors.add(n)
        queue.extend(n.all_input_nodes)
    return ancestors


def _execute_shared_expert_in_aux_stream(
    gm: GraphModule, moe_ops: List[Callable]
) -> Tuple[GraphModule, int]:
    """Move shared-expert computation to the auxiliary CUDA stream.

    For each MoE fused op in the graph:
      1. Walk forward to find the ``aten.add.Tensor`` that merges the
         shared-expert output and the routed-expert output.
      2. Identify which ``add`` input is the routed branch (descended from
         the MoE node) and which is the shared-expert branch.
      3. Trace the shared-expert branch backwards to collect all its
         computation nodes and identify the fork point (the latest common
         ancestor shared with the MoE / routing path).
      4. Insert ``begin_aux_stream_passthrough`` before the first shared-expert
         op to switch to the auxiliary CUDA stream.
      5. Insert ``end_aux_stream_passthrough`` after the last shared-expert op
         to switch back to the main stream.
      6. Insert ``wait_aux_stream_passthrough`` on the routed-branch input
         just before the ``add`` so the main stream waits for the auxiliary
         stream to finish before merging outputs.
    """
    graph = gm.graph
    num_replaced = 0

    # Collect targets first to avoid mutating while iterating.
    target_nodes = [n for n in graph.nodes if is_op(n, moe_ops)]
    if not target_nodes:
        return gm, 0

    node_order = {node: i for i, node in enumerate(graph.nodes)}

    for moe_node in target_nodes:
        # ---- Step 1: Find the merge ``add`` node. ----
        add_node = _find_merge_add(moe_node)
        if add_node is None:
            ad_logger.warning(
                f"No merge add found downstream of MoE node {moe_node.name}; "
                "skipping multi-stream transform for this node."
            )
            continue

        # ---- Step 2: Determine which ``add`` input is routed vs. shared. ----
        arg0, arg1 = add_node.args[0], add_node.args[1]
        arg0_ancestors = _get_ancestors(arg0)

        if moe_node in arg0_ancestors or arg0 is moe_node:
            routed_output, shared_output = arg0, arg1
        else:
            routed_output, shared_output = arg1, arg0

        # ---- Step 3: Collect shared-expert nodes & find fork point. ----
        moe_ancestors = _get_ancestors(moe_node)
        moe_ancestors.add(moe_node)

        shared_nodes: List[Node] = []
        fork_point: Optional[Node] = None
        visited: Set[Node] = set()
        queue = [shared_output]

        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)

            # Skip static weight / parameter nodes.
            if n.op == "get_attr":
                continue

            if n in moe_ancestors:
                # This node is on the MoE / routing path — candidate fork point.
                if fork_point is None or node_order.get(n, 0) > node_order.get(fork_point, 0):
                    fork_point = n
                continue

            shared_nodes.append(n)
            for inp in n.all_input_nodes:
                queue.append(inp)

        if not shared_nodes or fork_point is None:
            ad_logger.warning(
                f"Could not identify shared-expert subgraph for MoE node "
                f"{moe_node.name}; skipping multi-stream transform for this node."
            )
            continue

        # Order shared nodes by their position in the graph.
        shared_nodes.sort(key=lambda n: node_order.get(n, 0))
        first_shared = shared_nodes[0]

        # Sanity check: the first shared op must directly consume the fork
        # point so we can wire begin_aux_stream_passthrough into it.
        if fork_point not in first_shared.all_input_nodes:
            ad_logger.warning(
                f"First shared-expert op ({first_shared.name}) does not directly "
                f"consume fork point ({fork_point.name}); skipping."
            )
            continue

        # ---- Step 4: Insert begin_aux before the first shared-expert op. ----
        # NOTE: do NOT bake ``torch.cuda.current_device()`` into the graph —
        # that would hard-code device 0 and break on other ranks in a
        # multi-GPU setup.  Omitting ``device`` lets the passthrough
        # functions resolve the device at **runtime** (default ``-1``).
        with graph.inserting_before(first_shared):
            begin_aux_node = graph.call_function(
                begin_aux_stream_passthrough,
                args=(fork_point,),
            )

        # Create a data dependency: first_shared reads begin_aux output
        # instead of fork_point.
        first_shared.args = tuple(
            begin_aux_node if arg is fork_point else arg for arg in first_shared.args
        )

        # ---- Step 5: Insert end_aux after the last shared-expert op. ----
        with graph.inserting_after(shared_output):
            end_aux_node = graph.call_function(
                end_aux_stream_passthrough,
                args=(shared_output,),
            )

        # Replace shared-expert input to ``add`` with end_aux output.
        add_node.args = tuple(
            end_aux_node if arg is shared_output else arg for arg in add_node.args
        )

        # ---- Step 6: Insert wait_aux before the ``add``. ----
        with graph.inserting_before(add_node):
            wait_aux_node = graph.call_function(
                wait_aux_stream_passthrough,
                args=(routed_output,),
            )

        add_node.args = tuple(
            wait_aux_node if arg is routed_output else arg for arg in add_node.args
        )

        num_replaced += 1

    return gm, num_replaced


@TransformRegistry.register("multi_stream_moe")
class MultiStreamMOE(BaseTransform):
    """Multi-stream execution of MoE layers that have shared experts and routed experts."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        base_ops = [
            torch.ops.auto_deploy.trtllm_moe_fused,
            torch.ops.auto_deploy.triton_moe_fused,
            torch.ops.auto_deploy.trtllm_quant_fp8_moe_fused,
            torch.ops.auto_deploy.trtllm_quant_nvfp4_moe_fused,
        ]

        # Ensure that aux stream and events for the current device are added to the CudaStreamManager.
        cuda_stream_manager.add_device(torch.cuda.current_device())
        gm, num_matches = _execute_shared_expert_in_aux_stream(gm, base_ops)

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        with open("/tmp/after_multi_stream_moe.txt", "w") as f:
            f.write(str(gm.graph))
        return gm, info
