"""Insert lightweight cross-rank barriers every Nth decoder layer.

When AutoDeploy captures all decoder layers in a single full-iteration CUDA
graph, cross-rank skew accumulates because there are no per-layer
``cudaGraphLaunch`` re-syncs (unlike the PyTorch backend which captures
many small graphs).  Inside each layer the lamport one-shot allreduce
absorbs the skew in its spin-wait, inflating the average AR latency well
above its compute cost.

This transform inserts a tiny extra ``trtllm_dist_all_reduce`` (a small
zero-valued bf16 vector — sized to satisfy the lamport-fusion kernel's
``hidden_dim % kElemsPerAccess == 0`` alignment constraint) every ``N``
post-MoE allreduce sites.  The AR output is reduced to a scalar with
``aten.sum`` and added back into the live data path via ``aten.add``
(adding the all-reduced sum of zero is a true numerical no-op) so the FX
graph stays valid and the barrier cannot be DCE'd.  Each barrier bounds
the skew that can accumulate between sync points within the single
full-iteration graph.

Activation:
    * YAML: ``transforms.insert_layer_barriers.enabled = true`` and
      ``barrier_every_n_layers: <N>``.
    * Env override: ``AD_LAYER_BARRIER_N=<N>`` (integer >= 1) overrides the
      YAML ``barrier_every_n_layers`` AND auto-enables the transform.  The
      env var takes precedence over YAML.  ``0`` or unset disables the
      override.

The transform must run AFTER ``fuse_allreduce_residual_rmsnorm`` so it
sees the fused
``dist::trtllm_fused_allreduce_residual_rmsnorm`` op (single-op end-of-
layer marker) rather than the un-fused AR + add + rmsnorm sequence.
"""

import operator
import os
from typing import List, Optional, Tuple

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)

# Buffer name registered on the GraphModule root.  A single shared zero
# bf16 buffer is reused across all barrier sites — barrier AR calls read
# it in parallel (read-only, AR is non-mutating) and never observe a data
# race.  Size is chosen so that ``hidden_dim`` (= last dim) divides the
# ``kElemsPerAccess`` of the lamport-fusion AR kernel (8 elements for
# bf16 / fp16, 4 for fp32).  See
# ``cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.h``.
_BARRIER_DUMMY_BUFFER = "_ad_layer_barrier_dummy_zero"
_BARRIER_DUMMY_NUMEL = 8


_FUSED_AR_OP = torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm


def _env_override_n() -> Optional[int]:
    """Parse ``AD_LAYER_BARRIER_N`` env var.

    Returns the override integer if set to a value ``>= 1``; ``None``
    otherwise (covers unset, ``"0"``, and malformed values).
    """
    raw = os.environ.get("AD_LAYER_BARRIER_N")
    if raw is None:
        return None
    try:
        n = int(raw.strip())
    except (TypeError, ValueError):
        return None
    if n < 1:
        return None
    return n


class InsertLayerBarriersConfig(TransformConfig):
    """Config for :class:`InsertLayerBarriers`."""

    barrier_every_n_layers: int = Field(
        default=4,
        ge=1,
        description=(
            "Insert one cross-rank allreduce barrier after every Nth post-MoE "
            "fused allreduce.  Overridable via env var ``AD_LAYER_BARRIER_N``."
        ),
    )


@TransformRegistry.register("insert_layer_barriers")
class InsertLayerBarriers(BaseTransform):
    """Insert a tiny AR barrier every N post-MoE fused-AR sites.

    Must run AFTER ``fuse_allreduce_residual_rmsnorm``.
    """

    config: InsertLayerBarriersConfig

    @classmethod
    def get_config_class(cls):
        return InsertLayerBarriersConfig

    def _post_init(self):
        # Env-var override: ``AD_LAYER_BARRIER_N=<int>`` overrides the YAML
        # ``barrier_every_n_layers`` AND auto-enables the transform.  We
        # apply this in ``_post_init`` (before the BaseTransform dispatcher
        # checks ``self.config.enabled``) because the base class short-
        # circuits the whole ``_apply`` path when ``enabled`` is ``False``.
        env_n = _env_override_n()
        if env_n is not None:
            self.config.barrier_every_n_layers = env_n
            self.config.enabled = True
            self._activated_from_env = True
        else:
            self._activated_from_env = False

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # ------------------------------------------------------------------
        # By the time we reach _apply the BaseTransform dispatcher has
        # already gated on ``self.config.enabled``; ``_post_init`` is
        # responsible for honoring the env override.
        # ------------------------------------------------------------------
        n_layers = int(self.config.barrier_every_n_layers)
        from_env = bool(getattr(self, "_activated_from_env", False))

        # ------------------------------------------------------------------
        # Multi-rank guard: barriers only make sense for TP > 1.
        # ------------------------------------------------------------------
        world_size = int(getattr(shared_config, "world_size", 1) or 1)
        if world_size < 2:
            ad_logger.info(
                f"[insert_layer_barriers] activated but world_size={world_size}; nothing to do"
            )
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # ------------------------------------------------------------------
        # Resolve AR strategy from the shared dist config (mirrors the
        # ``fuse_allreduce_residual_rmsnorm`` transform).
        # ------------------------------------------------------------------
        if shared_config.dist_config is not None:
            strategy = str(shared_config.dist_config.allreduce_strategy)
        else:
            ad_logger.warning("[insert_layer_barriers] no dist_config on shared_config; skipping")
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        if hasattr(strategy, "name"):
            strategy = strategy.name  # enum -> str just in case

        # ------------------------------------------------------------------
        # Walk the graph in topological order; collect fused-AR call nodes.
        # The fused AR alternates [post_attn, post_moe, post_attn, post_moe, ...]
        # in topological order, so post-MoE = odd index (0-based: 1, 3, 5, ...).
        # ------------------------------------------------------------------
        fused_ar_nodes: List[Node] = [n for n in gm.graph.nodes if is_op(n, _FUSED_AR_OP)]
        post_moe_ar_nodes: List[Node] = [n for i, n in enumerate(fused_ar_nodes) if i % 2 == 1]
        if not post_moe_ar_nodes:
            ad_logger.info(
                "[insert_layer_barriers] no post-MoE fused-AR nodes found "
                "(did fuse_allreduce_residual_rmsnorm run first?); skipping"
            )
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Pick every Nth post-MoE AR.  With N=4 over 36 layers this yields
        # floor(36 / 4) = 9 barriers at post-MoE indices 3, 7, 11, ... (i.e.
        # at the *end* of layer groups [0..3], [4..7], [8..11], ...).
        target_ar_nodes: List[Node] = post_moe_ar_nodes[n_layers - 1 :: n_layers]

        # ------------------------------------------------------------------
        # Lazy: register the shared dummy buffer once (1-element bf16 zero).
        # AR over this stays zero because sum-of-zeros = zero, so adding it
        # to a downstream tensor is a numerical no-op.  We allocate on the
        # rank's local CUDA device so it lives alongside the model weights
        # (post_load_fusion runs after weight_load -> tensors are on cuda).
        # ------------------------------------------------------------------
        if not hasattr(gm, _BARRIER_DUMMY_BUFFER):
            try:
                device = torch.device(f"cuda:{int(torch.cuda.current_device())}")
            except Exception:
                device = torch.device("cuda")
            dummy = torch.zeros(_BARRIER_DUMMY_NUMEL, dtype=torch.bfloat16, device=device)
            gm.register_buffer(_BARRIER_DUMMY_BUFFER, dummy, persistent=False)

        # AR op + sum + add op.  We use the trtllm op directly (the AD
        # pipeline only inserts this transform under MPI mode where TRT-LLM
        # ops are available; the world_size guard above also screens
        # single-rank).  Sum reduces the [N]-vector AR output to a scalar
        # (still zero) so the broadcast add cleanly applies to any
        # downstream tensor shape without depending on hidden_dim.
        ar_op = torch.ops.auto_deploy.trtllm_dist_all_reduce.default
        sum_op = torch.ops.aten.sum.default
        add_op = torch.ops.aten.add.Tensor

        # ------------------------------------------------------------------
        # For each target post-MoE AR, insert a barrier and tie it back
        # into the live data path via ``getitem(0) + barrier_out``.
        # ------------------------------------------------------------------
        total_barriers = 0
        for ar_node in target_ar_nodes:
            # Find getitem(0) — the normed output of the fused AR which is
            # always consumed (lm_head for the last layer, next layer's
            # first GEMM otherwise).  Hijacking this edge guarantees a
            # downstream consumer exists (unlike getitem(1) / residual,
            # which is DCE'd after the final layer).
            getitem_0: Optional[Node] = None
            for u in ar_node.users:
                if (
                    u.op == "call_function"
                    and u.target is operator.getitem
                    and len(u.args) == 2
                    and u.args[1] == 0
                ):
                    getitem_0 = u
                    break
            if getitem_0 is None:
                # No consumer of getitem(0): unusual, skip safely.
                continue
            if not getitem_0.users:
                # Dead getitem (shouldn't happen in practice); skip.
                continue

            # Snapshot consumers BEFORE we start mutating; otherwise
            # ``replace_all_uses_with`` would feed the new ``add`` back
            # into itself.
            downstream_users = list(getitem_0.users.keys())

            with gm.graph.inserting_after(getitem_0):
                dummy_attr = gm.graph.create_node("get_attr", _BARRIER_DUMMY_BUFFER)
            with gm.graph.inserting_after(dummy_attr):
                barrier_out = gm.graph.call_function(ar_op, args=(dummy_attr, strategy))
            with gm.graph.inserting_after(barrier_out):
                barrier_scalar = gm.graph.call_function(sum_op, args=(barrier_out,))
            with gm.graph.inserting_after(barrier_scalar):
                gated = gm.graph.call_function(add_op, args=(getitem_0, barrier_scalar))

            # Re-route every original consumer of getitem_0 (except our
            # newly-created ``add``) to take ``gated`` instead.
            for user in downstream_users:
                if user is gated:
                    continue
                user.replace_input_with(getitem_0, gated)

            total_barriers += 1

        ad_logger.info(
            f"[insert_layer_barriers] activated "
            f"(N={n_layers}, source={'env' if from_env else 'yaml'}); "
            f"inserting {total_barriers} barriers"
        )

        info = TransformInfo(
            skipped=(total_barriers == 0),
            num_matches=total_barriers,
            is_clean=False,
            has_valid_shapes=False,
        )
        return gm, info
