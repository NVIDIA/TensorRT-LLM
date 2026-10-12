# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank agreement on decoder warmup shapes and phases."""

from collections.abc import Callable

from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import global_mpi_rank
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


class WarmupAgreement:
    """Keeps every rank of a warmup forward on the same shapes and phases."""

    def __init__(self, dist: Distributed | None, mapping: Mapping) -> None:
        self.dist = dist
        self.mapping = mapping

    def is_distributed_forward(self) -> bool:
        """Return whether model forward can communicate with peer workers.

        ``dist`` is optional. An engine built without a communicator cannot
        enter a collective at all, so it has no peers to strand and every
        warmup failure stays rank-local.
        """
        if self.dist is None:
            return False
        return self.dist.world_size > 1 or self.mapping.dwdp_enabled

    def allgather(self) -> Callable[[int], list[int]] | None:
        """Return an allgather over the ranks this warmup forward synchronizes with.

        ``None`` means agreement cannot be established here, so a missing batch
        stays fatal rather than being skipped unilaterally.

        DWDP is the case that cannot be answered: its peers are reached through
        a ``COMM_WORLD``-derived subgroup built in ``dwdp.py``, not through
        ``self.dist``, so a ``self.dist`` allgather would report a unanimity it
        never observed.
        """
        if self.dist is None or self.mapping.dwdp_enabled:
            return None
        if self.dist.world_size <= 1:
            return None
        return self.dist.allgather

    def agree_flag(self, flag: bool) -> bool:
        """Reduce a phase-entry decision to one the whole forward group shares.

        Several predicates that gate a warmup phase are rank-local. The
        capturable guided decoder is installed only on the last pipeline rank,
        so ``guided_decoder is None`` -- and through it
        ``can_run_general_warmup`` -- differs across pipeline stages. Mamba's
        entry test reads this rank's free KV capacity. Letting one rank enter a
        phase its peers skip strands whoever enters the collective, and it also
        unbalances the per-shape agreement below.

        Any rank opting out takes the whole group out with it.
        """
        allgather = self.allgather()
        if allgather is None:
            return flag
        return all(allgather(int(flag)))

    def agree_shapes(self, configs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Reduce warmup shapes to the ones every rank in the group proposed.

        ``_get_max_shape_warmup_requests`` derives shapes from this rank's free
        KV capacity, so under attention-DP the values -- and, once
        ``dict.fromkeys`` drops a collision, the length -- differ per rank.
        Ranks would then walk different loops and meet in different forwards.

        The intersection keeps rank 0's ordering, which matters because the
        general warmup list is ordered for torch.compile specialization.
        """
        allgather = self.allgather()
        if allgather is None:
            return configs
        per_rank = allgather([list(config) for config in configs])
        shared = set.intersection(
            *({tuple(config) for config in rank_configs} for rank_configs in per_rank)
        )
        agreed = [config for config in configs if config in shared]
        dropped = [config for config in configs if config not in shared]
        if dropped:
            logger.warning(
                f"Dropping warmup shapes {dropped} that not every rank could "
                f"propose; per-rank KV capacity differs. Remaining: {agreed}."
            )
        return agreed

    def should_run_warmup_batch(
        self, batch: ScheduledRequests | None, num_tokens: int, shape: str
    ) -> bool:
        """Decide whether this warmup shape runs, is skipped, or fails the rank.

        A rank that skips a shape its peers run leaves them blocked in that
        forward's collectives for the rest of the job. Skipping is therefore
        safe exactly when every rank in the forward group skips too, and an
        allgather establishes that before any rank enters the forward.

        The plan agreed by ``_agree_warmup_plan`` is what makes that allgather
        safe: every rank walks the same shape list, so this runs the same
        number of times everywhere.
        """
        if not self.is_distributed_forward():
            if batch is not None:
                return True
            # Safe to skip, but never silently: a skip during KV cache
            # estimation makes the profiling peak unrepresentative of
            # this shape.
            logger.warning(f"Skipping warmup shape ({shape}): not enough KV cache space.")
            return False

        allgather = self.allgather()
        if allgather is None:
            # No reachable group to agree with. Keep the TP-only check, which
            # still catches the attention-DP asymmetry it was written for.
            self.assert_all_tp_ranks_have_warmup_batch(batch, num_tokens)
            if batch is None:
                raise RuntimeError(
                    f"Warmup batch creation failed for shape ({shape}) on "
                    f"global_rank={global_mpi_rank()}, "
                    f"model_rank={self.dist.rank}, and this topology offers no "
                    f"way to confirm that peers are skipping it too. They may "
                    f"already be inside the matching forward, so this rank "
                    f"cannot skip the shape without stranding them."
                )
            return True

        flags = list(allgather(int(batch is not None)))
        if all(flags):
            return True
        if not any(flags):
            # Every rank in the forward group is skipping, so none of them is
            # left inside a collective. This is the ordinary outcome for a
            # shape that does not fit the configuration at all, such as a
            # mixed context+generation shape under ``max_batch_size=1``.
            logger.info(
                f"Skipping warmup shape ({shape}) on all "
                f"{len(flags)} ranks: not enough KV cache space."
            )
            return False

        all_tokens = list(allgather(num_tokens))
        failed_ranks = [i for i, flag in enumerate(flags) if not flag]
        raise RuntimeError(
            f"Warmup batch creation failed for shape ({shape}) on rank(s) "
            f"{failed_ranks} but succeeded on others, so entering this forward "
            f"would deadlock the ranks that still hold a batch. Per-rank "
            f"curr_max_num_tokens: {all_tokens}. This indicates asymmetric KV "
            f"cache capacity across ranks. Consider increasing "
            f"--kv_cache_free_gpu_mem_fraction."
        )

    def assert_all_tp_ranks_have_warmup_batch(self, batch, num_tokens: int) -> None:
        """Assert every TP rank has a valid warmup batch, or raise with diagnostics.

        Under attention-DP, each rank's KV cache available capacity can differ at
        runtime, causing _create_warmup_request to return None on some ranks while
        others proceed into forward() with tp_comm collectives — deadlocking the
        job. This check prevents the deadlock by failing early with diagnostic info.

        ``tp_size`` alone does not establish that peers are reachable: ``dist``
        is optional, and without a communicator there is no tp_comm collective
        to deadlock in.
        """
        if self.mapping.tp_size <= 1 or self.dist is None:
            return
        has_batch = int(batch is not None)
        all_flags = list(self.dist.tp_allgather(has_batch))
        if any(all_flags) and not all(all_flags):
            # Gather token counts for diagnostics
            all_tokens = list(self.dist.tp_allgather(num_tokens))
            failed_ranks = [i for i, f in enumerate(all_flags) if not f]
            raise RuntimeError(
                f"Warmup batch creation failed on TP rank(s) {failed_ranks} "
                f"but succeeded on others. This would cause a collective "
                f"deadlock. Per-rank curr_max_num_tokens: {all_tokens}. "
                f"This indicates asymmetric KV cache capacity across TP ranks. "
                f"Consider increasing --kv_cache_free_gpu_mem_fraction."
            )
