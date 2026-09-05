# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Captured-schedule helper functions for MLA decode TS examples."""

from contextlib import contextmanager

from cutlass.experimental.task_scheduling.schedule_builder import work_tile_loop

from .stage import MlaStage


def captured_loop_bounds(task_kwargs, default_start: int, default_step: int = 1):
    """Consume Task loop kwargs and return captured schedule loop bounds."""
    loop_start = task_kwargs.pop("domain_start", default_start)
    loop_step = task_kwargs.pop("step", default_step)
    loop_end = task_kwargs.pop("domain", None)
    if loop_end is None:
        loop_end = 1
    return loop_start, loop_end, loop_step


def work_queue_tail(work_queue, *, advance_label: str = "get_and_advance_work_tile"):
    """Advance the persistent work tile at the end of a task body."""
    if work_queue is None:
        return
    work_queue.wait()
    getattr(work_queue, advance_label)()
    work_queue.release()


def runtime_work_tile_skip_if(work_queue):
    """Return a concrete work queue's public runtime-skip predicate."""
    if work_queue is None or not getattr(work_queue, "enable_runtime_skip", False):
        return None
    return getattr(type(work_queue), "skip_work_tile_if", None)


@contextmanager
def work_tile_schedule_loop(
    work_queue,
    *,
    skip_if=None,
    non_skippable_prelude=None,
):
    """Wrap a captured task body in the standard TS persistent work-tile loop.

    A decode work queue may expose ``skip_work_tile_if`` for runtime-padded
    tiles. Only task data work is skippable; queue wait/advance/release remains
    outside the guard so every persistent task advances in lockstep. Pure
    register initializers may run in ``non_skippable_prelude`` so their values
    dominate the stock executor's separately guarded HEAD, LOOP, and TAIL
    regions. The prelude must not issue memory or pipeline operations.
    """
    if work_queue is not None:
        with work_tile_loop(work_queue, skip_if=skip_if) as work_tile:
            prelude_state = (
                non_skippable_prelude() if non_skippable_prelude is not None else None
            )
            if skip_if is None:
                yield work_tile, prelude_state
            else:
                with work_tile.skippable():
                    yield work_tile, prelude_state
            work_queue_tail(work_queue)
    else:
        yield None, None


def page_offsets_consume(smem_page_offsets, cached_page_ids=None):
    """Consume staged page offsets for the next K/V TMA transfer."""
    if smem_page_offsets is None:
        return None
    smem_page_offsets.wait()
    return smem_page_offsets.read_offsets(cached_page_ids=cached_page_ids)


def page_offsets_release(smem_page_offsets):
    """Release the page-offset stage after the K/V load has consumed it."""
    if smem_page_offsets is None:
        return
    smem_page_offsets.release()


def page_offsets_produce(smem_page_offsets, label, *, section: MlaStage):
    """Produce one page-offset stage using the named K/V slot callback."""
    smem_page_offsets.acquire()
    getattr(smem_page_offsets, label)(section=section)
    smem_page_offsets.commit()


def schedule_token_throttle_head(schedule_token_throttle):
    """Publish a load-task schedule token before issuing staged loads."""
    if schedule_token_throttle is None:
        return
    schedule_token_throttle.acquire()
    schedule_token_throttle.publish_schedule_token()
    schedule_token_throttle.commit()


def schedule_token_throttle_tail(schedule_token_throttle):
    """Consume a schedule token before the scheduler fetches more work."""
    if schedule_token_throttle is None:
        return
    schedule_token_throttle.wait()
    schedule_token_throttle.consume_schedule_token()
    schedule_token_throttle.release()


def staged_kv_tma_load(
    smem_kv,
    iterations: int,
    cached_k_pages,
    cached_v_pages,
    cached_next_v_pages,
    *,
    is_v: bool,
    use_next_v_pages: bool = False,
):
    """Produce staged K or delayed-V TMA work for one logical k-tile."""
    for subtile_idx in range(iterations):
        smem_kv.acquire()
        smem_kv.tma_load(
            cached_k_pages=cached_k_pages,
            cached_v_pages=cached_v_pages,
            cached_next_v_pages=cached_next_v_pages,
            is_v=is_v,
            subtile_idx=subtile_idx,
            use_next_v_pages=use_next_v_pages,
        )
        smem_kv.commit()


def staged_kv_load(
    smem_kv,
    *,
    head_dim_stages,
    producer_label,
    section: MlaStage,
    smem_page_offsets=None,
    cached_page_ids=None,
):
    """Issue all K/V producer stages for one logical K or V tile."""
    cached_page_ids = page_offsets_consume(smem_page_offsets, cached_page_ids)
    for stage_idx in range(head_dim_stages):
        smem_kv.acquire()
        getattr(smem_kv, producer_label)(
            stage_idx=stage_idx,
            section=section,
            cached_page_ids=cached_page_ids,
        )
        smem_kv.commit()
    page_offsets_release(smem_page_offsets)
    return cached_page_ids


def staged_qk_mma(
    smem_kv,
    tmem_s,
    iterations=None,
    *,
    head_dim_stages=None,
    consumer_label=None,
    include_acquire=True,
):
    """Consume staged K descriptors and produce one QK MMA score tile."""
    if consumer_label is None:
        for k_subtile_idx in range(iterations):
            smem_kv.wait()
            desc_k_base = smem_kv.k_desc(k_subtile_idx=k_subtile_idx)
            if k_subtile_idx == 0:
                tmem_s.acquire()
            tmem_s.qk_mma(desc_k_base=desc_k_base, k_subtile_idx=k_subtile_idx)
            smem_kv.release()
        tmem_s.commit()
        return

    if include_acquire:
        tmem_s.acquire()
    for k_subtile_idx in range(head_dim_stages):
        smem_kv.wait()
        kv_desc = getattr(smem_kv, consumer_label)(k_subtile_idx=k_subtile_idx)
        tmem_s.qk_mma(kv_desc=kv_desc, k_subtile_idx=k_subtile_idx)
        smem_kv.release()
    tmem_s.commit()


def staged_pv_mma(
    smem_kv,
    smem_p,
    tmem_o,
    iterations=None,
    *,
    head_dim_stages=None,
    consumer_label=None,
    producer_label=None,
    is_tail: bool = False,
):
    """Consume staged V descriptors and produce one PV MMA output tile."""
    smem_p.wait()
    if producer_label is None:
        desc_p_base = smem_p.p_desc()
        tmem_o.acquire()
        for v_subtile_idx in range(iterations):
            smem_kv.wait()
            desc_v_base = smem_kv.v_desc(v_subtile_idx=v_subtile_idx)
            tmem_o.pv_mma(
                desc_p_base=desc_p_base,
                desc_v_base=desc_v_base,
                v_subtile_idx=v_subtile_idx,
                is_tail=is_tail,
            )
            smem_kv.release()
        tmem_o.commit()
        smem_p.release()
        return

    smem_p.p_desc()
    tmem_o.acquire()
    is_tail = is_tail or (
        producer_label is not None and producer_label.find("tail") >= 0
    )
    for v_subtile_idx in range(head_dim_stages):
        smem_kv.wait()
        if producer_label.endswith("_0"):
            v_desc_0 = getattr(smem_kv, consumer_label)(v_subtile_idx=v_subtile_idx)
            getattr(tmem_o, producer_label)(
                v_desc_0=v_desc_0,
                v_subtile_idx=v_subtile_idx,
                is_tail=is_tail,
            )
        else:
            v_desc_1 = getattr(smem_kv, consumer_label)(v_subtile_idx=v_subtile_idx)
            getattr(tmem_o, producer_label)(
                v_desc_1=v_desc_1,
                v_subtile_idx=v_subtile_idx,
                is_tail=is_tail,
            )
        smem_kv.release()
    tmem_o.commit()
    smem_p.release()


def staged_qk_mma_k_tile(smem_k, tmem_s, iterations: int):
    """Consume one whole K stage and produce one QK score tile."""
    smem_k.wait()
    tmem_s.acquire()
    for k_subtile_idx in range(iterations):
        desc_k_base = smem_k.k_desc(k_subtile_idx=k_subtile_idx)
        tmem_s.qk_mma(desc_k_base=desc_k_base, k_subtile_idx=k_subtile_idx)
    tmem_s.commit()
    smem_k.release()


def staged_pv_mma_v_tile(smem_v, smem_p, tmem_o, iterations: int):
    """Consume one whole V stage plus one P stage and produce one O tile."""
    smem_p.wait()
    desc_p_base = smem_p.p_desc()
    smem_v.wait()
    tmem_o.acquire()
    for v_subtile_idx in range(iterations):
        desc_v_base = smem_v.v_desc(v_subtile_idx=v_subtile_idx)
        tmem_o.pv_mma(
            desc_p_base=desc_p_base,
            desc_v_base=desc_v_base,
            v_subtile_idx=v_subtile_idx,
        )
    tmem_o.commit()
    smem_v.release()
    smem_p.release()


def staged_pv_mma_v_tile_per_n(
    smem_v,
    smem_p,
    tmem_o,
    *,
    iterations_pv_k: int,
    iterations_pv_n: int,
):
    """Consume P/V and publish one O pipeline token per PV N-slice."""
    smem_p.wait()
    desc_p_base = smem_p.p_desc()
    smem_v.wait()
    for pv_n_idx in range(iterations_pv_n):
        tmem_o.acquire()
        for pv_k_idx in range(iterations_pv_k):
            desc_v_base = smem_v.v_desc_n_major(pv_n_idx=pv_n_idx, pv_k_idx=pv_k_idx)
            tmem_o.pv_mma_n_major(
                desc_p_base=desc_p_base,
                desc_v_base=desc_v_base,
                pv_n_idx=pv_n_idx,
                pv_k_idx=pv_k_idx,
            )
        tmem_o.commit()
    smem_v.release()
    smem_p.release()


def staged_pv_mma_tmem_p(
    smem_kv,
    tmem_p,
    tmem_o,
    *,
    head_dim_stages,
    consumer_label,
    producer_label="pv_mma_loop_tmem_p",
):
    """Consume TMEM P and staged V descriptors to produce one PV output tile."""
    tmem_p.wait()
    p_stage_idx = tmem_p.p_stage()
    tmem_o.acquire()
    is_tail = producer_label == "pv_mma_tail_tmem_p"
    for v_subtile_idx in range(head_dim_stages):
        smem_kv.wait()
        v_desc_0 = getattr(smem_kv, consumer_label)(v_subtile_idx=v_subtile_idx)
        getattr(tmem_o, producer_label)(
            p_stage_idx=p_stage_idx,
            v_desc_0=v_desc_0,
            v_subtile_idx=v_subtile_idx,
            is_tail=is_tail,
        )
        smem_kv.release()
    tmem_o.commit()
    tmem_p.release()
