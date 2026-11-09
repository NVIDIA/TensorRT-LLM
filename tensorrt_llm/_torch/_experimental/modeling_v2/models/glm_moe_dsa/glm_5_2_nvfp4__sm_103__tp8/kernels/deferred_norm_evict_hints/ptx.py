# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inline-PTX helpers that only the deferred_norm_evict_hints kernels use (CuTe DSL).

None of them computes a value that reaches an output: they are L2 cache
hints (bulk prefetch, eviction-priority policies on stores and bulk copies),
a named CTA barrier, a ``%globaltimer`` spin and test-only timer / SM-id
stamps. The arithmetic and the post_moe_hop workspace
protocol come unchanged from deferred_norm's helpers (``deferred_norm/ptx.py``).
"""

from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def _asm(res, args, text, cons, side_effects=True, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        args,
        text,
        cons,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(x, loc, ip):
    return x.ir_value(loc=loc, ip=ip)


@dsl_user_op
def prefetch_l2(gaddr: Int64, nbytes: Int32, *, loc=None, ip=None) -> None:
    """``cp.async.bulk.prefetch.L2`` of ``nbytes`` (16-byte multiple) at ``gaddr``."""
    _asm(
        None,
        [_v(gaddr, loc, ip), _v(nbytes, loc, ip)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def prefetch_line_l2(gaddr: Int64, *, loc=None, ip=None) -> None:
    """Per-thread ``prefetch.global.L2`` of the line holding ``gaddr`` (no
    uniform-operand serialization, unlike the bulk prefetch)."""
    _asm(None, [_v(gaddr, loc, ip)], "prefetch.global.L2 [$0];", "l", loc=loc, ip=ip)


@dsl_user_op
def policy_evict_last(*, loc=None, ip=None) -> Int64:
    """L2 cache policy: every access marked ``evict_last``."""
    return Int64(
        _asm(
            T.i64(), [], "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;", "=l", loc=loc, ip=ip
        )
    )


@dsl_user_op
def policy_evict_first(*, loc=None, ip=None) -> Int64:
    """L2 cache policy: every access marked ``evict_first``."""
    return Int64(
        _asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
            "=l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_v4_hint(
    addr: Int64, a: Int32, b: Int32, c: Int32, d: Int32, policy: Int64, *, loc=None, ip=None
) -> None:
    """``st.global.v4.b32`` with an L2 cache-policy hint (same store otherwise)."""
    _asm(
        None,
        [_v(addr, loc, ip)] + [_v(x, loc, ip) for x in (a, b, c, d)] + [_v(policy, loc, ip)],
        "st.global.L2::cache_hint.v4.b32 [$0], {$1, $2, $3, $4}, $5;",
        "l,r,r,r,r,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bulk_g2s_hint(
    saddr: Int32, gaddr: Int64, nbytes: Int32, mbar: Int32, policy: Int64, *, loc=None, ip=None
) -> None:
    """``cp.async.bulk`` global -> shared with an L2 cache-policy hint, completing
    on ``mbar`` with ``complete_tx`` (deferred_norm's copy otherwise)."""
    _asm(
        None,
        [
            _v(saddr, loc, ip),
            _v(gaddr, loc, ip),
            _v(nbytes, loc, ip),
            _v(mbar, loc, ip),
            _v(policy, loc, ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[$0], [$1], $2, [$3], $4;",
        "r,l,r,r,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def globaltimer(*, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc=loc, ip=ip))


@dsl_user_op
def smid(*, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %smid;", "=r", loc=loc, ip=ip))


@dsl_user_op
def st_b64(gaddr: Int64, v: Int64, *, loc=None, ip=None) -> None:
    _asm(
        None, [_v(gaddr, loc, ip), _v(v, loc, ip)], "st.global.b64 [$0], $1;", "l,l", loc=loc, ip=ip
    )


@dsl_user_op
def spin_until(deadline: Int64, *, loc=None, ip=None) -> None:
    """Spin until ``%globaltimer`` reaches ``deadline`` (bounded by the clock only)."""
    _asm(
        None,
        [_v(deadline, loc, ip)],
        "{\n.reg .pred p;\n.reg .b64 now;\nHINTS_SPIN:\nmov.u64 now, %globaltimer;\n"
        "setp.lt.s64 p, now, $0;\n@p bra HINTS_SPIN;\n}\n",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bar_sync_named(bar_id: int, count: int, *, loc=None, ip=None) -> None:
    """``bar.sync bar_id, count``: a CTA barrier over ``count`` threads only
    (``count`` a multiple of 32; ``bar_id`` != 0, which ``sync_threads`` uses)."""
    assert 0 < bar_id < 16 and count % 32 == 0
    _asm(None, [], f"bar.sync {bar_id}, {count};", "", loc=loc, ip=ip)
