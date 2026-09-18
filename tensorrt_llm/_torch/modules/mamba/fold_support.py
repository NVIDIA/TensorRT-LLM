"""Support registry for the folded save-last prefill (see ``fold_save_last_enabled``).

The fold materialises the save-last recurrent-state snapshot inside a single
context chunk; only the gated-delta-net mixer implements it. Mixers that cannot
run a folded chunk register the reason here at construction time, and the cache
manager, which decides per request whether to fold, falls back to the two-chunk
schedule when any reason is present.
"""

from typing import List, Optional

_UNSUPPORTED: List[str] = []


def mark_fold_unsupported(reason: str) -> None:
    """Record that the model contains a component that cannot run a folded chunk."""
    if reason not in _UNSUPPORTED:
        _UNSUPPORTED.append(reason)


def fold_unsupported_reason() -> Optional[str]:
    """The first registered reason the fold cannot be used, or ``None``."""
    return _UNSUPPORTED[0] if _UNSUPPORTED else None
