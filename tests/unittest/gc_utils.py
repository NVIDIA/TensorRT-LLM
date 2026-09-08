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

import gc
import inspect
import types
import weakref
from contextlib import contextmanager

from tensorrt_llm.logger import logger


def _diagnose_resource_leak(alive_ref: weakref.ReferenceType[object],
                            gc_was_enabled: bool) -> None:
    """Report external references if the resource remains alive."""
    leaked = alive_ref()
    if leaked is None:
        return

    # Restore GC so we can introspect
    if gc_was_enabled:
        gc.enable()
    # Give GC a chance to finalize anything pending
    gc.collect()

    # Find all objects still referring to our instance
    refs = gc.get_referrers(leaked)
    # Filter out inspection internals (frames, tracebacks, the weakref itself,
    # etc.)
    filtered = []
    for r in refs:
        # skip the weakref container itself
        if isinstance(r, dict) and any(
                isinstance(v, weakref.ref) and v() is leaked
                for v in r.values()):
            continue
        # skip our own local variables frame
        if inspect.isframe(r):
            continue
        # skip the generator’s internal cell
        if isinstance(r, types.CellType):
            continue
        filtered.append(r)

    # Build a human‐readable report
    report_lines = [
        f" - {type(r).__name__} at 0x{id(r):x}: {repr(r)[:200]!r}"
        for r in filtered
    ]
    report = "\n".join(report_lines) or "   <no non‐internal referrers found>"

    if filtered:
        raise AssertionError(
            "Resource was NOT freed upon context exit!\n"
            f"{len(filtered)} referrer(s) still alive:\n{report}\n")

    logger.info("Resource was freed upon context exit.")


@contextmanager
def assert_resource_freed(object_creation_func, *args, **kwargs):
    """Create a resource and assert it is destroyed when the context exits.

    The resource is created via object_creation_func(*args, **kwargs). The
    generational GC is disabled to force pure refcount freeing. If the resource
    is not freed, collect and report all remaining referrers.
    """
    # Ensure a clean start
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()

    resource = object_creation_func(*args, **kwargs)
    alive_ref = weakref.ref(resource)

    body_exception_active = False
    try:
        yield resource
    except BaseException:
        body_exception_active = True
        raise
    finally:
        shutdown_failed = False
        try:
            try:
                # Background threads are only released by shutdown(); dropping
                # the last reference does not run it in time. Looked up on the
                # type so no bound method keeps the instance alive.
                if callable(getattr(type(resource), "shutdown", None)):
                    resource.shutdown()
            except BaseException:
                # Cleanup must never mask an exception from the context body.
                shutdown_failed = True
                if not body_exception_active:
                    raise
                logger.exception(
                    "Resource shutdown failed while handling another exception")
        finally:
            try:
                # Drop our own strong reference
                try:
                    del resource
                except NameError:
                    pass

                # A failed shutdown remains on the active exception traceback,
                # which can temporarily retain the resource. Preserve that
                # exception instead of reporting it as a leak.
                if not shutdown_failed:
                    _diagnose_resource_leak(alive_ref, gc_was_enabled)
            finally:
                # Restore the caller's GC state even if shutdown or diagnostics
                # raise.
                if gc_was_enabled:
                    gc.enable()
