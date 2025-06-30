import gc
import inspect
import types
import weakref
from contextlib import contextmanager

from tensorrt_llm.logger import logger


@contextmanager
def assert_resource_freed(object_creation_func, *args, **kwargs):
    """
    Create a resource via object_creation_func(*args, **kwargs),
    disable the generational GC to force pure refcount freeing,
    yield the resource, then assert that it was destroyed when the context exits.
    If it wasn’t freed, collect and report all remaining referrers.
    """
    # Ensure a clean start
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()

    resource = object_creation_func(*args, **kwargs)
    alive_ref = weakref.ref(resource)

    try:
        yield resource

    finally:
        # Drop our own strong reference
        try:
            del resource
        except NameError:
            pass

        # If still alive, diagnose
        leaked = alive_ref()
        if leaked is not None:
            # Restore GC so we can introspect
            if gc_was_enabled:
                gc.enable()
            # Give GC a chance to finalize anything pending
            gc.collect()

            # Find all objects still referring to our instance
            refs = gc.get_referrers(leaked)
            # Filter out inspection internals (frames, tracebacks, the weakref itself, etc.)
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
            report = "\n".join(
                report_lines) or "   <no non‐internal referrers found>"

            if filtered:
                raise AssertionError(
                    "Resource was NOT freed upon context exit!\n"
                    f"{len(filtered)} referrer(s) still alive:\n{report}\n")
            else:
                logger.info("Resource was freed upon context exit.")

        # Otherwise, restore GC state
        if gc_was_enabled:
            gc.enable()
