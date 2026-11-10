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
"""Persistent cache of ``cute.compile`` results, shared by the processes and ranks that use it.

``TRTLLM_CUTE_DSL_CACHE_DIR=<dir>`` turns it on.
"""

import enum
import functools
import hashlib
import importlib.metadata
import inspect
import io
import os
import threading
import time
import uuid

from ..logger import logger

_CACHE_DIR_ENV = "TRTLLM_CUTE_DSL_CACHE_DIR"
# CuTe DSL releases whose internals the hooks below were checked against; a new nvidia-cutlass-dsl
# pin turns the cache off until it is added here and test_cute_dsl_file_cache.py passes on it.
_TESTED_DSL_VERSIONS = frozenset({"4.8.0"})
_KEY_VERSION = b"tllm-cute-dsl-file-cache-v1"
# The compile option that only names an output location.
_OPTION_PATHS = frozenset({"DumpDir"})
# The CuTe DSL's own post-compile hook, which attaches the TVM-FFI wrapper.
_TVM_FFI_HOOK = "CutlassBaseDSL.compile_and_cache.<locals>.post_compile_hook"

_local = threading.local()
_file_digests: dict[tuple[str, int, int], str] = {}
_dsl_digests: dict[type, bytes] = {}
_installed = False


def _stable(value) -> str:
    """Text of an option value that does not depend on the process."""
    if isinstance(value, (set, frozenset)):
        return repr(sorted(map(str, value)))
    text = repr(value)
    if " at 0x" in text:
        raise ValueError(f"unstable value {text}")
    return text


def _render(value) -> str:
    """Process-independent text of a TVM-FFI argument spec; ValueError when there is none."""
    kind = type(value)
    if kind.__module__.startswith("tvm_ffi"):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, enum.Enum):
        return f"{kind.__qualname__}.{value.name}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(map(_render, value)) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(sorted(f"{_render(k)}:{_render(v)}" for k, v in value.items())) + "}"
    if isinstance(value, type) or inspect.isroutine(value):
        return f"<{value.__module__}.{value.__qualname__}>"
    if kind.__module__.startswith("cutlass.base_dsl.tvm_ffi_builder"):
        fields = ",".join(f"{k}={_render(v)}" for k, v in sorted(vars(value).items()))
        return f"{kind.__name__}({fields})"
    return f"{kind.__module__}.{kind.__qualname__}:{_stable(value)}"


def _file_digest(path: str) -> str:
    st = os.stat(path)
    memo = (path, st.st_size, st.st_mtime_ns)
    digest = _file_digests.get(memo)
    if digest is None:
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        _file_digests[memo] = digest
    return digest


def _cache_key(backend, ctx, arch: str) -> str:
    """sha256 of what the compile reads: DSL build, traced module, pipeline, settings, call spec."""
    from cutlass.base_dsl.compiler import LinkLibraries

    dsl = backend._dsl
    h = hashlib.sha256(_KEY_VERSION)
    h.update(f"{type(dsl).__qualname__} {type(backend).__qualname__}\0".encode())
    # Digest of the DSL's Python sources and compiler library; the DSL memoizes one instance only.
    if type(dsl) not in _dsl_digests:
        _dsl_digests[type(dsl)] = dsl.get_version().digest()
    h.update(_dsl_digests[type(dsl)])
    buf = io.BytesIO()
    ctx.module.operation.write_bytecode(buf)
    h.update(buf.getvalue())
    # The string the backend compiles with, including any process-wide patches to it.
    h.update(
        f"\0pipeline {dsl.preprocess_pipeline(dsl._get_pipeline(ctx.pipeline), arch)}".encode()
    )
    h.update(f"\0debug {sorted(dsl.compile_options._debug_selectors)}".encode())
    if dsl.compile_options.enable_tvm_ffi:
        # The TVM-FFI wrapper is built after the passes from the call's arguments, not the module.
        spec = dsl._tvm_ffi_args_spec_converter(
            ctx.function_name, ctx.sig, ctx.full_args, ctx.full_kwargs
        )
        display_name = getattr(ctx.funcBody, "__name__", None)
        h.update(f"\0ffi {ctx.function_name} {display_name} {_render(spec)}".encode())
    # The DSL settings that change what is compiled, as the DSL's own JIT cache keys them.
    h.update(f"\0envar {dsl.envar.cache_key_str()}".encode())
    options = dsl.compile_options.options
    for cls, option in sorted(options.items(), key=lambda kv: kv[0].__name__):
        if cls.__name__ not in _OPTION_PATHS:
            h.update(f"\0option {cls.__name__}={_stable(option.value)}".encode())
    h.update(f"\0arch {arch}".encode())
    libs = (s.strip().strip("'") for s in str(options[LinkLibraries].value).split(","))
    for lib in filter(None, libs):
        h.update(f"\0lib {lib} {_file_digest(lib) if os.path.isfile(lib) else '-'}".encode())
    return h.hexdigest()


def _cacheable(dsl) -> bool:
    """False for compiles that write files beside their result or run with DSL caching off."""
    from cutlass.base_dsl.compiler import (
        DeviceTarget,
        FlattenLocsOutputJsonPath,
        KeepCUBIN,
        KeepPTX,
        KeepSASS,
        RemarkOutput,
    )

    envar, options = dsl.envar, dsl.compile_options.options
    if envar.no_cache or envar.disable_file_caching or getattr(envar, "keep_tokens", None):
        return False
    if any((envar.keep_ptx, envar.keep_cubin, envar.keep_sass, envar.keep_ir, envar.keep_ir_clean)):
        return False
    side_outputs = (
        KeepPTX,
        KeepCUBIN,
        KeepSASS,
        DeviceTarget,
        FlattenLocsOutputJsonPath,
        RemarkOutput,
    )
    if any(options[cls].value for cls in side_outputs):
        return False
    # A post-compile hook other than the TVM-FFI wrapper's may read more than the key holds.
    hook = dsl.compiler_provider._post_compile_hook
    expected = _TVM_FFI_HOOK if dsl.compile_options.enable_tvm_ffi else None
    if getattr(hook, "__qualname__", None) != expected:
        raise ValueError(f"unexpected post-compile hook {hook!r}")
    return True


def _load(path: str):
    """The module stored at ``path``, or None when there is none."""
    from cutlass._mlir import ir
    from cutlass.base_dsl.cache_helpers import read_bytecode_and_check_crc32

    try:
        f = open(path, "rb")
    except FileNotFoundError:
        return None
    with f:
        context = ir.Context()
        # The module keeps its context alive, and a threaded context keeps a pool of threads.
        context.enable_multithreading(False)
        with context:
            return read_bytecode_and_check_crc32(f)


def _store(path: str, module) -> tuple[float, float]:
    """Write an entry in the DSL's file-cache format; readers see the whole file or none.

    Returns the seconds spent writing the temporary file and renaming it into place.
    """
    from cutlass.base_dsl.cache_helpers import write_bytecode_with_crc32

    start = time.perf_counter()
    tmp = os.path.join(os.path.dirname(path), f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with open(tmp, "xb") as f:
            write_bytecode_with_crc32(f, module)
        written = time.perf_counter()
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return written - start, time.perf_counter() - written


def install_cute_dsl_file_cache() -> None:
    """Install the cache when ``TRTLLM_CUTE_DSL_CACHE_DIR`` names a directory."""
    global _installed
    directory = os.environ.get(_CACHE_DIR_ENV, "")
    if not directory or _installed:
        return
    try:
        import cutlass

        versions = {importlib.metadata.version("nvidia-cutlass-dsl"), cutlass.__version__}
    except (ImportError, AttributeError, importlib.metadata.PackageNotFoundError):
        versions = {None}
    # The installed release and the cutlass package that imports, which PYTHONPATH may change.
    if not versions <= _TESTED_DSL_VERSIONS:
        logger.warning(
            f"[cute_dsl_file_cache] not enabled: CuTe DSL {sorted(map(str, versions))} is not "
            f"{sorted(_TESTED_DSL_VERSIONS)}"
        )
        return
    try:
        # What the hooks below import or call per compile; a missing one leaves the DSL as it is.
        from cutlass._mlir import ir  # noqa: F401
        from cutlass.base_dsl import compile_backend, compiler
        from cutlass.base_dsl.cache_helpers import (  # noqa: F401
            read_bytecode_and_check_crc32,
            write_bytecode_with_crc32,
        )
        from cutlass.base_dsl.compiler import (  # noqa: F401
            DeviceTarget,
            FlattenLocsOutputJsonPath,
            KeepCUBIN,
            KeepPTX,
            KeepSASS,
            LinkLibraries,
            RemarkOutput,
        )
        from cutlass.base_dsl.dsl import BaseDSL
        from cutlass.base_dsl.env_manager import EnvironmentVarManager

        compile_callable = compiler.CompileCallable._compile
        backend_compile = compile_backend.CompilerBackend.compile
        compile_cache_state = compile_backend.CompileCacheState
        for owner, name in (
            (EnvironmentVarManager, "cache_key_str"),
            (BaseDSL, "get_version"),
            (BaseDSL, "preprocess_pipeline"),
            (BaseDSL, "_get_pipeline"),
            (compile_backend.CompilerBackend, "_resolve_compile_gpu_arch"),
            (compile_backend.CompilerBackend, "_validate_compile"),
            (compile_backend.CompilerBackend, "_validate_execution_target"),
            (compile_backend.CompilerBackend, "_build_compiled_function"),
        ):
            getattr(owner, name)
    except (ImportError, AttributeError) as e:
        logger.warning(f"[cute_dsl_file_cache] not enabled, unexpected CuTe DSL layout: {e}")
        return
    directory = os.path.abspath(os.path.expanduser(directory))
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        logger.warning(f"[cute_dsl_file_cache] not enabled, directory {directory} unusable: {e}")
        return

    def load_hit(backend, ctx, path):
        """The compiled function of the entry at ``path``, or None when there is no usable one."""
        try:
            module = _load(path)
            if module is None:
                return None
            backend._validate_compile(ctx)
            arch = backend._resolve_compile_gpu_arch()
            backend._validate_execution_target(arch)
            state = compile_cache_state(module=module, cache_hit=True, load_from_file_cache=True)
            return backend._build_compiled_function(ctx, arch, state)
        except Exception as e:  # noqa: BLE001 - an unusable entry must never stop a compile
            logger.warning_once(
                f"[cute_dsl_file_cache] {ctx.function_name}: entry {os.path.basename(path)} "
                f"unusable ({type(e).__name__}: {e}), compiling",
                key=f"cute_dsl_file_cache.entry.{path}",
            )
            return None

    @functools.wraps(compile_callable)
    def compile_marked(self, *args, **kwargs):
        _local.depth = getattr(_local, "depth", 0) + 1
        try:
            return compile_callable(self, *args, **kwargs)
        finally:
            _local.depth -= 1

    @functools.wraps(backend_compile)
    def compile_cached(self, ctx):
        # Only cute.compile calls; the DSL's own JIT cache covers the rest.
        if not getattr(_local, "depth", 0) or ctx.cache_enabled:
            return backend_compile(self, ctx)
        dsl = self._dsl
        start = time.perf_counter()
        key = None
        try:
            if _cacheable(dsl):
                key = _cache_key(self, ctx, self._resolve_compile_gpu_arch())
        except Exception as e:  # noqa: BLE001 - a key failure must never stop a compile
            logger.warning_once(
                f"[cute_dsl_file_cache] {ctx.function_name} is not cached: no key "
                f"({type(e).__name__}: {e})",
                key=f"cute_dsl_file_cache.key.{ctx.function_name}",
            )
        keyed = time.perf_counter()
        steps = f"key {keyed - start:.3f}s"
        path = os.path.join(directory, f"{dsl.name.lower()}_{key}.mlir") if key else None
        if path is not None:
            fn = load_hit(self, ctx, path)
            loaded = time.perf_counter()
            steps += f", load {loaded - keyed:.3f}s"
            if fn is not None:
                logger.debug(
                    f"[cute_dsl_file_cache] hit {ctx.function_name} key={key[:16]} "
                    f"in {loaded - start:.3f}s ({steps})"
                )
                return fn
        compile_start = time.perf_counter()
        fn = backend_compile(self, ctx)
        steps += f", compile {time.perf_counter() - compile_start:.3f}s"
        if path is not None and fn.ir_module is not None:
            try:
                write, rename = _store(path, fn.ir_module)
                steps += f", write {write:.3f}s, rename {rename:.3f}s"
            except Exception as e:  # noqa: BLE001 - a failed store must never fail the compile
                logger.warning_once(
                    f"[cute_dsl_file_cache] entries are not stored in {directory} "
                    f"({type(e).__name__}: {e})",
                    key="cute_dsl_file_cache.store",
                )
        elapsed = time.perf_counter() - start
        if path is None:
            logger.debug(
                f"[cute_dsl_file_cache] uncached {ctx.function_name} in {elapsed:.3f}s ({steps})"
            )
        else:
            logger.info(
                f"[cute_dsl_file_cache] miss {ctx.function_name} key={key[:16]} "
                f"in {elapsed:.3f}s ({steps})"
            )
        return fn

    compiler.CompileCallable._compile = compile_marked
    compile_backend.CompilerBackend.compile = compile_cached
    _installed = True
    logger.info(f"[cute_dsl_file_cache] cute.compile results are cached in {directory}")
