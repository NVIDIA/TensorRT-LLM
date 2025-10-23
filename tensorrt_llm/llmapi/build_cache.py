import contextlib
import datetime
import enum
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import filelock

import tensorrt_llm
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.llmapi.utils import enable_llm_debug, print_colored
from tensorrt_llm.logger import logger


def get_build_cache_config_from_env() -> tuple[bool, str]:
    """
    Get the build cache configuration from the environment variables
    """
    build_cache_enabled = os.environ.get('TLLM_LLMAPI_BUILD_CACHE') == '1'
    build_cache_root = os.environ.get(
        'TLLM_LLMAPI_BUILD_CACHE_ROOT',
        '/tmp/.cache/tensorrt_llm/llmapi/')  # nosec B108
    return build_cache_enabled, build_cache_root


class BuildCacheConfig:
    """
    Configuration for the build cache.

    Attributes:
        cache_root (str): The root directory for the build cache.
        max_records (int): The maximum number of records to store in the cache.
        max_cache_storage_gb (float): The maximum amount of storage (in GB) to use for the cache.

    Note:
        The build-cache assumes the weights of the model are not changed during the execution. If the weights are
        changed, you should remove the caches manually.
    """

    def __init__(self,
                 cache_root: Optional[Path] = None,
                 max_records: int = 10,
                 max_cache_storage_gb: float = 256):
        self._cache_root = cache_root
        self._max_records = max_records
        self._max_cache_storage_gb = max_cache_storage_gb

    @property
    def cache_root(self) -> Path:
        _build_cache_enabled, _build_cache_root = get_build_cache_config_from_env(
        )
        return self._cache_root or Path(_build_cache_root)

    @property
    def max_records(self) -> int:
        return self._max_records

    @property
    def max_cache_storage_gb(self) -> float:
        return self._max_cache_storage_gb


class BuildCache:
    """
    The BuildCache class is a class that manages the intermediate products from the build steps.

    NOTE: currently, only engine-building is supported
    TODO[chunweiy]: add support for other build steps, such as quantization, convert_checkpoint, etc.
    """
    # The version of the cache, will be used to determine if the cache is compatible
    CACHE_VERSION = 0

    def __init__(self, config: Optional[BuildCacheConfig] = None):

        _, default_cache_root = get_build_cache_config_from_env()
        config = config or BuildCacheConfig()

        self.cache_root = config.cache_root or Path(default_cache_root)
        self.max_records = config.max_records
        self.max_cache_storage_gb = config.max_cache_storage_gb

        if config.max_records < 1:
            raise ValueError("max_records should be greater than 0")

    def free_storage_in_gb(self) -> float:
        ''' Get the free storage capacity of the cache. '''
        # measure the root directory
        if self.cache_root.parent.exists():
            usage = shutil.disk_usage(self.cache_root.parent)
            return usage.free / 1024**3
        return 0

    def get_engine_building_cache_stage(self,
                                        build_config: BuildConfig,
                                        model_path: Optional[Path] = None,
                                        force_rebuild: bool = False,
                                        **kwargs) -> 'CachedStage':
        '''
        Get the build step for engine building.
        '''
        build_config_str = json.dumps(self.prune_build_config_for_cache_key(
            build_config.to_dict()),
                                      sort_keys=True)

        kwargs_str = json.dumps(kwargs, sort_keys=True)

        return CachedStage(parent=self,
                           kind=CacheRecord.Kind.Engine,
                           cache_root=self.cache_root,
                           force_rebuild=force_rebuild,
                           inputs=[build_config_str, model_path, kwargs_str])

    def prune_caches(self, has_incoming_record: bool = False):
        '''
        Clean up the cache records to make sure the cache size is within the limit

        Args:
            has_incoming_record (bool): If the cache has incoming record, the existing records will be further pruned to
            reserve space for the incoming record
        '''
        if not self.cache_root.exists():
            return
        self._clean_up_cache_dir()
        records = []
        for dir in self.cache_root.iterdir():
            records.append(self._load_cache_record(dir))
        records.sort(key=lambda x: x.time, reverse=True)
        max_records = self.max_records - 1 if has_incoming_record else self.max_records
        # prune the cache to meet max_records and max_cache_storage_gb limitation
        while len(records) > max_records or sum(
                r.storage_gb for r in records) > self.max_cache_storage_gb:
            record = records.pop()
            # remove the directory and its content
            shutil.rmtree(record.path)

    @staticmethod
    def prune_build_config_for_cache_key(build_config: dict) -> dict:
        black_list = ['dry_run']
        dic = build_config.copy()
        for key in black_list:
            if key in dic:
                dic.pop(key)
        return dic

    def load_cache_records(self) -> List["CacheRecord"]:
        '''
        Load all the cache records from the cache directory
        '''
        records = []
        if not self.cache_root.exists():
            return records

        for dir in self.cache_root.iterdir():
            records.append(self._load_cache_record(dir))
        return records

    def _load_cache_record(self, cache_dir) -> "CacheRecord":
        '''
        Get the cache record from the cache directory
        '''
        metadata = json.loads((cache_dir / 'metadata.json').read_text())
        storage_gb = sum(f.stat().st_size for f in cache_dir.glob('**/*')
                         if f.is_file()) / 1024**3
        return CacheRecord(kind=CacheRecord.Kind.__members__[metadata['kind']],
                           storage_gb=storage_gb,
                           path=cache_dir,
                           time=datetime.datetime.fromisoformat(
                               metadata['datetime']))

    def _clean_up_cache_dir(self):
        '''
        Clean up the files in the cache directory, remove anything that is not in the cache
        '''
        # get all the files and directies in the cache_root
        if not self.cache_root.exists():
            return
        for file_or_dir in self.cache_root.iterdir():
            if not self.is_cache_valid(file_or_dir):
                logger.info(f"Removing invalid cache directory {dir}")
                if file_or_dir.is_file():
                    file_or_dir.unlink()
                else:
                    shutil.rmtree(file_or_dir)

    def is_cache_valid(self, cache_dir: Path) -> bool:
        '''
        Check if the cache directory is valid
        '''
        if not cache_dir.exists():
            return False

        metadata_path = cache_dir / 'metadata.json'
        if not metadata_path.exists():
            return False

        metadata = json.loads(metadata_path.read_text())
        if metadata.get('version') != BuildCache.CACHE_VERSION:
            return False

        content = cache_dir / 'content'
        if not content.exists():
            return False

        return True


@dataclass
class CachedStage:
    '''
    CachedStage is a class that represents a stage in the build process, it helps to manage the intermediate product.

    The cache is organized as follows:

    this_cache_dir/     # name is like "engine-<hash>"
        metadata.json   # the metadata of the cache
        content/        # the actual product of the build step, such trt-llm engine directory
    '''
    # The parent should be kept alive by CachedStep instance
    parent: BuildCache
    cache_root: Path
    # The inputs will be used to determine if the step needs to be re-run, so all the variables should be put here
    inputs: List[Any]
    kind: "CacheRecord.Kind"
    # If force_rebuild is set to True, the cache will be ignored
    force_rebuild: bool = False

    def get_hash_key(self):
        lib_version = tensorrt_llm.__version__
        input_strs = [str(i) for i in self.inputs]
        return hashlib.md5(
            f"{lib_version}-{input_strs}".encode()).hexdigest()  # nosec B324

    def get_cache_path(self) -> Path:
        '''
        The path to the product of the build step, will be overwritten if the step is re-run
        '''
        return self.cache_root / f"{self.kind.value}-{self.get_hash_key()}"

    def get_engine_path(self) -> Path:
        return self.get_cache_path() / 'content'

    def get_cache_metadata(self) -> dict:
        res = {
            "version": BuildCache.CACHE_VERSION,
            "datetime": datetime.datetime.now().isoformat(),
            "kind": self.kind.name,
        }
        return res

    def is_cached(self) -> bool:
        '''
        Check if the product of the build step is in the cache
        '''
        if self.force_rebuild:
            return False
        try:
            if self.get_cache_path().exists():
                metadata = json.loads(
                    (self.get_cache_path() / 'metadata.json').read_text())
                if metadata["version"] == BuildCache.CACHE_VERSION:
                    return True
        except:
            pass

        return False

    @contextlib.contextmanager
    def write_guard(self):
        ''' Guard the cache writing process.

        The cache writing process should be atomic, so the filelock is used to protect the cache writing process. And
        the cache metadata will be written to the cache directory.

        Args:
            final_engien_dir: the final engine directory
        '''
        self.parent.prune_caches(has_incoming_record=True)

        target_dir = self.get_cache_path()
        if enable_llm_debug():
            print_colored(f"Writing cache to {target_dir}\n", "yellow")

        # To avoid the cache modification conflict, a dummy directory is used to write the cache, and then rename it to
        # the target directory
        dummy_target_dir = Path(f"{target_dir.parent}/{target_dir.name}.dummy")

        dummy_target_dir.mkdir(parents=True, exist_ok=True)
        # TODO[chunweiy]: deal with the cache modification conflict
        lock = filelock.FileLock(dummy_target_dir / '.filelock', timeout=10)

        with open(dummy_target_dir / 'metadata.json', 'w') as f:
            f.write(json.dumps(self.get_cache_metadata()))

        with lock:
            yield dummy_target_dir / 'content'

            # If engine building is successful, rename the dummy directory to the target directory
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(dummy_target_dir, target_dir)


@dataclass(unsafe_hash=True)
class CacheRecord:
    '''
    CacheRecord is a class that represents a record in the cache directory.
    '''

    class Kind(enum.Enum):
        Engine = 'engine'
        Checkpoint = 'checkpoint'

    kind: Kind
    storage_gb: float
    path: Path
    time: datetime.datetime
