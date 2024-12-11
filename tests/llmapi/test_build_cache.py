import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tensorrt_llm.llmapi import BuildConfig
from tensorrt_llm.llmapi.build_cache import BuildCache, BuildCacheConfig


def test_BuildStep():
    with TemporaryDirectory() as tempdir:
        build_cache = BuildCache(BuildCacheConfig(Path(tempdir)))
        build_step = build_cache.get_engine_building_cache_stage(
            build_config=BuildConfig(), hf_model_name="test")
        assert not build_step.is_cached()
        print(build_step.get_cache_path().absolute())
        assert build_step.get_cache_metadata(
        )["version"] == BuildCache.CACHE_VERSION

        with build_step.write_guard() as product_path:
            product_path.mkdir()
            with open(product_path / 'config.json', 'w') as f:
                f.write(json.dumps({"a": 1, "b": 2}))

        assert build_step.is_cached()


def test_BuildCache_clean_untracked_path():
    # The BuildCache could cleanup the untracked files/dirs within the cache_root
    with TemporaryDirectory() as tempdir:
        build_cache = BuildCache(BuildCacheConfig(Path(tempdir)))
        (build_cache.cache_root / 'untracked').mkdir()
        (build_cache.cache_root / 'untracked_file').touch()

        build_cache.prune_caches()
        assert not (build_cache.cache_root / 'untracked').exists()


def test_BuildCache_clean_cache_exceed_record_limit():
    # The BuildCache could cleanup the cache if the number of records exceed the limit
    with TemporaryDirectory() as tempdir:
        build_cache = BuildCache(BuildCacheConfig(Path(tempdir), max_records=2))
        build_config = BuildConfig()

        def create_cache(hf_model_name: str):
            step = build_cache.get_engine_building_cache_stage(
                build_config=build_config, hf_model_name=hf_model_name)
            assert not step.is_cached()
            with step.write_guard() as product_path:
                product_path.mkdir()
                with open(product_path / 'config.json', 'w') as f:
                    f.write(json.dumps({"a": 1, "b": 2}))

        records = set()

        for i in range(3):
            create_cache(str(i))
            for record in build_cache.load_cache_records():
                records.add(record)

        assert len(records) == 3

        records = sorted(records, key=lambda x: x.time)
        print(records)

        print(f"cache structure: {list(build_cache.cache_root.glob('*'))}")
        assert len(list(build_cache.cache_root.glob('*'))) == 2

        final_records = build_cache.load_cache_records()
        # The earliest one should be pruned
        assert records[0] not in final_records
        for record in final_records:
            assert record in records


def test_build_cache_prune_untracked_files():
    # The BuildCache could cleanup the untracked files/dirs within the cache_root
    # The broken cache such as empty cache record directory should be pruned as well
    with TemporaryDirectory() as tempdir:
        build_cache = BuildCache(BuildCacheConfig(cache_root=Path(tempdir)))
        (build_cache.cache_root / 'untracked').mkdir()
        (build_cache.cache_root / 'untracked_file').touch()
        (build_cache.cache_root / 'broken_cache').mkdir()

        build_cache.prune_caches()
        assert not (build_cache.cache_root / 'untracked').exists()
        assert not (build_cache.cache_root / 'untracked_file').exists()
        assert not (build_cache.cache_root / 'broken_cache').exists()


if __name__ == '__main__':
    #test_build_get_updated_build_cache()
    test_build_cache_prune_untracked_files()
