import json
import os
import time
from pathlib import Path
from typing import Union

import tensorrt as trt

from ..builder import BuildConfig
from ..logger import logger
from ..models.modeling_utils import PretrainedConfig


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(engine)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


class EngineConfig:

    def __init__(self, pretrained_config: 'PretrainedConfig',
                 build_config: 'BuildConfig', version: str):
        self.pretrained_config = pretrained_config
        self.build_config = build_config
        self.version = version

    @classmethod
    def from_json_file(cls, config_file):
        with open(config_file) as f:
            config = json.load(f)
            return cls(PretrainedConfig.from_dict(config['pretrained_config']),
                       BuildConfig.from_dict(config['build_config']),
                       config['version'])

    def to_dict(self):
        return {
            'version': self.version,
            'pretrained_config': self.pretrained_config.to_dict(),
            'build_config': self.build_config.to_dict(),
        }


class Engine:

    def __init__(self, config: EngineConfig, engine: trt.IHostMemory):
        self.config = config
        self.engine = engine

    def save(self, engine_dir: str):
        if self.config.pretrained_config.mapping.rank == 0:
            with open(os.path.join(engine_dir, 'config.json'),
                      "w",
                      encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=4)
        serialize_engine(
            self.engine,
            os.path.join(
                engine_dir,
                f'rank{self.config.pretrained_config.mapping.rank}.engine'))

    @classmethod
    def from_dir(cls, engine_dir: str, rank: int = 0):
        with open(os.path.join(engine_dir, f'rank{rank}.engine'), 'rb') as f:
            engine_buffer = f.read()

        config = EngineConfig.from_json_file(
            os.path.join(engine_dir, 'config.json'))
        config.pretrained_config.set_rank(rank)

        return cls(config, engine_buffer)


def get_engine_version(engine_dir: str) -> Union[None, str]:
    engine_dir = Path(engine_dir)
    config_path = engine_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    if 'version' not in config:
        return None

    return config['version']
