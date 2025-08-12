"""Helper functions for config-related settings."""

import os
from pathlib import Path
from typing import Any, Dict, List, Union

from omegaconf import DictConfig, OmegaConf
from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, YamlConfigSettingsSource
from pydantic_settings.sources.types import PathType


def deep_merge_dicts(*confs: Union[Dict, DictConfig]) -> Dict:
    """Deep merge a list of dictionaries via OmegaConf.merge.

    Args:
        *confs: A list of dictionaries or DictConfig objects to merge.

    Returns:
        A merged dictionary.
    """
    if len(confs) == 0:
        return {}
    merged_conf = OmegaConf.merge(*[OmegaConf.create(conf) for conf in confs])
    result = OmegaConf.to_container(merged_conf, resolve=True)
    assert isinstance(result, Dict), f"Expected dict, got {type(result)}"
    return result


class DynamicYamlWithDeepMergeSettingsSource(YamlConfigSettingsSource):
    """YAML config settings source that dynamically loads files and merges them via deep update.

    We utilize the omegaconf library for deep merging.
    """

    def _read_files(self, files: PathType | None) -> dict[str, Any]:
        if files is None:
            return {}
        if isinstance(files, (str, os.PathLike)):
            files = [files]

        confs = []
        for file in files:
            file_path = Path(file).expanduser()
            if file_path.is_file():
                confs.append(OmegaConf.load(file_path))

        return deep_merge_dicts(*confs)

    def __call__(self):
        """Call additional config files based on current state."""
        yaml_data = self.yaml_data  # this points to the default yaml data now
        additional_files_data = self._read_files(self.current_state.get("yaml_configs", []))

        return deep_merge_dicts(yaml_data, additional_files_data)


class DynamicYamlMixInForSettings:
    """Mix-in class for settings providing dynamic yaml loading as lowest priority source.

    NOTE: This class must come FIRST in the MRO such that `yaml_configs` can be processed before
    since otherwise we cannot load default values from the `yaml_configs` first.

    This mix-in enforces the following precedence order:
    - init settings
    - env settings
    - dotenv settings
    - file secret settings
    - yaml configs
    - default settings

    You can learn more about the different settings sources in
    https://docs.pydantic.dev/latest/concepts/pydantic_settings/#field-value-priority.

    Note in particular how yaml settings have precedence only over default settings. You can hence
    think of the yaml settings as a way to override default settings.

    Also consider the following consequences of precedence order in nested config settings:
    - yaml configs for outer settings get converted to init settings for inner settings and hence
      ALWAYS take precedence over yaml configs specified for inner settings.
        - This implies inner settings from outer yaml configs also take precedence over outer inner
          settings like env settings since they are now init settings from the view of the inner
          settings.
    - Explicitly initialized fields for inner settings take precedence over outer yaml configs for
      inner settings since they are provided as init arguments.
    - Check out ``tests/unittest/_torch/auto_deploy/unit/singlegpu/utils/test_config.py`` for more
      examples.


    You can also provide multiple yaml config files to load. In this case, the files are deep merged
    together in the order they are provided. Hence, the following order (decreasing precedence) for
    multiple yaml config files is:
        - default yaml provided as ``yaml_file`` argument in the ``model_config`` (``ConfigDict``)
        - argument 0 of ``yaml_configs``
        - argument 1 of ``yaml_configs``
        - ...
        - last argument of ``yaml_configs``
    """

    yaml_configs: List[PathType] = Field(
        default_factory=list,
        description="Additional yaml config files to load.",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customise settings sources."""
        deferred_yaml_settings = DynamicYamlWithDeepMergeSettingsSource(settings_cls)
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            deferred_yaml_settings,  # yaml files have lowest priority just before default values
        )
