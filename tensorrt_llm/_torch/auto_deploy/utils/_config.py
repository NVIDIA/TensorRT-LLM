"""Helper functions for config-related settings."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, YamlConfigSettingsSource
from pydantic_settings.sources.types import DEFAULT_PATH, PathType


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.yaml_file_path not in [DEFAULT_PATH, None]:
            raise ValueError(
                "Static yaml config via yaml_file in config is not supported. Please "
                "specify the `yaml_default` field in your pydantic model instead."
            )

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
            else:
                raise ValueError(f"File {file} does not exist")

        return deep_merge_dicts(*confs)

    def __call__(self):
        """Call additional config files based on current state.

        This function also takes care of identifying the correct default yaml file based on the
        following precedence order (highest -> lowest priority):
        1. provided mode
        2. provided yaml_default
        3. yaml_default from default mode
        4. default yaml_default
        """
        # check default yaml sources from highest to lowest priority
        # NOTE: later in model validation, we throw an error if something was incorrectly configured
        settings_cls: DynamicYamlMixInForSettings = self.settings_cls
        default_file: Optional[str] = None
        if "mode" in self.current_state:
            # later in field validation, it should throw an error if mode was invalid string...
            # but we don't want to fail here, hence we use .get()
            default_file = settings_cls._get_yaml_default_from_mode(self.current_state["mode"])
        if not default_file and "yaml_default" in self.current_state:
            default_file = self.current_state["yaml_default"]
        if not default_file:
            # Only attempt to use default mode if the class defines a mode field
            if "mode" in settings_cls.model_fields:
                default_mode = settings_cls.model_fields["mode"].get_default()
                if default_mode:
                    default_file = settings_cls._get_yaml_default_from_mode(default_mode)
        if not default_file:
            default_file = settings_cls.model_fields["yaml_default"].get_default()

        # construct config files list
        config_files = []
        if default_file:
            config_files.append(default_file)  # default file has lowest priority
        config_files.extend(self.current_state.get("yaml_extra", []))

        merged_data = self._read_files(config_files)
        return merged_data


class DynamicYamlMixInForSettings:
    """Mix-in class for settings providing dynamic yaml loading as lowest priority source.

    NOTE: This class must come FIRST in the MRO such that `yaml_extra` can be processed before
    since otherwise we cannot load default values from the `yaml_extra` first.

    This mix-in enforces the following precedence order (highest -> lowest priority):
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
    together in the order they are provided. Hence, the following order (lowest -> highest priority)
    for multiple yaml config files is:
        - default yaml provided as ``yaml_default`` field (or yaml retrieved from the mode)
        - argument 0 of ``yaml_extra``
        - argument 1 of ``yaml_extra``
        - ...
        - last argument of ``yaml_extra`` (highest priority, last yaml getting merged into config)
    """

    # should be set as field by the child class! mode is a simple switch to control the default
    # config via the yaml_default field. It can be set by the child class to provide a
    # convenient, user-facing config to switch between different default configs.
    # mode: str

    # should be overwritten by the child class!
    @classmethod
    def _get_yaml_default_from_mode(cls, mode: Optional[str]) -> Optional[str]:
        """Get the default yaml file from the mode or return None if no default yaml is found."""
        return None

    yaml_default: str = Field(
        default="",
        description="The default yaml file to load. This field can be used to fully customize the "
        "configuration and behavior of the AutoDeploy Pipeline. Expert Use Only!",
    )

    yaml_extra: List[PathType] = Field(
        default_factory=list,
        description="Additional yaml config files to load to be merged into the default yaml file "
        'with higher priority. "Later" files have higher priority. Should be used with care!',
    )

    @model_validator(mode="after")
    def validate_mode_and_yaml_default_not_both_provided(self):
        """Validate that both mode and yaml_default are not provided simultaneously.

        When mode is specified (non-empty), it automatically selects a yaml_default
        via the _mode_to_yaml_default mapping. If yaml_default is also explicitly
        provided, this creates ambiguity about which configuration should take precedence.
        """
        if {"mode", "yaml_default"} <= self.model_fields_set:
            raise ValueError(
                "Cannot provide both 'mode' and 'yaml_default' simultaneously. "
                "The 'mode' field automatically selects a default YAML configuration, "
                "so providing 'yaml_default' explicitly creates ambiguity. "
                f"Either use mode='{self.mode}' OR yaml_default='{self.yaml_default}', but not both."
            )

        return self

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
