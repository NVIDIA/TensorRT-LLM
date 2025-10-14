"""The interface for all export patches.

This module defines the base classes and interfaces for all export patches.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Type, Union, final

from pydantic import BaseModel, Field

from ..utils.logger import ad_logger


class ExportPatchError(Exception):
    """An exception raised when an export patch fails."""

    pass


class ExportPatchConfig(BaseModel):
    """Base configuration class for export patches."""

    model_config = {
        "extra": "allow",  # Allow subclasses to add more fields
    }

    enabled: bool = Field(
        default=True,
        description="Whether to enable this patch.",
    )
    skip_on_error: bool = Field(
        default=False,
        description="Whether to skip the patch if an error occurs during application.",
    )


class DisabledExportPatchConfig(ExportPatchConfig):
    """Standard configuration for an export patch that is disabled by default."""

    enabled: bool = Field(
        default=False,
        description="Whether to enable this patch.",
    )


class BaseExportPatch(ABC):
    """Base class for all export patches.

    Export patches are context managers that apply temporary modifications
    to the global state during torch.export, then revert them afterwards.
    """

    config: ExportPatchConfig
    _patch_key: str  # Set by ExportPatchRegistry.register() decorator

    @classmethod
    def get_patch_key(cls) -> str:
        """Get the short name of the patch."""
        if hasattr(cls, "_patch_key"):
            return cls._patch_key
        raise NotImplementedError(
            f"Patch class {cls.__name__} must be registered with ExportPatchRegistry.register() "
            "or manually implement get_patch_key()"
        )

    @classmethod
    def get_config_class(cls) -> Type[ExportPatchConfig]:
        """Get the configuration class for the patch."""
        return ExportPatchConfig

    @final
    def __init__(self, config: ExportPatchConfig):
        """Initialize the patch.

        Args:
            config: The configuration for the patch.
        """
        if not isinstance(config, self.get_config_class()):
            config = self.get_config_class()(**config.model_dump())
        self.config = config
        self.original_values = {}
        self._post_init()

    def _post_init(self):
        """Post-initialization hook that can be overridden by subclasses."""
        pass

    @final
    @classmethod
    def from_kwargs(cls, **kwargs) -> "BaseExportPatch":
        """Create a patch from kwargs."""
        config = cls.get_config_class()(**kwargs)
        return cls(config=config)

    @final
    def __enter__(self):
        """Enter the context manager and apply the patch."""
        if not self.config.enabled:
            ad_logger.debug(f"Patch {self.get_patch_key()} is disabled, skipping")
            return self

        try:
            ad_logger.debug(f"Applying patch: {self.get_patch_key()}")
            self._apply_patch()
        except Exception as e:
            error_msg = f"Patch {self.get_patch_key()} failed to apply"
            if self.config.skip_on_error:
                ad_logger.warning(f"{error_msg}: {e}")
            else:
                raise ExportPatchError(error_msg) from e

        return self

    @final
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and revert the patch."""
        if not self.config.enabled:
            return

        try:
            ad_logger.debug(f"Reverting patch: {self.get_patch_key()}")
            self._revert_patch()
        except Exception as e:
            error_msg = f"Patch {self.get_patch_key()} failed to revert"
            if self.config.skip_on_error:
                ad_logger.warning(f"{error_msg}: {e}")
            else:
                raise ExportPatchError(error_msg) from e

    @abstractmethod
    def _apply_patch(self):
        """Apply the patch. Should store original values in self.original_values."""
        pass

    @abstractmethod
    def _revert_patch(self):
        """Revert the patch using stored original values."""
        pass


class DisabledBaseExportPatch(BaseExportPatch):
    """A base class for export patches that are disabled by default."""

    config: DisabledExportPatchConfig

    @classmethod
    def get_config_class(cls) -> Type[ExportPatchConfig]:
        """Get the configuration class for the patch."""
        return DisabledExportPatchConfig


class ContextManagerPatch(BaseExportPatch):
    """A patch that wraps an existing context manager.

    This allows easy registration of context managers as patches without
    having to implement the full BaseExportPatch interface.

    Subclasses must implement `init_context_manager()` to return the context manager.
    """

    def _post_init(self):
        self.context_manager: Any = None

    @abstractmethod
    def init_context_manager(self) -> Any:
        """Initialize and return the context manager.

        Returns:
            A context manager that will be used during export.
        """
        pass

    def _apply_patch(self):
        """Apply the patch by entering the context manager."""
        self.context_manager = self.init_context_manager()
        self.context_manager.__enter__()

    def _revert_patch(self):
        """Revert the patch by exiting the context manager."""
        if self.context_manager is not None:
            self.context_manager.__exit__(None, None, None)
            self.context_manager = None


class ExportPatchRegistry:
    """Registry for export patches."""

    _registry: Dict[str, Type[BaseExportPatch]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseExportPatch]], Type[BaseExportPatch]]:
        """Register a patch class with the given name."""

        def inner(patch_cls: Type[BaseExportPatch]) -> Type[BaseExportPatch]:
            cls._registry[name] = patch_cls
            # Auto-store the patch key as a class attribute
            patch_cls._patch_key = name
            return patch_cls

        return inner

    @classmethod
    def get(cls, name: str) -> Type[BaseExportPatch]:
        """Get a patch class by name."""
        if not cls.has(name):
            raise ValueError(f"Unknown patch: {name}")
        return cls._registry[name]

    @classmethod
    def get_config_class(cls, name: str) -> Type[ExportPatchConfig]:
        """Get the configuration class for a patch by name."""
        return cls.get(name).get_config_class()

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a patch is registered."""
        return name in cls._registry

    @classmethod
    def create_patch(
        cls, name: str, config: Union[ExportPatchConfig, Dict[str, Any]]
    ) -> BaseExportPatch:
        """Create a patch instance by name."""
        patch_cls = cls.get(name)
        if isinstance(config, dict):
            config = patch_cls.get_config_class()(**config)
        return patch_cls(config)

    @classmethod
    def list_patches(cls) -> List[str]:
        """List all registered patch names."""
        return list(cls._registry.keys())


@contextmanager
def apply_export_patches(
    patch_configs: Optional[Dict[str, Union[ExportPatchConfig, Dict[str, Any]]]] = None,
    patch_list: Optional[List[str]] = None,
):
    """Context manager to apply multiple patches.

    Args:
        patch_configs: Dict mapping patch names to their configurations.
    """
    # Validate that both patch_configs and patch_list are not provided simultaneously
    if patch_configs is not None and patch_list is not None:
        raise ValueError("Cannot specify both patch_configs and patch_list. Use only one.")

    # Handle patch configuration
    if patch_list is not None:
        # Convert patch_list to patch_configs format
        patch_configs = {patch_name: {} for patch_name in patch_list}
    elif patch_configs is None:
        # Default patch configurations - apply all registered patches with default settings
        patch_configs = {patch_name: {} for patch_name in ExportPatchRegistry.list_patches()}

    # Create patch instances
    patches = [ExportPatchRegistry.create_patch(k, conf) for k, conf in patch_configs.items()]

    # Apply patches using nested context managers
    if not patches:
        yield
        return

    def _apply_patches(remaining_patches):
        if not remaining_patches:
            yield
            return

        patch = remaining_patches[0]
        with patch:
            yield from _apply_patches(remaining_patches[1:])

    # log applied patches
    ad_logger.debug(
        f"applying export patches: {', '.join([patch.get_patch_key() for patch in patches])}"
    )

    yield from _apply_patches(patches)
