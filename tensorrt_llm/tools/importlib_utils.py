# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Optional, Union


def import_custom_module_from_file(
        custom_module_path: Union[str, Path]) -> Optional[ModuleType]:
    """Import a custom module from a single file.

    Args:
        custom_module_path (Union[str, Path]): The path to the custom module file.

    Returns:
        The imported module object.

    Raises:
        ImportError: If the module cannot be imported.
    """
    if isinstance(custom_module_path, str):
        custom_module_path = Path(custom_module_path)
    print(f"Importing custom module from file: {custom_module_path}")

    # Import single Python file
    module = None
    spec = importlib.util.spec_from_file_location(custom_module_path.stem,
                                                  str(custom_module_path))
    if spec is not None:
        module = importlib.util.module_from_spec(spec)
        if spec.loader is not None:
            spec.loader.exec_module(module)
            print(
                f"Successfully imported custom module from file: {custom_module_path}"
            )
        else:
            raise ImportError(
                f"Failed to import custom module from {custom_module_path}")
    else:
        raise ImportError(
            f"Failed to import custom module from {custom_module_path}")
    return module


def import_custom_module_from_dir(
        custom_module_path: Union[str, Path]) -> Optional[ModuleType]:
    """Import a custom module from a directory.

    Args:
        custom_module_path (Union[str, Path]): The path to the custom module directory.

    Returns:
        The imported module object.

    Raises:
        ImportError: If the module cannot be imported.

    Note:
        This function will add the parent directory of the custom module directory to sys.path.
        This is useful for importing modules that are not in the current working directory.
    """
    if isinstance(custom_module_path, str):
        custom_module_path = Path(custom_module_path)
    print(f"Importing custom module from directory: {custom_module_path}")

    # Import directory as a package
    # Add the parent directory to sys.path so we can import the package
    import sys
    parent_dir = str(custom_module_path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Import the package
    module = None
    package_name = custom_module_path.name
    try:
        module = importlib.import_module(package_name)
        print(
            f"Successfully imported custom module from directory: {custom_module_path}"
        )
    except ImportError as e:
        raise ImportError(
            f"Failed to import package {package_name} from {custom_module_path}: {e}"
        )
    return module


def import_custom_module(
        custom_module_path: Union[str, Path]) -> Optional[ModuleType]:
    """Import a custom module from a file or directory.

    Args:
        custom_module_path (Union[str, Path]): The path to the custom module file or directory.

    Returns:
        The imported module object.

    Raises:
        ImportError: If the module cannot be imported.
        FileNotFoundError: If the custom module path does not exist.
    """
    if isinstance(custom_module_path, str):
        custom_module_path = Path(custom_module_path)
    print(f"Importing custom module from: {custom_module_path}")

    if custom_module_path.exists():
        if custom_module_path.is_file():
            return import_custom_module_from_file(custom_module_path)
        elif custom_module_path.is_dir():
            return import_custom_module_from_dir(custom_module_path)
        else:
            raise FileNotFoundError(
                f"Custom module path {custom_module_path} is neither a file nor a directory."
            )
    else:
        raise FileNotFoundError(
            f"Custom module path {custom_module_path} does not exist.")
    return None
