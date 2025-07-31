import importlib.util
from pathlib import Path
from typing import Union


def import_custom_module_from_file(custom_module_path: Union[str, Path]):
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


def import_custom_module_from_dir(custom_module_path: Union[str, Path]):
    if isinstance(custom_module_path, str):
        custom_module_path = Path(custom_module_path)
    print(f"Importing custom module from directory: {custom_module_path}")

    # Import directory as a package
    # Add the parent directory to sys.path so we can import the package
    import sys
    parent_dir = custom_module_path.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

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


def import_custom_module(custom_module_path: Union[str, Path]):
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
