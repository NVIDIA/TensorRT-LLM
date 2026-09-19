#!/usr/bin/env python3
"""Extract required_devices and device_type from pytest markers in a test file.

Parses the target class and function in the given test file and reports
constraints implied by TensorRT-LLM pytest markers.

Output (stdout): JSON with keys:
  required_devices  int   - minimum GPUs needed (default 1)
  device_type       str   - e.g. "Blackwell (B200/GB200)", "" if unconstrained
  sources           dict  - {"required_devices": "derived"|"default",
                             "device_type":       "derived"|"default"}
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

# Maps marker name / alias → (field, value)
_DEVICE_TYPE_MARKERS = {
    "skip_pre_blackwell": "Blackwell (B200/GB200)",
    "skip_pre_hopper": "Hopper+ (H100/H200)",
}


def _parse_test_cmd(test_cmd: str) -> tuple[str, str | None, str | None]:
    """Return (file_path, class_name, test_name) from a pytest node-id string."""
    # Strip leading 'pytest' / 'python -m pytest' tokens
    parts = test_cmd.split()
    node_id = next((p for p in parts if p.endswith(".py") or ".py::" in p), None)
    if node_id is None:
        raise ValueError(f"Cannot find a .py path in test_cmd: {test_cmd!r}")

    file_part, *rest = node_id.split("::")
    class_name = rest[0] if len(rest) >= 2 else None
    test_name = rest[1] if len(rest) >= 2 else (rest[0] if rest else None)
    return file_part, class_name, test_name


def _decorator_names(decorator: ast.expr) -> list[str]:
    """Flatten a decorator AST node to a list of candidate name strings."""
    names = []
    if isinstance(decorator, ast.Name):
        names.append(decorator.id)
    elif isinstance(decorator, ast.Attribute):
        names.append(decorator.attr)
        names.append(f"{_attr_chain(decorator)}")
    elif isinstance(decorator, ast.Call):
        names.extend(_decorator_names(decorator.func))
    return names


def _attr_chain(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_attr_chain(node.value)}.{node.attr}"
    return ""


def _get_int_arg(call: ast.Call) -> int | None:
    """Return the first integer argument of a Call node, if any."""
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            return arg.value
    for kw in call.keywords:
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
            return kw.value.value
    return None


def _extract_from_decorators(
    decorators: list[ast.expr],
) -> tuple[int | None, str | None]:
    """Return (required_devices, device_type) found in a decorator list."""
    required_devices: int | None = None
    device_type: str | None = None

    for dec in decorators:
        names = _decorator_names(dec)

        # skip_less_device(N)
        if any(n in ("skip_less_device",) for n in names):
            if isinstance(dec, ast.Call):
                val = _get_int_arg(dec)
                if val is not None:
                    required_devices = val

        # skip_pre_blackwell / skip_pre_hopper
        for marker, dtype in _DEVICE_TYPE_MARKERS.items():
            if any(marker in n for n in names):
                device_type = dtype

        # pytest.mark.skipif(condition, reason="... requires N GPU(s)...")
        if any("skipif" in n for n in names) and isinstance(dec, ast.Call):
            for kw in dec.keywords:
                if kw.arg == "reason" and isinstance(kw.value, ast.Constant):
                    m = re.search(r"(\d+)\s+GPU", kw.value.value, re.IGNORECASE)
                    if m:
                        required_devices = int(m.group(1))

    return required_devices, device_type


def extract_markers(
    test_file: str,
    class_name: str | None,
    test_name: str | None,
) -> dict:
    path = Path(test_file)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    tree = ast.parse(path.read_text(), filename=test_file)

    class_rd: int | None = None
    class_dt: str | None = None
    func_rd: int | None = None
    func_dt: str | None = None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and (class_name is None or node.name == class_name):
            c_rd, c_dt = _extract_from_decorators(node.decorator_list)
            if c_rd is not None:
                class_rd = c_rd
            if c_dt is not None:
                class_dt = c_dt

            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    test_name is None or item.name == test_name
                ):
                    f_rd, f_dt = _extract_from_decorators(item.decorator_list)
                    if f_rd is not None:
                        func_rd = f_rd
                    if f_dt is not None:
                        func_dt = f_dt

        # Also handle module-level functions (no enclosing class)
        if (
            class_name is None
            and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and (test_name is None or node.name == test_name)
        ):
            f_rd, f_dt = _extract_from_decorators(node.decorator_list)
            if f_rd is not None:
                func_rd = f_rd
            if f_dt is not None:
                func_dt = f_dt

    # Function-level markers take precedence over class-level
    required_devices = func_rd if func_rd is not None else class_rd
    device_type = func_dt if func_dt is not None else class_dt

    return {
        "required_devices": required_devices if required_devices is not None else 1,
        "device_type": device_type or "",
        "sources": {
            "required_devices": "derived" if required_devices is not None else "default",
            "device_type": "derived" if device_type is not None else "default",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--test-cmd",
        help="Full pytest command (file path and node-id are parsed automatically)",
    )
    group.add_argument("--test-file", help="Path to the test file")
    parser.add_argument("--class-name", help="Test class name (optional)")
    parser.add_argument("--test-name", help="Test function name (optional)")
    args = parser.parse_args()

    if args.test_cmd:
        try:
            test_file, class_name, test_name = _parse_test_cmd(args.test_cmd)
        except ValueError as exc:
            print(json.dumps({"error": str(exc)}))
            sys.exit(1)
    else:
        test_file = args.test_file
        class_name = args.class_name
        test_name = args.test_name

    try:
        result = extract_markers(test_file, class_name, test_name)
    except FileNotFoundError as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
