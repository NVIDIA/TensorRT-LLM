"""
Test that prebuilt wheel extraction includes all necessary Python files.

"""
from pathlib import Path


def test_cpp_extension_wrapper_files_exist():
    """Verify that C++ extension wrapper Python files were extracted from prebuilt wheel."""
    import tensorrt_llm

    trtllm_root = Path(tensorrt_llm.__file__).parent

    # C++ extensions that have Python wrapper files generated during build
    required_files = {
        'deep_gemm':
        ['__init__.py', 'testing/__init__.py', 'utils/__init__.py'],
        'deep_ep': ['__init__.py', 'buffer.py', 'utils.py'],
        'flash_mla': ['__init__.py', 'flash_mla_interface.py'],
    }

    missing_files = []
    for ext_dir, files in required_files.items():
        for file in files:
            file_path = trtllm_root / ext_dir / file
            if not file_path.exists():
                missing_files.append(str(file_path.relative_to(trtllm_root)))

    assert not missing_files, (
        f"Missing Python wrapper files for C++ extensions: {missing_files}\n"
        f"This indicates setup.py may not be extracting Python files from prebuilt wheels.\n"
        f"Check setup.py extract_from_precompiled() function.")


def test_cpp_extensions_importable():
    """Verify that C++ extension wrappers can be imported successfully."""
    import_tests = [
        ('tensorrt_llm.deep_gemm', 'fp8_mqa_logits'),
        ('tensorrt_llm.deep_ep', 'Buffer'),
        ('tensorrt_llm.flash_mla', 'flash_mla_interface'),
    ]

    failed_imports = []
    for module_name, attr_name in import_tests:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            if not hasattr(module, attr_name):
                failed_imports.append(
                    f"{module_name}.{attr_name} (attribute not found)")
        except ImportError as e:
            failed_imports.append(f"{module_name} (ImportError: {e})")

    assert not failed_imports, (
        f"Failed to import C++ extension wrappers: {failed_imports}\n"
        f"This may indicate missing Python files or circular import issues.")


if __name__ == '__main__':
    print("Testing C++ extension wrapper files...")
    test_cpp_extension_wrapper_files_exist()
    print("✅ All required Python files exist")

    print("\nTesting C++ extension imports...")
    test_cpp_extensions_importable()
    print("✅ All imports successful")

    print("\n✅ All tests passed!")
