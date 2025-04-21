import argparse
import os
import subprocess


def install_python_dependencies(llm_src):
    subprocess.run(
        f"cd {llm_src} && pip3 install --retries 1 -r requirements-dev.txt",
        shell=True,
        check=True)
    subprocess.run(
        f"pip3 install --force-reinstall --no-deps {llm_src}/../tensorrt_llm-*.whl",
        shell=True,
        check=True)
    subprocess.run(
        "pip3 install --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-tensorrt-pypi/simple "
        "--ignore-installed trt-test-db==1.8.5+bc6df7",
        shell=True,
        check=True)


def verify_l0_test_lists(llm_src):
    test_db_path = f"{llm_src}/tests/integration/test_lists/test-db"
    test_list = f"{llm_src}/l0_test.txt"

    # Remove dynamically generated perf tests
    subprocess.run(f"rm -f {test_db_path}/*perf*", shell=True, check=True)
    subprocess.run(
        f"trt-test-db -d {test_db_path} --test-names --output {test_list}",
        shell=True,
        check=True)

    subprocess.run(
        f"cd {llm_src}/tests/integration/defs && "
        f"pytest --apply-test-list-correction --test-list={test_list} --co -q",
        shell=True,
        check=True)


def verify_qa_test_lists(llm_src):
    test_qa_path = f"{llm_src}/tests/integration/test_lists/qa"
    # Remove dynamically generated perf tests
    subprocess.run(f"rm -f {test_qa_path}/*perf*", shell=True, check=True)
    test_def_files = subprocess.check_output(
        f"ls -d {test_qa_path}/*.txt", shell=True).decode().strip().split('\n')
    for test_def_file in test_def_files:
        subprocess.run(
            f"cd {llm_src}/tests/integration/defs && "
            f"pytest --apply-test-list-correction --test-list={test_def_file} --co -q",
            shell=True,
            check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Check test lists for L0 and QA.")
    parser.add_argument("--l0",
                        action="store_true",
                        help="Enable L0 test list verification.")
    parser.add_argument("--qa",
                        action="store_true",
                        help="Enable QA test list verification.")
    args = parser.parse_args()
    llm_src = os.path.realpath("TensorRT-LLM/src")

    install_python_dependencies(llm_src)
    # Verify L0 test lists
    if args.l0:
        verify_l0_test_lists(llm_src)
    else:
        print("Skipping L0 test list verification.")
    # Verify QA test lists
    if args.qa:
        verify_qa_test_lists(llm_src)
    else:
        print("Skipping QA test list verification.")


if __name__ == "__main__":
    main()
