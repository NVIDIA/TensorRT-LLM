"""
This script is used to verify test lists for L0, QA, and waives file.

Usage:
When in a development or container environment, run the following command:
    python $LLM_ROOT/scripts/check_test_list.py --l0 --qa --waive

Options:
--l0:    Check only the L0 tests located in $LLM_ROOT/tests/integration/test_list/test_db/*.yml.
--qa:    Check only the QA tests under $LLM_ROOT/tests/integration/test_list/*.txt.
--waive: Check only the tests in $LLM_ROOT/tests/integration/test_list/waives.txt.

Note:
All the perf tests will be excluded since they are generated dynamically.
"""
import argparse
import os
import subprocess

# The markers in our test lists, need to be preprocess before checking
MARKER_LIST_IN_TEST = [" TIMEOUT"]


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

    # Remove the duplicated test names
    cleaned_lines = set()
    with open(test_list, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Remove markers and rest of the line if present
        cleaned_line = line.strip()

        # Handle ISOLATION marker removal (including comma patterns)
        if 'ISOLATION,' in cleaned_line:
            # Case: "ISOLATION,OTHER_MARKER" -> remove "ISOLATION,"
            cleaned_line = cleaned_line.replace('ISOLATION,', '').strip()
        elif ',ISOLATION' in cleaned_line:
            # Case: "OTHER_MARKER,ISOLATION" -> remove ",ISOLATION"
            cleaned_line = cleaned_line.replace(',ISOLATION', '').strip()
        elif ' ISOLATION' in cleaned_line:
            # Case: standalone "ISOLATION" -> remove " ISOLATION"
            cleaned_line = cleaned_line.replace(' ISOLATION', '').strip()

        # Handle other markers (like TIMEOUT) - remove marker and everything after it
        for marker in MARKER_LIST_IN_TEST:
            if marker in cleaned_line and marker != " ISOLATION":
                cleaned_line = cleaned_line.split(marker, 1)[0].strip()
                break

        if cleaned_line:
            cleaned_lines.add(cleaned_line)

    with open(test_list, "w") as f:
        f.writelines(f"{line}\n" for line in sorted(cleaned_lines))

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


def verify_waive_list(llm_src):
    waives_list_path = f"{llm_src}/tests/integration/test_lists/waives.txt"
    # Remove prefix and markers in wavies.txt
    processed_lines = set()
    with open(waives_list_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Skip Perf tests due to they are dynamically generated
        if "perf/test_perf.py" in line:
            continue

        # Check for SKIP marker in waives.txt and split by the first occurrence
        line = line.split(" SKIP ", 1)[0].strip()

        # If the line starts with 'full:', process it
        if line.startswith("full:"):
            line = line.split("/", 1)[1].lstrip("/")

        processed_lines.add(line)

    # Write the processed lines to a tmp file
    tmp_waives_file = f"{llm_src}/processed_waive_list.txt"
    with open(tmp_waives_file, "w") as f:
        f.writelines(f"{line}\n" for line in sorted(processed_lines))

    subprocess.run(
        f"cd {llm_src}/tests/integration/defs && "
        f"pytest --apply-test-list-correction --test-list={tmp_waives_file} --co -q",
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
    parser.add_argument("--waive",
                        action="store_true",
                        help="Enable test list verification for waive file.")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    llm_src = os.path.abspath(os.path.join(script_dir, "../"))

    install_python_dependencies(llm_src)
    # Verify L0 test lists
    if args.l0:
        print("Starting L0 test list verification...")
        verify_l0_test_lists(llm_src)
    else:
        print("Skipping L0 test list verification.")

    # Verify QA test lists
    if args.qa:
        print("Starting QA test list verification...")
        verify_qa_test_lists(llm_src)
    else:
        print("Skipping QA test list verification.")

    # Verify waive test lists
    if args.waive:
        print("Starting waive list verification...")
        verify_waive_list(llm_src)
    else:
        print("Skipping waive list verification.")


if __name__ == "__main__":
    main()
