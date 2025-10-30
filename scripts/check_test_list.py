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
        f"pytest --test-list={test_list} --output-dir={llm_src} -s --co -q",
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
            f"pytest --test-list={test_def_file} --output-dir={llm_src} -s --co -q",
            shell=True,
            check=True)
        # append all the test_def_file to qa_test.txt
        with open(f"{llm_src}/qa_test.txt", "a") as f:
            with open(test_def_file, "r") as test_file:
                lines = test_file.readlines()
                for line in lines:
                    # Remove 'TIMEOUT' marker and strip spaces
                    cleaned_line = line.split(" TIMEOUT ", 1)[0].strip()
                    if cleaned_line:
                        f.write(f"{cleaned_line}\n")


def verify_waive_list(llm_src, args):
    waives_list_path = f"{llm_src}/tests/integration/test_lists/waives.txt"
    dup_cases_record = f"{llm_src}/dup_cases.txt"
    non_existent_cases_record = f"{llm_src}/nonexits_cases.json"
    # Remove prefix and markers in wavies.txt
    dedup_lines = {
    }  # Track all occurrences: processed_line -> [(line_no, original_line), ...]
    processed_lines = set()
    with open(waives_list_path, "r") as f:
        lines = f.readlines()

    for line_no, line in enumerate(lines, 1):
        original_line = line.strip()
        line = line.strip()

        if not line:
            continue

        # Skip Perf tests due to they are dynamically generated
        if "perf/test_perf.py" in line:
            continue

        # Check for SKIP marker in waives.txt and split by the first occurrence
        line = line.split(" SKIP", 1)[0].strip()

        # Track all occurrences of each processed line
        if line in dedup_lines:
            dedup_lines[line].append((line_no, original_line))
        else:
            dedup_lines[line] = [(line_no, original_line)]

        # If the line starts with 'full:', process it
        if line.startswith("full:"):
            line = line.split("/", 1)[1].lstrip("/")

        # Skip unittests due to we don't need to have an entry in test-db yml
        if line.startswith("unittest/"):
            continue

        # Check waived cases also in l0_text.txt and qa_text.txt
        found_in_l0_qa = False
        if args.l0:
            with open(f"{llm_src}/l0_test.txt", "r") as f:
                l0_lines = f.readlines()
                for l0_line in l0_lines:
                    if line == l0_line.strip():
                        found_in_l0_qa = True
                        break
        if args.qa:
            with open(f"{llm_src}/qa_test.txt", "r") as f:
                qa_lines = f.readlines()
                for qa_line in qa_lines:
                    if line == qa_line.strip():
                        found_in_l0_qa = True
                        break
        if not found_in_l0_qa:
            with open(non_existent_cases_record, "a") as f:
                f.write(
                    f"Non-existent test name in l0 or qa list found in waives.txt: {line}\n"
                )

        processed_lines.add(line)

    # Write duplicate report after processing all lines
    for processed_line, occurrences in dedup_lines.items():
        if len(occurrences) > 1:
            with open(dup_cases_record, "a") as f:
                f.write(
                    f"Duplicate waive records found for '{processed_line}' ({len(occurrences)} occurrences):\n"
                )
                for i, (line_no, original_line) in enumerate(occurrences, 1):
                    f.write(
                        f"  Occurrence {i} at line {line_no}: '{original_line}'\n"
                    )
                f.write(f"\n")

    # Write the processed lines to a tmp file
    tmp_waives_file = f"{llm_src}/processed_waive_list.txt"
    with open(tmp_waives_file, "w") as f:
        f.writelines(f"{line}\n" for line in sorted(processed_lines))

    subprocess.run(
        f"cd {llm_src}/tests/integration/defs && "
        f"pytest --test-list={tmp_waives_file} --output-dir={llm_src} -s --co -q",
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
    pass_flag = True
    # Verify L0 test lists
    if args.l0:
        print("-----------Starting L0 test list verification...-----------",
              flush=True)
        verify_l0_test_lists(llm_src)
    else:
        print("-----------Skipping L0 test list verification.-----------",
              flush=True)

    # Verify QA test lists
    if args.qa:
        print("-----------Starting QA test list verification...-----------",
              flush=True)
        verify_qa_test_lists(llm_src)
    else:
        print("-----------Skipping QA test list verification.-----------",
              flush=True)

    # Verify waive test lists
    if args.waive:
        print("-----------Starting waive list verification...-----------",
              flush=True)
        verify_waive_list(llm_src, args)
    else:
        print("-----------Skipping waive list verification.-----------",
              flush=True)

    invalid_json_file = os.path.join(llm_src, "invalid_tests.json")
    if os.path.isfile(invalid_json_file) and os.path.getsize(
            invalid_json_file) > 0:
        print("Invalid cases:")
        with open(invalid_json_file, "r") as f:
            print(f.read())
        print("Invalid test names found, please correct them first!!!\n")
        pass_flag = False

    duplicate_cases_file = os.path.join(llm_src, "dup_cases.txt")
    if os.path.isfile(duplicate_cases_file) and os.path.getsize(
            duplicate_cases_file) > 0:
        print("Duplicate cases found:")
        with open(duplicate_cases_file, "r") as f:
            print(f.read())
        print(
            "Duplicate test names found in waives.txt, please delete one or combine them first!!!\n"
        )
        pass_flag = False

    non_existent_cases_file = os.path.join(llm_src, "nonexits_cases.json")
    if os.path.isfile(non_existent_cases_file) and os.path.getsize(
            non_existent_cases_file) > 0:
        print("Non-existent cases found in waives.txt:")
        with open(non_existent_cases_file, "r") as f:
            print(f.read())
        print(
            "Non-unit test test name in waives.txt but not in l0 test list or qa list, please delete them first!!!\n"
        )
        pass_flag = False

    if not pass_flag:
        exit(1)


if __name__ == "__main__":
    main()
