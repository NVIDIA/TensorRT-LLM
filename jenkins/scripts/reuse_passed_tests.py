import argparse
import os
import sys
import xml.etree.ElementTree as ET

import test_rerun


def get_passed_tests(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist")
        return

    # Parse the JUnit XML file and extract passed test names
    passed_tests = []
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
        suite = root.find('testsuite')
        for testcase in suite.iter('testcase'):
            # Check test status
            has_failure = testcase.find('failure') is not None
            has_error = testcase.find('error') is not None
            has_skipped = testcase.find('skipped') is not None
            if not has_failure and not has_error and not has_skipped:
                # Parse the test name
                classname = testcase.attrib.get('classname', '')
                name = testcase.attrib.get('name', '')
                filename = testcase.attrib.get('file', '')
                test_name = test_rerun.parse_name(classname, name, filename)
                passed_tests.append(test_name)
    except Exception as e:
        print(f"Failed to parse {input_file}: {e}")
        return

    # Write passed test names to output file, one per line
    with open(output_file, 'w') as f:
        for test in passed_tests:
            f.write(test + '\n')


def remove_passed_tests(input_file, passed_tests_file):
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist")
        return
    if not os.path.exists(passed_tests_file):
        print(f"Passed tests file {passed_tests_file} does not exist")
        return

    passed_tests = []
    # Read passed tests from file
    with open(passed_tests_file, 'r') as f:
        for line in f:
            passed_tests.append(line.strip())

    tests_to_keep = []
    # Remove passed tests from input file
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip() not in passed_tests:
                tests_to_keep.append(line.strip())

    # Delete input file
    try:
        os.remove(input_file)
    except Exception as e:
        print(f"Failed to delete {input_file}: {e}")
    # Write tests to keep to input file
    with open(input_file, 'w') as f:
        for test in tests_to_keep:
            f.write(test + '\n')


if __name__ == '__main__':
    if (sys.argv[1] == "get_passed_tests"):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input-file',
                            required=True,
                            help='Input XML file containing test results')
        parser.add_argument('--output-file',
                            required=True,
                            help='Output file to write passed tests')
        args = parser.parse_args(sys.argv[2:])
        get_passed_tests(args.input_file, args.output_file)
    elif (sys.argv[1] == "remove_passed_tests"):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input-file',
                            required=True,
                            help='Input XML file containing test results')
        parser.add_argument('--passed-tests-file',
                            required=True,
                            help='File containing passed tests')
        args = parser.parse_args(sys.argv[2:])
        remove_passed_tests(args.input_file, args.passed_tests_file)
