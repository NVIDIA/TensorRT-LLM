import argparse
import glob
import json
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate test duration file.")
parser.add_argument(
    "--duration-file",
    type=str,
    default="new_test_duration.json",
    help="Path to the output duration file (default: new_test_duration.json)")
args = parser.parse_args()

# Define the directory containing the test result folders
TEST_RESULTS_DIR = os.getcwd()

# Define the output file paths
FULL_RESULT_LOG = "full_result.log"
NEW_TEST_DURATION = args.duration_file

# Step 1: Prepare full_result.log
with open(FULL_RESULT_LOG, 'w') as full_result_file:
    print(f"TEST_RESULTS_DIR: {TEST_RESULTS_DIR}")
    for report_csv in glob.glob(os.path.join(TEST_RESULTS_DIR, '*/report.csv')):
        print(f"Processing {report_csv}...")
        with open(report_csv, 'r') as csv_file:
            for line in csv_file:
                if 'passed' in line:
                    full_result_file.write(line)

# Step 2: Generate new_test_duration.json
test_durations = {}

# Read the full_result.log file line by line
with open(FULL_RESULT_LOG, 'r') as file:
    for line in file:
        # Extract the first column and the last column
        columns = line.strip().split(',')
        first_column = columns[0]
        last_column = columns[-1]

        # Remove from left to first '/' in the first column
        test_name = first_column.split('/', 1)[-1]
        # Replace \"\" with \" and ]\" with ] in case we got these in names from report.csv
        # which will broken the json parse
        test_name = test_name.replace(']\"', ']').replace('\"\"', '\"')

        try:
            last_column = float(last_column)
        except ValueError:
            print(
                f"Warning: Could not convert {last_column} to float. Skipping.")
            continue

        # Add to the test duration dictionary
        test_durations[test_name] = last_column

# Write the test durations to the new test duration file
with open(NEW_TEST_DURATION, 'w') as file:
    json.dump(test_durations, file, indent=3)

print(f"Test durations have been written to {NEW_TEST_DURATION}")
