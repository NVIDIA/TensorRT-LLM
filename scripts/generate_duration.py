import argparse
import csv
import glob
import json
import os

# Default CSV columns from pytest-csv:
#   id, module, name, file, doc, markers, status, message, duration
STATUS_COLUMN = 6
DURATION_COLUMN = 8

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
NEW_TEST_DURATION = args.duration_file

# report.csv contains merged results (regular, isolation, and rerun tests)
all_csv_files = sorted(glob.glob(os.path.join(TEST_RESULTS_DIR,
                                              '*/report.csv')))

print(f"TEST_RESULTS_DIR: {TEST_RESULTS_DIR}")
print(f"Found {len(all_csv_files)} CSV report file(s)")

test_durations = {}
passed_count = 0
skipped_count = 0

for report_csv in all_csv_files:
    print(f"Processing {report_csv}...")
    try:
        with open(report_csv, 'r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if len(row) <= max(STATUS_COLUMN, DURATION_COLUMN):
                    continue

                status = row[STATUS_COLUMN].strip()
                if status != 'passed':
                    skipped_count += 1
                    continue

                test_id = row[0].strip()
                duration_str = row[DURATION_COLUMN].strip()

                # Remove stage name prefix (everything up to first '/')
                test_name = test_id.split('/', 1)[-1]

                try:
                    duration = float(duration_str)
                except ValueError:
                    # Fall back to last column if the fixed index doesn't work
                    try:
                        duration = float(row[-1].strip())
                    except ValueError:
                        print(
                            f"  Warning: Could not parse duration for {test_name}. Skipping."
                        )
                        continue

                test_durations[test_name] = duration
                passed_count += 1
    except Exception as e:
        print(f"  Warning: Failed to process {report_csv}: {e}")

# Write the test durations to the output file
with open(NEW_TEST_DURATION, 'w') as file:
    json.dump(test_durations, file, indent=3)

print(f"\nSummary:")
print(f"  CSV files processed : {len(all_csv_files)}")
print(f"  Passed rows collected : {passed_count}")
print(f"  Non-passed rows skipped: {skipped_count}")
print(f"  Unique tests in output : {len(test_durations)}")
print(f"  Output written to      : {NEW_TEST_DURATION}")
