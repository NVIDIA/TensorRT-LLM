import argparse
import sys

# Generate the merged waive list:
# 1. Parse the current MR waive list, and get the removed lines from the diff
# 2. Parse the TOT waive list
# 3. Merge the current MR waive list and TOT waive list, and remove the removed lines from the step 1


def get_remove_lines_from_diff_file(diff_file):
    with open(diff_file, 'r') as f:
        diff = f.read()
    lines = diff.split('\n')
    remove_lines = [
        line[1:] + '\n' for line in lines
        if len(line) > 1 and line.startswith('-')
    ]
    return remove_lines


def parse_waive_txt(waive_txt):
    with open(waive_txt, 'r') as f:
        lines = f.readlines()
    waive_list = [line for line in lines if line.strip()]
    return waive_list


def write_waive_list(waive_list, output_file):
    with open(output_file, 'w') as f:
        for line in waive_list:
            f.write(line)


def merge_waive_list(cur_list, main_list, remove_lines, output_file):
    merged = list(dict.fromkeys(cur_list + main_list))
    for line in reversed(remove_lines):
        for i in range(len(merged) - 1, -1, -1):
            if merged[i] == line:
                merged.pop(i)
                break
    write_waive_list(merged, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cur-waive-list',
                        required=True,
                        help='Current waive list')
    parser.add_argument('--latest-waive-list',
                        required=True,
                        help='Latest waive list')
    parser.add_argument('--diff-file',
                        required=True,
                        help='File containing diff of the waive list')
    parser.add_argument('--output-file', required=True, help='Output file')
    args = parser.parse_args(sys.argv[1:])
    cur_list = parse_waive_txt(args.cur_waive_list)
    main_list = parse_waive_txt(args.latest_waive_list)
    remove_lines = get_remove_lines_from_diff_file(args.diff_file)
    merge_waive_list(cur_list, main_list, remove_lines, args.output_file)
