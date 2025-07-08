import argparse
import sys


def get_remove_lines_from_diff(diff):
    lines = diff.split('\n')
    remove_lines = [
        line[1:] + '\n' for line in lines if len(line) > 1 and line[0] == '-'
    ]
    return remove_lines


def parse_waive_txt(waive_txt):
    with open(waive_txt, 'r') as f:
        lines = f.readlines()
    waive_list = [line for line in lines if len(line) != 0]
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
    parser.add_argument('--diff', required=True, help='Diff of the waive list')
    parser.add_argument('--output-file', required=True, help='Output file')
    args = parser.parse_args(sys.argv[1:])
    cur_list = parse_waive_txt(args.cur_waive_list)
    main_list = parse_waive_txt(args.latest_waive_list)
    remove_lines = get_remove_lines_from_diff(args.diff)
    merge_waive_list(cur_list, main_list, remove_lines, args.output_file)
