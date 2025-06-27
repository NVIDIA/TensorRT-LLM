import argparse
import sys


def parse_diff(diff):
    remove_tests = set()
    lines = diff.split('\n')
    for line in lines:
        if len(line) == 0:
            continue
        if line[0] == '-' and "SKIP" in line:
            test = line[1:line.find('SKIP')].strip()
            remove_tests.add(test)
    return list(remove_tests)


def parse_waive_txt(waive_txt):
    with open(waive_txt, 'r') as f:
        lines = f.readlines()
    waive_list = []
    for line in lines:
        if len(line) == 0:
            continue
        waive_list.append(line)
    return waive_list


def merge_waive_list(cur_list, main_list, remove_tests):
    merged = list(dict.fromkeys(cur_list + main_list))
    for test in remove_tests:
        merged = [
            item for item in merged if not (test in item and 'SKIP' in item)
        ]
    print(merged)
    return merged


def write_waive_list(waive_list, output_file):
    with open(output_file, 'w') as f:
        for line in waive_list:
            f.write(line)
        f.write('\n')


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
    remove_lines = parse_diff(args.diff)
    merged = merge_waive_list(cur_list, main_list, remove_lines)
    write_waive_list(merged, args.output_file)
