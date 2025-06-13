import argparse
import sys


def parse_diff(diff):
    remove_lines = []
    lines = diff.split('\n')
    for line in lines:
        if len(line) == 0:
            continue
        if line[0] == '-':
            remove_lines.append(line[1:])
    return remove_lines


def parse_waive_txt(waive_txt):
    with open(waive_txt, 'r') as f:
        lines = f.readlines()
    waive_list = []
    for line in lines:
        if len(line) == 0:
            continue
        waive_list.append(line)
    return waive_list


def merge_waive_list(cur_list, main_list, remove_lines):
    merged = list(dict.fromkeys(cur_list + main_list))
    for line in remove_lines:
        line += '\n'
        if line in merged:
            merged.remove(line)
    print(merged)
    return merged

def write_waive_list(waive_list, output_file):
    with open(output_file, 'w') as f:
        for line in waive_list:
            f.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cur-waive-list', required=True, help='Current waive list')
    parser.add_argument('--latest-waive-list', required=True, help='Latest waive list')
    parser.add_argument('--diff', required=True, help='Diff of the waive list')
    parser.add_argument('--output-file', required=True, help='Output file')

    cur_list = parse_waive_txt(args.cur_waive_list)
    main_list = parse_waive_txt(args.latest_waive_list)
    remove_lines = parse_diff(args.diff)
    merged = merge_waive_list(cur_list, main_list, remove_lines)
    write_waive_list(merged, args.output_file)
