#!/usr/bin/env python3
import csv
import json
import os
import re
from typing import NamedTuple

import click


def process_file(filename):
    # Extract isl, osl, and streaming from filename
    # format is like report-2000.64.64.json
    basename = os.path.basename(filename)
    reg = re.compile(r'report-(\d+)\.(\d+)\.(\d+)(-streaming)?\.json')
    matched = reg.match(basename)
    assert matched, f"Filename {filename} does not match expected format"
    isl, osl = map(int, matched.groups()[1:3])
    streaming = 'yes' if '-streaming' in filename else 'no'

    # Load JSON content from file
    with open(filename, 'r') as f:
        data = json.load(f)

    # Extract relevant fields from JSON content
    llmapi_throughput = data['llmapi']['token_throughput']
    cpp_throughput = data['cpp']['token_throughput']

    return isl, osl, llmapi_throughput, cpp_throughput, streaming


class Record(NamedTuple):
    isl: int
    osl: int
    streaming: str
    llmapi_throughput: float
    cpp_throughput: float
    ratio: str


@click.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--output-file', type=click.Path(), default='output.csv')
def main(directory, output_file):
    # Create CSV writer
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header row
        writer.writerow(['isl', 'osl', 'streaming', 'llmapi', 'cpp', 'ratio'])

        records = []

        # Process each file in directory
        for filename in os.listdir(directory):
            if filename.startswith('report') and filename.endswith('.json'):
                isl, osl, llmapi_throughput, cpp_throughput, streaming = process_file(
                    os.path.join(directory, filename))
                records.append(
                    Record(
                        isl,
                        osl,
                        streaming,
                        llmapi_throughput,
                        cpp_throughput,
                        ratio=
                        f"{float(llmapi_throughput) / float(cpp_throughput) * 100.:.2f}%"
                    ))

        # sort by isl, osl and streaming
        records = sorted(records, key=lambda x: (x.isl, x.osl, x.streaming))
        for record in records:
            writer.writerow(record)


if __name__ == '__main__':
    main()
