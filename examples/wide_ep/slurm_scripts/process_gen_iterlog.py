import argparse
import glob
import re

import pandas as pd

mtp_accept_rate = {1: 1.86, 2: 2.42, 3: 2.68}


def process_files(dir_prefix):
    summary_data = []
    pattern = f"{dir_prefix}*/concurrency_*/gen_only.txt"
    files = glob.glob(pattern)
    print(f"Found {len(files)} files matching pattern {pattern}")

    for file in files:
        data = []
        # Extract parameter information from file path
        # Match ctx(number)_gen(number)_(tep|dep)(number)_batch(number)_eplb(number)_mtp(number)
        match = re.search(
            r'ctx\d+_gen\d+_(tep|dep)(\d+)_batch(\d+)_eplb(\d+)(?:_mtp(\d+))?',
            file)
        if not match:
            # print(f"No match found for file {file}")
            continue

        # Extract concurrency number from path
        concurrency_match = re.search(r'concurrency_(\d+)', file)
        if not concurrency_match:
            print(f"No concurrency match found for file {file}")
            continue

        # Directly use the second format parsing logic
        attn_type = match.group(1)
        rank_num = int(match.group(2))
        int(match.group(3))
        eplb_num = int(match.group(4))
        mtp_num = int(match.group(5)) if match.group(5) else 0
        concurrency = int(concurrency_match.group(1))

        # Determine tp_rank and ep_rank based on folder name
        if attn_type == 'tep':
            ep_rank = rank_num
        else:  # dep
            ep_rank = rank_num
        name = f"{attn_type}_{rank_num}_eplb{eplb_num}_mtp{mtp_num}"

        # Read and parse log file
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Use regex to match specified format log lines
            log_pattern = r'iter = (\d+), global_rank = (\d+), rank = (\d+), currank_total_requests = (\d+)/(\d+), elapsed_time = ([\d.]+)s, timestamp = ([^,]+), num_scheduled_requests: (\d+), states = \{\'num_ctx_requests\': (\d+), \'num_ctx_tokens\': (\d+), \'num_generation_tokens\': (\d+)\}'

            matches = re.findall(log_pattern, content)

            if matches:
                # Process each matched log line
                for match in matches:
                    iter_num = int(match[0])
                    global_rank = int(match[1])
                    rank = int(match[2])
                    current_requests = int(match[3])
                    total_requests = int(match[4])
                    elapsed_time = float(match[5])
                    timestamp = match[6]
                    num_scheduled_requests = int(match[7])
                    num_ctx_requests = int(match[8])
                    num_ctx_tokens = int(match[9])
                    num_generation_tokens = int(match[10])

                    # Calculate throughput metrics
                    # Here you can calculate corresponding performance metrics as needed
                    throughput_per_user = num_generation_tokens / elapsed_time if elapsed_time > 0 else 0

                    data.append({
                        'concurrency': concurrency,
                        'iter': iter_num,
                        'global_rank': global_rank,
                        'rank': rank,
                        'current_requests': current_requests,
                        'total_requests': total_requests,
                        'elapsed_time': elapsed_time,
                        'timestamp': timestamp,
                        'num_scheduled_requests': num_scheduled_requests,
                        'num_ctx_requests': num_ctx_requests,
                        'num_ctx_tokens': num_ctx_tokens,
                        'num_generation_tokens': num_generation_tokens,
                        'throughput_per_user': throughput_per_user
                    })
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
        # if data is not empty, save to csv
        if data:
            df = pd.DataFrame(data)
            df = df.sort_values(['concurrency', 'iter'])
            # file name is the same as the file prefix + .csv
            output_file = file.split('.')[0] + '.csv'

            # Filter rows where num_ctx_tokens == 0
            df = df[df['num_ctx_tokens'] == 0]

            df = df.iloc[50:-10]
            if attn_type == 'tep':
                df = df[df['num_scheduled_requests'] == int(concurrency)]
                df = df[df['num_generation_tokens'] == int(concurrency *
                                                           (mtp_num + 1))]
            elif attn_type == 'dep':
                df = df[df['num_scheduled_requests'] == int(concurrency /
                                                            ep_rank)]
                df = df[df['num_generation_tokens'] == int(concurrency /
                                                           ep_rank *
                                                           (mtp_num + 1))]

            df.to_csv(output_file, index=False)
            print(f"Data saved to {output_file}")
            print(f"Total records processed: {len(data)}")

            # check df is empty
            if df.empty:
                print(f"No valid data found for {file}")
            else:
                # get elapsed_time avg time
                elapsed_time_avg = df['elapsed_time'].mean()
                throughput_per_user = 1 / elapsed_time_avg if elapsed_time_avg > 0 else 0
                throughput_per_user = throughput_per_user * mtp_accept_rate[
                    mtp_num] if mtp_num > 0 else throughput_per_user
                output_throughput = throughput_per_user * concurrency
                throughput_per_gpu = output_throughput / ep_rank
                summary_data.append({
                    'name': name,
                    'concurrency': concurrency,
                    'throughput_per_user': throughput_per_user,
                    'throughput_per_gpu': throughput_per_gpu,
                    'output_throughput': output_throughput,
                    'elapsed_time_avg': elapsed_time_avg,
                    'number_iters': len(df)
                })

    if summary_data:
        # Create DataFrame and sort
        df = pd.DataFrame(summary_data)
        df = df.sort_values(['name', 'concurrency'])

        # Save as CSV
        output_file = f"{dir_prefix}_iterlog.csv"
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        print(f"Total records processed: {len(data)}")
    else:
        print("No valid data found to save")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process benchmark files and aggregate data.')
    parser.add_argument('--dir_prefix',
                        help='Directory prefix to search for benchmark files')
    args = parser.parse_args()
    process_files(args.dir_prefix)
