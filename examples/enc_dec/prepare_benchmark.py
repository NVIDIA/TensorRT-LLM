import argparse
import json
import numpy as np
import random
from pathlib import Path

def generate_samples(
        num_samples, 
        input_mean, input_std, input_min, input_max,
        context_mean, context_std, context_min, context_max,
        output_mean, output_std, output_min, output_max,
        output_file,
        max_input_id,
        num_vocabs=8):
    
    # Create metadata
    metadata = {
        "workload_type": "token-norm-dist",
        "input_mean": input_mean,
        "input_stdev": input_std,
        "output_mean": output_mean,
        "output_stdev": output_std,
        "num_requests": num_samples,
        "tokenize_vocabsize": 2048 * num_vocabs,  # Now using num_vocabs parameter
        "max_input_len": input_max,
        "max_output_len": output_max,
        "workload_name": f"workload_type:token-norm-dist__input_mean:{input_mean}__input_stdev:{input_std}__output_mean:{output_mean}__output_stdev:{output_std}__num_requests:{num_samples}__tokenize_vocabsize:{2048 * num_vocabs}__max_input_len:{input_max}__max_output_len:{output_max}"
    }
    
    samples = []
    
    for i in range(num_samples):
        # Generate random lengths capped at max values
        input_len = min(max(input_min, int(np.random.normal(input_mean, input_std))), input_max)
        context_len = min(max(context_min, int(np.random.normal(context_mean, context_std))), context_max)
        output_len = min(max(output_min, int(np.random.normal(output_mean, output_std))), output_max)
        
        # Generate input_ids: random ints in range (0, 2048)
        input_ids = [random.randint(0, max_input_id - 1) for _ in range(input_len)]
        
        # Generate context_ids as specified
        context_matrix = np.random.randint(0, 2048, size=(context_len, num_vocabs))
        # Set first row to zeros
        context_matrix[0, :] = 0
        
        # Shift each column by i * 2048
        for i in range(num_vocabs):
            context_matrix[:, i] += i * 2048
        
        # Flatten to 1D array
        context_ids = context_matrix.flatten().tolist()
        
        # Create sample
        sample = {
            "input_len": input_len,
            "input_ids": input_ids,
            "context_ids": context_ids,
            # no need to multiply by num_vocabs,
            # this defines number of decoder iterations
            "output_len": output_len,
            "task_id": -1  # As in your example
        }
        
        samples.append(sample)
    
    # Create the full JSON structure
    json_data = {
        "metadata": metadata,
        "samples": samples
    }
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Generated {num_samples} samples and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate sample JSON data with configurable parameters')
    
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='samples.json', help='Output JSON file')
    parser.add_argument('--num_vocabs', type=int, default=8, help='Number of vocabularies')
    
    parser.add_argument('--input_len', type=int, nargs=4, metavar=('MEAN', 'STD', 'MIN', 'MAX'),
                        default=[128, 0, 128, 128], help='Input length parameters: mean, std, max')
    parser.add_argument('--context_len', type=int, nargs=4, metavar=('MEAN', 'STD', 'MIN', 'MAX'),
                        default=[3 * 75, 0, 3 * 75, 3 * 75], help='Context length parameters: mean, std, max')
    parser.add_argument('--output_len', type=int, nargs=4, metavar=('MEAN', 'STD', 'MIN', 'MAX'),
                        default=[5 * 75, 0, 5 * 75,  5 * 75], help='Output length parameters: mean, std, max')
    parser.add_argument('--max_input_id', type=int, default=2048, help='Max input id')
    
    args = parser.parse_args()
    
    generate_samples(
        args.samples,
        args.input_len[0], args.input_len[1], args.input_len[2], args.input_len[3],
        args.context_len[0], args.context_len[1], args.context_len[2], args.context_len[3],
        args.output_len[0], args.output_len[1], args.output_len[2], args.output_len[3],
        args.output,
        args.max_input_id,
        args.num_vocabs
    )

if __name__ == "__main__":
    main()
