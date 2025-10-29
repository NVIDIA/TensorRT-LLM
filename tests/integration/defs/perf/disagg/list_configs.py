"""
List and inspect test configurations
"""

import argparse
from config_loader import ConfigLoader
from common import EnvManager


def main():
    parser = argparse.ArgumentParser(description="List test configurations")
    parser.add_argument("--base-dir", default="test_configs", help="Base config directory")
    parser.add_argument("--test-type", help="Filter by test type (disagg, widep, etc.)")
    parser.add_argument("--category", help="Filter by category (perf, accuracy)")
    parser.add_argument("--model", help="Filter by model name")
    parser.add_argument("--gpu-type", help="Filter by GPU type (GB200, H100, etc.). Default: from GPU_TYPE env var")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed info")
    parser.add_argument("--show-metrics", action="store_true", help="Show metrics config")
    parser.add_argument("--show-all-gpus", action="store_true", help="Show all configs regardless of GPU support")
    
    args = parser.parse_args()
    
    loader = ConfigLoader(base_dir=args.base_dir)
    
    # If --show-all-gpus is specified, pass empty string to disable GPU filtering
    gpu_filter = "" if args.show_all_gpus else args.gpu_type
    
    configs = loader.scan_configs(
        test_type=args.test_type,
        test_category=args.category,
        model_name=args.model,
        gpu_type=gpu_filter
    )
    
    print(f"\nFound {len(configs)} test configurations\n")
    print("=" * 80)
    
    # Group by test_type and category
    grouped = {}
    for config in configs:
        key = (config.test_type, config.test_category)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(config)
    
    for (test_type, category), group_configs in sorted(grouped.items()):
        print(f"\n{test_type} / {category}")
        print("-" * 40)
        print(f"  Total: {len(group_configs)} configurations")
        
        # Group by model
        by_model = {}
        for config in group_configs:
            if config.model_name not in by_model:
                by_model[config.model_name] = []
            by_model[config.model_name].append(config)
        
        for model, model_configs in sorted(by_model.items()):
            print(f"\n  {model}: {len(model_configs)} configs")
            for config in model_configs:
                filename = config.config_path.split('/')[-1]
                if '\\' in config.config_path:
                    filename = config.config_path.split('\\')[-1]
                print(f"    - {filename}")
                
                if args.verbose:
                    gen_config = config.config_data['worker_config']['gen']
                    print(f"      TP: {gen_config['tensor_parallel_size']}, "
                          f"Batch: {gen_config['max_batch_size']}, "
                          f"DP: {gen_config['enable_attention_dp']}")
                
                if args.show_metrics:
                    metrics = config.metrics_config
                    print(f"      Metrics log: {metrics.log_file}")
                    print(f"      Metric names: {', '.join(metrics.metric_names)}")
                
                if args.verbose or args.show_all_gpus:
                    print(f"      Supported GPUs: {', '.join(config.supported_gpus)}")
    
    print("\n" + "=" * 80)
    print(f"\nTotal: {len(configs)} configurations")
    
    # Show GPU type information
    if not args.show_all_gpus:
        current_gpu = args.gpu_type or EnvManager.get_gpu_type()
        print(f"Filtered for GPU type: {current_gpu}")
    
    # Show summary
    print("\nSummary:")
    print(f"  Models: {len(loader.get_all_models())}")
    print(f"  Test types: {', '.join(loader.get_all_test_types())}")


if __name__ == "__main__":
    main()

