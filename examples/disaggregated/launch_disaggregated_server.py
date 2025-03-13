import argparse
import asyncio
import logging
from typing import List

from tensorrt_llm.llmapi.disagg_utils import (CtxGenServerConfig,
                                              parse_disagg_config_file)
from tensorrt_llm.serve import OpenAIDisaggServer

logging.basicConfig(level=logging.INFO)


def get_server_urls(server_configs: List[CtxGenServerConfig]) -> List[str]:
    ctx_server_urls = []
    gen_server_urls = []
    for cfg in server_configs:
        if cfg.type == "ctx":
            ctx_server_urls.append(f"http://{cfg.hostname}:{cfg.port}")
        else:
            gen_server_urls.append(f"http://{cfg.hostname}:{cfg.port}")

    return ctx_server_urls, gen_server_urls


def main():
    parser = argparse.ArgumentParser(description="Run disaggregated server")
    parser.add_argument("-c",
                        "--disagg_config_file",
                        type=str,
                        required=True,
                        help="Path to the YAML configuration file")
    parser.add_argument("-t",
                        "--server_start_timeout",
                        type=int,
                        default=180,
                        help="Server start timeout")
    parser.add_argument("-r",
                        "--request_timeout",
                        type=int,
                        default=180,
                        help="Request timeout")

    args = parser.parse_args()

    disagg_cfg = parse_disagg_config_file(args.disagg_config_file)

    ctx_server_urls, gen_server_urls = get_server_urls(
        disagg_cfg.server_configs)

    server = OpenAIDisaggServer(
        ctx_servers=ctx_server_urls,
        gen_servers=gen_server_urls,
        req_timeout_secs=args.request_timeout,
        server_start_timeout_secs=args.server_start_timeout)

    asyncio.run(server(disagg_cfg.hostname, disagg_cfg.port))


if __name__ == "__main__":
    main()
