import argparse

from openai import OpenAI


def run_completion(args: argparse.Namespace):
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="tensorrt_llm",
    )
    prompt = args.prompt if args.prompt else "Where is New York?"
    completion = client.completions.create(
        model="llama-v3-8b-instruct-hf",
        prompt=[prompt],
        top_p=args.top_p,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        echo=args.echo,
        n=args.n,
        stream=args.stream,
        extra_body={
            "top_k": args.top_k,
            "use_beam_search": args.use_beam_search,
        },
    )
    if args.stream:
        for chunk in completion:
            print(chunk)
    else:
        for choice in completion.choices:
            print(choice.text)


def run_chat(args: argparse.Namespace):
    # TODO{pengyunl}: multi-run chat example for openai api
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="tensorrt_llm",
    )
    prompt = args.prompt if args.prompt else "Where is New York?"
    completion = client.chat.completions.create(
        model="llama-v3-8b-instruct-hf",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        top_p=args.top_p,
        temperature=args.temperature,
        stream=args.stream,
        n=args.n,
        extra_body={
            "top_k": args.top_k,
            "use_beam_search": args.use_beam_search,
            "echo": args.echo,
        },
    )
    if args.stream:
        for chunk in completion:
            print(chunk)
    else:
        for choice in completion.choices:
            print(choice.message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--echo", action="store_true", default=False)
    parser.add_argument("--stream", action="store_true", default=False)
    parser.add_argument("--use_beam_search", action="store_true", default=False)
    parser.add_argument("--api",
                        type=str,
                        choices=["chat", "completions"],
                        default="chat")
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()
    if args.api == "chat":
        run_chat(args)
    else:
        run_completion(args)
