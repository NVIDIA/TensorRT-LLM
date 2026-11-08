from dynasor.core.evaluator import math_equal


def quick_parse(text: str) -> str:
    """Parse LaTeX text content."""
    if "\\text{" in text and "}" in text:
        # Find all occurrences of \text{...} and remove them
        while "\\text{" in text:
            start = text.find("\\text{")
            if start == -1:
                break
            end = text.find("}", start)
            if end == -1:
                break
            # Replace \text{content} with just content
            content = text[start + 6 : end]  # 6 is length of '\text{'
            text = text[:start] + content + text[end + 1 :]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if answer equals ground truth."""
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


def prepare_prompt(question: str, tokenizer, model_type: str = "deepseek") -> str:
    """Prepare prompt for a single question."""
    if model_type == "deepseek":
        # Format prompt using chat template for DeepSeek
        messages = [
            {
                "role": "system",
                "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n",
            },
            {"role": "user", "content": question},
        ]
    else:
        # Format for GPT-like models
        messages = [{"role": "user", "content": question}]

    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return full_prompt
