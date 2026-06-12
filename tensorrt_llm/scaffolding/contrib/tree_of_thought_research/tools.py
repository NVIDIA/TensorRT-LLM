from tensorrt_llm.scaffolding.task import OpenAIToolDescription

TOT_RESEARCH_TOOLS = [
    OpenAIToolDescription(
        name="tavily_search",
        description=(
            "Search the web via Tavily and return concise result snippets. "
            "Use this when the branch needs fresh or source-backed facts."
        ),
        parameters={
            "query": {
                "type": "array",
                "items": {"type": "string", "description": "A search query."},
                "minItems": 1,
                "description": "One or more web search queries.",
            }
        },
    ),
    OpenAIToolDescription(
        name="fetch_webpage",
        description=(
            "Fetch and summarize known URLs. Use it after search results identify "
            "pages that should be read directly."
        ),
        parameters={
            "url": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "URLs to fetch.",
            },
            "goal": {
                "type": "string",
                "description": "What information to extract from the page.",
            },
            "parse_type": {
                "type": "string",
                "enum": ["html", "pdf"],
                "description": "Whether the URL is an HTML page or a PDF.",
            },
        },
    ),
    OpenAIToolDescription(
        name="python_interpreter",
        description=(
            "Run Python in the configured sandbox for calculations, parsing, "
            "or consistency checks. Use print() for required output. The Python "
            "environment only has basic packages by default; if extra packages "
            "are needed before import, first call python_interpreter with code "
            "such as: import subprocess, sys; subprocess.check_call([sys.executable, "
            "'-m', 'pip', 'install', '<package>'])."
        ),
        parameters={
            "code": {
                "type": "string",
                "description": "Python source code to execute.",
            },
        },
    ),
    OpenAIToolDescription(
        name="reflection",
        description=(
            "Record a planning or reasoning step when no external lookup or "
            "computation is needed for this branch."
        ),
        parameters={
            "reflection": {
                "type": "string",
                "description": "The branch-local reasoning or plan.",
            }
        },
    ),
    OpenAIToolDescription(
        name="complete_task",
        description=(
            "Use this only when the branch has enough evidence or derivation to "
            "finish the task. Provide the final answer and concise justification."
        ),
        parameters={
            "answer": {
                "type": "string",
                "description": "The final answer for this branch.",
            },
            "justification": {
                "type": "string",
                "description": "Brief support for why this answer is final.",
            },
        },
    ),
]
