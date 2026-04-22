from tensorrt_llm.scaffolding.task import OpenAIToolDescription

conduct_research_tool = OpenAIToolDescription(
    name="conduct_research",
    description="Conduct research on a given topic",
    parameters={"research_topic": {"type": "string", "description": "The topic of the research"}},
)

complete_research_tool = OpenAIToolDescription(
    name="complete_research", description="Complete the research", parameters={}
)

think_tool = OpenAIToolDescription(
    name="think_tool",
    description="Think about the research",
    parameters={"think": {"type": "string", "description": "The reflection of the research"}},
)

tavily_search_tool = OpenAIToolDescription(
    name="tavily_search",
    description=(
        "Perform web searches via Tavily and return formatted results. "
        "Accepts one or more queries in a single call."
    ),
    parameters={
        "query": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "A search query.",
            },
            "minItems": 1,
            "description": "List of search queries (one or more strings).",
        }
    },
)

google_scholar_tool = OpenAIToolDescription(
    name="google_scholar",
    description=(
        "Search Google Scholar (and optional web via Custom Search) for academic "
        "papers and related sources. Accepts one or more queries in a single call."
    ),
    parameters={
        "query": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "A search query.",
            },
            "minItems": 1,
            "description": "List of search queries (one or more strings).",
        },
        "limit": {
            "type": "integer",
            "description": "Optional cap on results per query (server default if omitted).",
        },
    },
)

fetch_webpage_tool = OpenAIToolDescription(
    name="fetch_webpage",
    description=(
        "Fetch raw text from one or more web pages or PDF URLs via the configured "
        "reader backend (e.g. Jina / ScraperAPI)."
    ),
    parameters={
        "url": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "URLs to fetch.",
        },
        "parse_type": {
            "type": "string",
            "enum": ["html", "pdf"],
            "description": "Whether the URL is HTML or a PDF.",
        },
    },
)

python_interpreter_tool = OpenAIToolDescription(
    name="python_interpreter",
    description=(
        "Execute Python in the configured sandbox. Use print() for output you need "
        "to see. For calculations, data checks, or parsing scraped text."
    ),
    parameters={
        "code": {
            "type": "string",
            "description": "Python source to execute.",
        }
    },
)

reflection_tool = OpenAIToolDescription(
    name="reflection",
    description="For reflection and strategic planning during research",
    parameters={"reflection": {"type": "string"}},
)
