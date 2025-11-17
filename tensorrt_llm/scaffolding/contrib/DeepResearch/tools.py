from tensorrt_llm.scaffolding.task import OpenAIToolDescription

# TODO: Definition of researcher tools subject to certain specifications.
# TODO: Add more tools (e.g., MCP tools) beyond search.
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
    description="For conducting web searches to gather information",
    parameters={"query": {"type": "string"}},
)

reflection_tool = OpenAIToolDescription(
    name="reflection",
    description="For reflection and strategic planning during research",
    parameters={"reflection": {"type": "string"}},
)
