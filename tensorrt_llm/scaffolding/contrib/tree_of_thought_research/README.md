# Tree-of-Thought Research

This directory contains a research-oriented Tree-of-Thought controller:
`TOTResearchController`.

The controller expands several candidate thoughts per depth, executes exactly one
native tool call for each branch, scores the observed branches with the generation
model, keeps the top branches, and writes the final answer from the selected
trajectory.

## Tools

The tool schemas are defined locally in `tot_research_controllers.py`:

- `tavily_search` for web search.
- `fetch_webpage` for reading known URLs through `VisitController`.
- `python_interpreter` for computation or parsing.
- `reflection` for planning-only branches.

## Examples

Run a single prompt:

```bash
python examples/scaffolding/contrib/tree_of_thought_research/run_tot_research.py \
  --config examples/scaffolding/contrib/open_deep_research/config.example.yaml \
  --prompt "Use current sources to compare TensorRT-LLM and vLLM for serving LLMs."
```
