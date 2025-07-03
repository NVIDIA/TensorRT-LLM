from tensorrt_llm.llmapi import LlmArgs
from tensorrt_llm.llmapi.utils import DocTagger, generate_api_docs_as_docstring


def test_doc_tagger():
    doc_tagger = DocTagger()
    doc_tagger()
    print(generate_api_docs_as_docstring(LlmArgs))
