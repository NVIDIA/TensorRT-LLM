from tensorrt_llm.llmapi import LlmArgs
from tensorrt_llm.llmapi.utils import (ApiStatusRegistry,
                                       generate_api_docs_as_docstring)


def test_api_status_registry():

    @ApiStatusRegistry.set_api_status("beta")
    def _my_method(self, *args, **kwargs):
        pass

    assert ApiStatusRegistry.get_api_status(_my_method) == "beta"

    @ApiStatusRegistry.set_api_status("prototype")
    def _my_method(self, *args, **kwargs):
        pass

    # will always keep the first status, and the behaviour will be unknown if
    # one method is registered with a different status in different files.
    assert ApiStatusRegistry.get_api_status(_my_method) == "beta"

    class App:

        @ApiStatusRegistry.set_api_status("beta")
        def _my_method(self, *args, **kwargs):
            pass

    assert ApiStatusRegistry.get_api_status(App._my_method) == "beta"


def test_generate_api_docs_as_docstring():
    doc = generate_api_docs_as_docstring(LlmArgs)
    assert ":tag:`beta`" in doc, "the label is not generated"
    print(doc)
