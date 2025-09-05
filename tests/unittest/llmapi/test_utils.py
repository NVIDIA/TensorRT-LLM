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


class DelayedAssert:

    def __init__(self, store_stack: bool = False):
        self.assertions = []
        self.store_stack = store_stack

    def add(self, result: bool, msg: str):
        import traceback
        self.assertions.append(
            (bool(result), str(msg), traceback.format_stack()))

    def get_msg(self):
        ret = ['Some assertions failed:']
        for result, msg, stack in self.assertions:
            ret.append('\n'.join([
                f'Assert result: {result}', msg,
                ''.join(stack) if self.store_stack else ''
            ]))
        ret = '\n-----------------------------------------\n'.join(ret)
        ret = 'Some assertions failed:\n' + ret
        return ret

    def clear(self):
        self.assertions.clear()

    def assert_all(self):
        assert all(ret[0] for ret in self.assertions), self.get_msg()
        self.clear()
