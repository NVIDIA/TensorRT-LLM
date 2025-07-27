from tensorrt_llm.llmapi.utils import ApiStatusRegistry, BackendType


def test_api_status_registry():

    @ApiStatusRegistry.set_api_status("beta")
    def _my_method(self, *args, **kwargs):
        pass

    assert ApiStatusRegistry.get_api_status(_my_method) == "beta"

    @ApiStatusRegistry.set_api_status("prototype")
    def _my_method(self, *args, **kwargs):
        pass

    assert ApiStatusRegistry.get_api_status(_my_method) == "prototype"

    class App:

        @ApiStatusRegistry.set_api_status("beta")
        def _my_method(self, *args, **kwargs):
            pass

    assert ApiStatusRegistry.get_api_status(App._my_method) == "beta"


class TestBackendType:

    def test_display_name(self):
        assert str(BackendType.PYTORCH) == "PyTorch"
        assert str(BackendType.TENSORRT) == "TensorRT"
        assert str(BackendType._AUTODEPLOY) == "AutoDeploy"

    def test_value(self):
        assert BackendType.PYTORCH.canonical_value == "pytorch"
        assert BackendType.TENSORRT.canonical_value == "tensorrt"
        assert BackendType._AUTODEPLOY.canonical_value == "_autodeploy"

    def test_canonical_values(self):
        assert "pytorch" in BackendType.canonical_values()
        assert "tensorrt" in BackendType.canonical_values()
        assert "_autodeploy" in BackendType.canonical_values()
        # for other values ...
