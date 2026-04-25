import os

from tensorrt_llm.executor import utils


def test_spawn_proxy_hmac_key_env_is_consumed(monkeypatch):
    key = "01" * 32
    env_var = utils.LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value
    utils._SPAWN_PROXY_PROCESS_IPC_HMAC_KEY = None
    monkeypatch.setenv(env_var, key)

    try:
        assert utils.get_spawn_proxy_process_ipc_hmac_key_env() == bytes.fromhex(key)
        assert env_var not in os.environ
        assert utils.get_spawn_proxy_process_ipc_hmac_key_env() == bytes.fromhex(key)
    finally:
        utils._SPAWN_PROXY_PROCESS_IPC_HMAC_KEY = None
