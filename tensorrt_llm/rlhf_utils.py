from typing import List

import torch

from tensorrt_llm._ray_utils import control_action_decorator
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm._torch.virtual_memory import (materialize_with_tag,
                                                release_with_tag,
                                                verify_sleep_wakeup_tags)
from tensorrt_llm.logger import logger


class WorkerExtension:

    @control_action_decorator
    def sleep(self, sleep_tags: List[str]):
        try:
            tags = verify_sleep_wakeup_tags(sleep_tags)
            logger.info(f"PyExecutor sleep: {tags}")
            torch.cuda.synchronize()
            release_with_tag(*tags)
            torch.cuda.synchronize()
        except Exception as e:
            logger.error(f"Encountered an error in sleep: {e}")
            raise e

    @control_action_decorator
    def wakeup(self, wakeup_tags: List[str]):
        try:
            tags = verify_sleep_wakeup_tags(wakeup_tags)
            logger.info(f"PyExecutor wakeup: {tags}")
            torch.cuda.synchronize()
            materialize_with_tag(*tags)
            torch.cuda.synchronize()
        except Exception as e:
            logger.error(f"Encountered an error in wakeup")
            raise e

    @control_action_decorator
    def update_weights(self, ipc_handles: dict):
        try:
            logger.info(f"PyExecutor update_weight_from_ipc_handles")
            ipc_handles = ipc_handles
            device_uuid = get_device_uuid(self.device_id)

            if device_uuid not in ipc_handles:
                raise ValueError(
                    f"Device UUID {device_uuid} not found in ipc_handles")

            weights = {}
            all_handles = ipc_handles[device_uuid]

            for param_name, tensor_handle in all_handles:
                func, args = tensor_handle
                list_args = list(args)
                list_args[6] = self.device_id  # Set target device
                tensor = func(*list_args)
                weights[param_name] = tensor

            self.engine.model_engine.model.load_weights(weights)
            torch.cuda.synchronize()
            self.engine.reset_prefix_cache()

        except Exception as e:
            logger.error(f"Encountered an error in update_weight")
            raise e
