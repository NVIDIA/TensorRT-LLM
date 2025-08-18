import ray


class ColocateWorkerExtension:
    """
    The class for TRTLLM's RayGPUWorker to inherit from.
    """

    def additional_method(self):
        print(f"Additional method called in {int(ray.get_gpu_ids()[0])}")
