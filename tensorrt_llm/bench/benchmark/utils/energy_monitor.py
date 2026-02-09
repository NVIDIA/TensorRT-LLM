try:
    from pynvml import (
        NVMLError,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetTotalEnergyConsumption,
        nvmlInit,
        nvmlShutdown,
    )

    has_nvml = True
except ImportError:
    has_nvml = False


class EnergyMonitor:
    def __init__(self, world_size):
        global has_nvml
        self._start_energies = None
        if has_nvml:
            try:
                nvmlInit()
                self._device_count = min(nvmlDeviceGetCount(), world_size)
                self._handles = [nvmlDeviceGetHandleByIndex(i) for i in range(self._device_count)]
            except NVMLError:
                has_nvml = False

    def start(self):
        if has_nvml:
            self._start_energies = [
                nvmlDeviceGetTotalEnergyConsumption(handle) for handle in self._handles
            ]

    def stop(self, statistics, world_size):
        if not has_nvml:
            return statistics.set_energy(None)

        total_energy = 0.0
        for handle, start_energy in zip(self._handles, self._start_energies):
            energy = (nvmlDeviceGetTotalEnergyConsumption(handle) - start_energy) / 1000.0
            total_energy += energy

        total_energy *= world_size / self._device_count
        nvmlShutdown()

        statistics.set_energy(total_energy)
