"""
AdapterSlotManager for managing slots that stores LoRA indices.
"""

from collections import OrderedDict
from typing import List, Optional

from ...pyexecutor.resource_manager import PeftCacheManager
from ...pyexecutor.scheduler import RequestList


class AdapterSlotManager:
    """
    Manages max_num_adapters ordered slots for distinct task_ids to enable CUDA Graph compatibility.

    Each slot can hold one adapter (task_id) and maintains a consistent ordering that allows
    the CUDA Graph to be captured with fixed buffer layouts.
    """

    def __init__(self, max_num_adapters: int):
        """
        Initialize the AdapterSlotManager.

        Args:
            max_num_adapters: Maximum number of LoRA adapters that can be active simultaneously
        """
        self.max_num_adapters = max_num_adapters

        # Slot management
        self.slot2task: List[Optional[int]] = [None] * max_num_adapters
        self.task2slot: OrderedDict[int, int] = OrderedDict()  # represent LRU order

        # State tracking
        self.slots_changed = False

    def find_free_slot(self) -> int:
        """
        Find a free slot. Return slot_id if found, otherwise return None.
        """
        return self.slot2task.index(None)

    def remove_task(self, task_id: int) -> Optional[int]:
        """
        Remove a task_id from slots. Return its slot_id if present otherwise return None.
        """
        slot_id = self.task2slot.pop(task_id, None)
        if slot_id is not None:
            self.slots_changed = True
            self.slot2task[slot_id] = None
        return slot_id

    def get_or_assign_task(self, task_id: int) -> tuple[int, Optional[int]]:
        """
        Assign a task_id to a slot and do LRU eviction if necessary.
        If already in any slot, update LRU order.
        Return: pair (assigned slot_id, evicted task_id)
        """
        evicted_task = None
        if task_id in self.task2slot:
            self.task2slot.move_to_end(task_id)
        else:
            self.slots_changed = True
            if len(self.task2slot) < self.max_num_adapters:
                free_slot = self.find_free_slot()
                self.slot2task[free_slot] = task_id
                self.task2slot[task_id] = free_slot
            else:
                # evict lru
                evicted_task, evicted_slot = self.task2slot.popitem(last=False)
                self.slot2task[evicted_slot] = task_id
                self.task2slot[task_id] = evicted_slot
        return self.task2slot[task_id], evicted_task

    def remove_evicted_slots_in_cpp(self, peft_cache_manager: PeftCacheManager):
        """
        Validate slots by removing tasks that are not cached in PeftCacheManager.
        """
        for task_id in self.slot2task:
            if task_id is not None:
                if not peft_cache_manager.is_task_cached_device(task_id):
                    self.remove_task(task_id)

    def update_slots(
        self, requests: RequestList, peft_cache_manager: PeftCacheManager
    ) -> list[int]:
        """
        Get slot mapping for all requests in a scheduled batch.

        Args:
            scheduled_requests: The scheduled requests for the current batch

        Returns:
            Dict mapping request_id to slot_id, with slot_id=max_num_adapters for base model requests
        """
        # remove task evicted in PeftCacheManager in C++
        self.remove_evicted_slots_in_cpp(peft_cache_manager)

        # check if total number of unique tasks in the requests is not larger than max_num_adapters
        tasks = [request.lora_task_id for request in requests]
        unique_tasks = {t for t in tasks if t is not None}
        assert len(unique_tasks) <= self.max_num_adapters, (
            f"Batch with more unique LoRA adapters ({len(unique_tasks)}) than max_num_adapters={self.max_num_adapters} "
            "is not supported"
        )

        # assign slots to tasks
        for i, task in enumerate(tasks):
            if task is None:
                tasks[i] = self.max_num_adapters
            else:
                tasks[i], evicted_task = self.get_or_assign_task(task)

        return tasks

    def get_slot_to_task_mapping(self) -> tuple[Optional[int], ...]:
        """
        Get current slot to task mapping.

        Returns:
            Tuple mapping slot_id to task_id (or None if slot is empty)
        """
        return tuple(self.slot2task)

    def has_slots_changed(self) -> bool:
        """Check if slot assignments have changed since last check."""
        return self.slots_changed

    def reset_slots_changed(self):
        """Reset the slots_changed flag."""
        self.slots_changed = False
