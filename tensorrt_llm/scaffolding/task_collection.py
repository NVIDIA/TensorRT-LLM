from enum import Enum
from typing import List, Type

from .controller import ParallelProcess
from .task import ChatTask, DropKVCacheTask, GenerationTask, Task


class TaskCollection:

    def __init__(self):
        # reserved for future use
        pass

    def before_yield(self, tasks: List[Task]):
        pass

    def after_yield(self, tasks: List[Task]):
        pass


def with_task_collection(name: str, task_collection_cls: Type[TaskCollection],
                         **task_collection_kwargs):

    def decorator(controller_cls: Type["Controller"]):
        original_init = controller_cls.__init__
        original_process = controller_cls.process

        # add task collection to controller
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.task_collections[name] = task_collection_cls(
                **task_collection_kwargs)

        def new_process(self, tasks: List[Task], **kwargs):

            class TaskCollectionWrapper:

                def __init__(self, task_collection, gen):
                    self.task_collection = task_collection
                    self.gen = gen

                def __call__(self):
                    for obj in self.gen:
                        if isinstance(obj, ParallelProcess):
                            new_sub_gens = []
                            for sub_gen in obj.sub_gens:
                                new_sub_gen = TaskCollectionWrapper(
                                    self.task_collection, sub_gen)
                                new_sub_gens.append(new_sub_gen)
                            obj.sub_gens = new_sub_gens

                            yield obj
                        else:  # obj is a list of tasks
                            self.task_collection.before_yield(obj)
                            yield obj
                            self.task_collection.after_yield(obj)

                def __iter__(self):
                    return self.__call__()

            original_gen = original_process(self, tasks, **kwargs)
            new_gen = TaskCollectionWrapper(self.task_collections[name],
                                            original_gen)
            return new_gen()

        controller_cls.__init__ = new_init
        controller_cls.process = new_process

        return controller_cls

    return decorator


class GenerationTokenCounter(TaskCollection):

    def __init__(self):
        super().__init__()
        self.generation_token_count = 0
        self.pre_worker_token_sum = 0

    def before_yield(self, tasks: List[Task]):
        self.pre_worker_token_sum = 0
        for task in tasks:
            if isinstance(task, GenerationTask) or issubclass(
                    type(task), GenerationTask):
                if task.output_tokens:
                    self.pre_worker_token_sum += len(task.output_tokens)

    def after_yield(self, tasks: List[Task]):
        post_worker_token_sum = 0
        for task in tasks:
            # only support GenerationTask for now
            if isinstance(task, GenerationTask) or issubclass(
                    type(task), GenerationTask):
                if task.output_tokens:
                    post_worker_token_sum += len(task.output_tokens)
        self.generation_token_count += post_worker_token_sum - self.pre_worker_token_sum


class SubRequestMarker(TaskCollection):

    class UniqueIdGenerator:

        def __init__(self):
            self.unique_id = -1

        def generate(self):
            self.unique_id += 1
            return self.unique_id

    unique_id_generator = UniqueIdGenerator()
    top_node_count = 0

    def __init__(self, node_name: str, is_top_level: bool = False):
        super().__init__()
        self.node_name = node_name
        if is_top_level:
            self.unique_id = SubRequestMarker.top_node_count
            SubRequestMarker.top_node_count += 1
        else:
            self.unique_id = SubRequestMarker.unique_id_generator.generate()
        self.sub_node_id_tansfer_map = {}
        self.sub_node_counter = {}

    def before_yield(self, tasks: List[Task]):
        for task in tasks:
            if not isinstance(task, ChatTask):
                continue

            if task.unique_id is None:
                # new request from this chat task
                task.unique_id = SubRequestMarker.unique_id_generator.generate()
                task.sub_request_markers = [("LeafNode", task.unique_id)]

            # transfer the global unique id to the unique id from this node
            sub_node_name, raw_sub_node_id = task.sub_request_markers[-1]
            if sub_node_name not in self.sub_node_id_tansfer_map:
                sub_node_id = 0
                self.sub_node_id_tansfer_map[sub_node_name] = {
                    raw_sub_node_id: sub_node_id
                }
                self.sub_node_counter[sub_node_name] = sub_node_id + 1
            else:
                sub_node_id_map = self.sub_node_id_tansfer_map[sub_node_name]
                if raw_sub_node_id in sub_node_id_map:
                    sub_node_id = sub_node_id_map[raw_sub_node_id]
                else:
                    sub_node_id = self.sub_node_counter[sub_node_name]
                    sub_node_id_map[raw_sub_node_id] = sub_node_id
                    self.sub_node_counter[sub_node_name] += 1
            task.sub_request_markers[-1] = (sub_node_name, sub_node_id)

            task.sub_request_markers.append((self.node_name, self.unique_id))

    def after_yield(self, tasks: List[Task]):
        for task in tasks:
            if not isinstance(task, ChatTask):
                continue
            task.unique_id = None


def sub_request_node(node_name: str, is_top_level: bool = False):

    def decorator(controller_cls: Type["Controller"]):
        controller_cls_with_sub_request_marker = with_task_collection(
            "sub_request_marker",
            SubRequestMarker,
            node_name=node_name,
            is_top_level=is_top_level)(controller_cls)
        return controller_cls_with_sub_request_marker

    return decorator


class ChatCollection(TaskCollection):

    global_chat_task_set = set()

    def __init__(self):
        super().__init__()
        self.chat_tasks = []

    def before_yield(self, tasks: List[Task]):
        for task in tasks:
            if not isinstance(task, ChatTask):
                continue
            if id(task) not in ChatCollection.global_chat_task_set:
                ChatCollection.global_chat_task_set.add(id(task))
                self.chat_tasks.append(task)

    def __del__(self):
        for task in self.chat_tasks:
            ChatCollection.global_chat_task_set.remove(id(task))


class DropKVCacheWorkerTag(Enum):
    DROP_KV_CACHE = "drop_kv_cache"


def drop_kv_cache_scope():

    def decorator(controller_cls: Type["Controller"]):
        controller_cls_with_chat_collection = with_task_collection(
            "ChatCollection", ChatCollection)(controller_cls)
        original_process = controller_cls_with_chat_collection.process

        def new_process(self, tasks: List[Task], **kwargs):

            def wrapper():
                yield from original_process(self, tasks, **kwargs)

                drop_kv_cache_tasks = []
                for task in self.task_collections["ChatCollection"].chat_tasks:
                    drop_kv_cache_tasks.append(
                        DropKVCacheTask(
                            chat_task=task,
                            worker_tag=DropKVCacheWorkerTag.DROP_KV_CACHE))
                yield drop_kv_cache_tasks

            return wrapper()

        controller_cls_with_chat_collection.process = new_process
        return controller_cls_with_chat_collection

    return decorator
