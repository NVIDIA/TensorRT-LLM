from typing import List, Type

from .controller import ParallelProcess
from .task import GenerationTask, Task


class TaskCollection:

    def __init__(self):
        # reserved for future use
        pass

    def before_yield(self, tasks: List[Task]):
        pass

    def after_yield(self, tasks: List[Task]):
        pass


def with_task_collection(name: str, task_collection_cls: Type[TaskCollection]):

    def decorator(controller_cls: Type["Controller"]):
        original_init = controller_cls.__init__
        original_process = controller_cls.process

        # add task collection to controller
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.task_collections[name] = task_collection_cls()

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
