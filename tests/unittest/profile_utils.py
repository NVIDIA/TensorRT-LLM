from contextlib import contextmanager
from functools import wraps

from tensorrt_llm import profiler


def profile(tag):

    @contextmanager
    def profile_range(tag):
        profiler.start(tag)
        yield
        profiler.stop(tag)

    def inner_decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            """A wrapper function"""
            with profile_range(tag):
                func(*args, **kwargs)
            profiler.summary()

        return wrapper

    return inner_decorator
