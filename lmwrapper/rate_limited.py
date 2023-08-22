import time
import threading
from typing import Callable, ParamSpec, Concatenate, TypeVar

Param = ParamSpec("Param")
RetType = TypeVar("RetType")
OriginalFunc = Callable[Param, RetType]
DecoratedFunc = Callable[Concatenate[Param], RetType]


# https://stackoverflow.com/a/43727014
def rate_limited(max_per_minute: float) -> Callable[[OriginalFunc], DecoratedFunc]:
    """
    Decorator that make functions not be called faster than max_per_minute
    """

    lock = threading.Lock()
    minimum_interval = 1.0 / (float(max_per_minute) / 60.0)

    def decorator(func: OriginalFunc) -> DecoratedFunc:
        last_time_called = [0.0]

        def rate_limited_function(*args, **kwargs) -> RetType:
            lock.acquire()
            elapsed = time.perf_counter() - last_time_called[0]
            left_to_wait = minimum_interval - elapsed

            if left_to_wait > 0:
                time.sleep(left_to_wait)

            lock.release()

            ret = func(*args, **kwargs)
            last_time_called[0] = time.perf_counter()
            return ret

        return rate_limited_function

    return decorator
