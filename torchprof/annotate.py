from contextlib import contextmanager
from functools import wraps
from typing import Callable, Optional

import torch
import torch.autograd.profiler as tprofiler
from torch.cuda import nvtx

REGION_PREFIX = "torchprof_region::"

CUDA_SYNCHRONIZE = True


def cuda_synchronize(val: Optional[bool] = None):
    global CUDA_SYNCHRONIZE
    if val is not None:
        CUDA_SYNCHRONIZE = val
    return CUDA_SYNCHRONIZE


@contextmanager
def region(name: str):
    with tprofiler.record_function(f"{REGION_PREFIX}{name}"):
        with nvtx_range(name):
            yield
            if cuda_synchronize():
                torch.cuda.synchronize()


@contextmanager
def nvtx_range(name: str):
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


def func(name: Optional[str] = None):
    def decorator(f: Callable):
        nonlocal name
        if name is None:
            name = f.__name__

        name += "()"

        @wraps(f)
        def wrapper(*args, **kwargs):
            nonlocal name
            assert name is not None
            with region(name):
                return f(*args, **kwargs)

        return wrapper

    return decorator
