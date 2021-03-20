from contextlib import contextmanager
from functools import wraps
from typing import Callable, Optional

import torch
import torch.autograd.profiler as tprofiler
from torch.cuda import nvtx

REGION_PREFIX = "torchprof_region::"

GLOBALS = {"nvtx": True, "sync_cuda": True}


def global_settings(**settings):
    global GLOBALS

    restore_globals = {**GLOBALS}
    try:
        GLOBALS.update(settings)
        yield
    finally:
        GLOBALS.update(restore_globals)


@contextmanager
def region(name: str):
    range_cm = (
        nvtx_range(name)
        if GLOBALS["nvtx"]
        else tprofiler.record_function(f"{REGION_PREFIX}{name}")
    )

    with range_cm:  # type: ignore[attr-defined]
        yield
        if GLOBALS["sync_cuda"]:
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
