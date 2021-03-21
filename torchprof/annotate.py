from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, Optional, TypeVar, cast

import torch
import torch.autograd.profiler as tprofiler
from torch.cuda import nvtx

REGION_PREFIX = "torchprof_region::"

GLOBALS = {"nvtx": True, "sync_cuda": True}


@contextmanager
def global_settings(**settings) -> Generator[None, None, None]:
    global GLOBALS

    restore_globals = {**GLOBALS}
    try:
        GLOBALS.update(settings)
        yield
    finally:
        GLOBALS.update(restore_globals)


@contextmanager
def region(name: str) -> Generator[None, None, None]:
    range_cm = (
        lambda: nvtx_range(name)
        if GLOBALS["nvtx"]
        else lambda: tprofiler.record_function(f"{REGION_PREFIX}{name}")
    )
    # https://github.com/python/mypy/issues/5512
    with range_cm():  # type: ignore[attr-defined]
        yield
        if GLOBALS["sync_cuda"]:
            torch.cuda.synchronize()


@contextmanager
def nvtx_range(name: str) -> Generator[None, None, None]:
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


F = TypeVar("F", bound=Callable[..., Any])


def func(name: Optional[str] = None) -> Callable[[F], F]:
    def decorator(f: F) -> F:
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

        return cast(F, wrapper)

    return decorator
