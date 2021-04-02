from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, ContextManager, Iterator, Optional, TypeVar, cast

import torch
import torch.autograd.profiler as tprofiler
from torch.cuda import nvtx

REGION_PREFIX = "torchprof_region::"

GLOBALS = {"nvtx": True, "sync_cuda": True, "enabled": True}


@contextmanager
def global_settings(**settings) -> Iterator[None]:
    global GLOBALS

    restore_globals = {**GLOBALS}
    try:
        GLOBALS.update(settings)
        yield
    finally:
        GLOBALS.update(restore_globals)


@contextmanager
def region(name: str) -> Iterator[None]:
    if GLOBALS["enabled"]:
        maybe_sync()
        # https://github.com/python/mypy/issues/5512
        with named_range(name):
            yield
            maybe_sync()
    else:
        yield


def maybe_sync():
    if GLOBALS["sync_cuda"]:
        stream = torch.cuda.current_stream()
        if not stream.query():
            stream.synchronize()


def named_range(name: str) -> ContextManager[None]:
    # https://github.com/python/mypy/issues/5512#issuecomment-803490883
    return (
        nvtx_range(name)  # type: ignore[return-value]
        if GLOBALS["nvtx"]
        else tprofiler.record_function(f"{REGION_PREFIX}{name}")
    )


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
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


# Pytorch modules don't play nice with mypy. All attributes are assumed to be
# Tensor|Module, but this is not true. This is a quick way to eliminate type
# errors for Pytorch nn.Module
PytorchModule = Any


def module(mod: PytorchModule, name: str) -> Callable[[], None]:
    orig_forward = mod.forward

    @wraps(orig_forward)
    def _wrapper(*args, **kwargs):
        with region(name):
            res = orig_forward(*args, **kwargs)
        return res

    mod.forward = _wrapper

    def restore_forward():
        mod.forward = orig_forward

    return restore_forward
