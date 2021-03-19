from dataclasses import dataclass
from functools import wraps
from typing import Any, Generator, Tuple

import torch.autograd.profiler as tprofiler
from torch import nn

NN_MODULE_PREFIX = "torch_nn_module::"


@dataclass
class Trace:
    path: Tuple[str]
    leaf: bool
    module: nn.Module
    name: str


def walk_modules(module, name="", path=()) -> Generator[Trace, None, None]:
    """Generator. Walks through a PyTorch Module and outputs Trace tuples"""
    if not name:
        name = module.__class__.__name__
    named_children = list(module.named_children())
    path = path + (name,)
    yield Trace(path, len(named_children) == 0, module, name)
    # recursively walk into all submodules
    for name, child_module in named_children:
        yield from walk_modules(child_module, name=name, path=path)


# Pytorch modules don't play nice with mypy. All attributes are assumed to be
# Tensor|Module, but this is not true. This is a quick way to eliminate type
# errors for Pytorch nn.Module
PytorchModule = Any


def add_hook_trace(trace: Trace):
    module: PytorchModule = trace.module
    if hasattr(module, "_orig_forward"):
        return

    module._orig_forward = module.forward
    name = NN_MODULE_PREFIX + ".".join(trace.path)

    @wraps(module._orig_forward)
    def _wrapper(*args, **kwargs):
        with tprofiler.record_function(name):
            res = module._orig_forward(*args, **kwargs)
        return res

    module.forward = _wrapper


def remove_hook_trace(trace: Trace):
    module: PytorchModule = trace.module
    if hasattr(module, "_orig_forward"):
        module.forward = module._orig_forward
        delattr(module, "_orig_forward")
