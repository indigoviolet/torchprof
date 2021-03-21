from dataclasses import dataclass
from typing import Callable, Generator, Optional, Tuple

from torch import nn

from .annotate import module


@dataclass
class Trace:
    path: Tuple[str]
    leaf: bool
    module: nn.Module
    name: str

    _remove_hook: Optional[Callable[[], None]] = None

    def add_hook(self):
        name = ".".join(self.path)
        self._remove_hook = module(self.module, name)

    def remove_hook(self):
        assert self._remove_hook is not None, "Hook has not been added"
        self._remove_hook()
        self._remove_hook = None


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
