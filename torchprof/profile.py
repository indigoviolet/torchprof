from __future__ import annotations

import re
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import cached_property, wraps
from typing import Any, Callable, ClassVar, Dict, Generator, List, Optional, Tuple, cast

import attr
import colorama
import torch.autograd.profiler as tprofiler
from rich.console import Console
from tabulate import tabulate
from torch import nn
from tqdm import tqdm

NN_MODULE_PREFIX = "torch_nn_module::"


@dataclass
class ColoredPrinter:
    terminal_jupyter_hack: bool

    COLORS = {
        0: "white",
        1: "red",
        2: "green",
        3: "blue",
        4: "yellow",
        5: "magenta",
        6: "cyan",
        7: "red",
        8: "green",
        9: "blue",
        10: "yellow",
        11: "magenta",
        12: "cyan",
    }

    def print(self, level: int, *txts: str) -> List[str]:
        color = self.COLORS.get(level) or "white"
        return [self.print_one(t, color) for t in txts]

    def print_one(self, txt: str, color: str) -> str:
        with self.console.capture() as capture:
            self.console.print(f"[{color}]{txt}[/]")
        return cast(str, capture.get().strip())

    @cached_property
    def console(self):
        # https://github.com/willmcgugan/rich/issues/870
        #
        # In emacs jupyter (ein), we need this hack to not print
        # =<rich.jupyter.JupyterRenderable>= everywhere
        #
        # Even though we have to specify =color_system=truecolor=, this will only
        # print 8 colors and not even the bright versions (at least in my usage)

        colorama.deinit()
        if self.terminal_jupyter_hack:
            colorama.init(convert=False, strip=False)
            return Console(force_jupyter=False, color_system="truecolor")
        else:
            colorama.init()
            return Console()


@attr.s(auto_attribs=True)
class Event:
    name: str
    id: int
    parent_id: Optional[int]
    children_ids: List[int]
    self_cpu_time: float
    self_cuda_time: float
    cpu_time: float
    cuda_time: float
    count: float
    ancestor_ids: Optional[List[int]] = attr.ib(default=None)

    events_by_id: ClassVar[Dict[int, Event]] = {}

    def __attrs_post_init__(self):
        Event.register(self)

    @classmethod
    def register(cls, instance):
        cls.events_by_id[instance.id] = instance

    @cached_property
    def parent(self):
        if self.is_root:
            return None
        assert self.parent_id is not None
        return Event.events_by_id[self.parent_id]

    @cached_property
    def label(self):
        if (
            not self.is_root
            and self.name.startswith(NN_MODULE_PREFIX)
            and self.name.startswith(self.parent.name)
        ):
            # this adds one char for delimiter
            return self.name[(len(self.parent.name) + 1) :]
        else:
            parts = self.name.split("::")
            return self.name if parts[0] == "aten" else parts[-1]

    @property
    def is_root(self):
        return self.parent_id is None

    @property
    def level(self) -> int:
        return len(self.ancestor_ids) if self.ancestor_ids is not None else 0

    @cached_property
    def children(self):
        return [Event.events_by_id[i] for i in self.children_ids]

    @cached_property
    def ancestors(self):
        assert self.ancestor_ids is not None
        return [Event.events_by_id[i] for i in self.ancestor_ids]

    @cached_property
    def ancestor_names(self):
        return [e.name for e in self.ancestors]

    @cached_property
    def path(self):
        return self.ancestor_names + [self.name]

    @cached_property
    def path_id(self):
        return hash(tuple(self.path))

    @classmethod
    def from_function_event(cls, evt: tprofiler.FunctionEvent) -> Event:
        return Event(
            name=evt.name,
            id=evt.id,
            parent_id=(evt.cpu_parent.id if evt.cpu_parent is not None else None),
            children_ids=[e.id for e in evt.cpu_children],
            self_cpu_time=evt.self_cpu_time_total,
            self_cuda_time=evt.self_cuda_time_total,
            cpu_time=evt.cpu_time_total,
            cuda_time=evt.cuda_time_total,
            count=evt.count,
        )

    @classmethod
    def from_group(cls, evts: List[Event]) -> Event:
        rep = evts[0]
        return Event(
            name=rep.name,
            id=rep.path_id,
            parent_id=(rep.parent.path_id if rep.parent_id is not None else None),
            children_ids=list(set([c.path_id for c in rep.children])),
            self_cpu_time=cls._get_sum(evts, "self_cpu_time"),
            self_cuda_time=cls._get_sum(evts, "self_cuda_time"),
            cpu_time=cls._get_sum(evts, "cpu_time"),
            cuda_time=cls._get_sum(evts, "cuda_time"),
            count=cls._get_sum(evts, "count"),
            ancestor_ids=list(set([a.path_id for a in rep.ancestors])),
        )

    @classmethod
    def _get_sum(cls, evts: List[Event], attr_name: str) -> float:
        return sum(getattr(e, attr_name) for e in evts)

    def matches(self, res: List[re.Pattern]) -> bool:
        if not len(res):
            return True
        return any(p.search(self.name) for p in res)


@attr.s(auto_attribs=True)
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


@contextmanager
def profile(
    model: nn.Module, enabled: bool = True, **kwargs
) -> Generator[Optional[ProfileParser], None, None]:
    if not enabled:
        with nullcontext():
            yield None
    else:
        traces = list(walk_modules(model))
        try:
            for t in tqdm(traces, desc="Adding traces"):
                _add_hook_trace(t)
            with tprofiler.profile(**kwargs) as prof:
                with tprofiler.record_function("Total"):
                    yield ProfileParser(prof)
        finally:
            for t in tqdm(traces, desc="Removing traces"):
                _remove_hook_trace(t)


# Pytorch modules don't play nice with mypy. All attributes are assumed to be
# Tensor|Module, but this is not true. This is a quick way to eliminate type
# errors for Pytorch nn.Module
PytorchModule = Any


def _add_hook_trace(trace: Trace):
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


def _remove_hook_trace(trace: Trace):
    module: PytorchModule = trace.module
    if hasattr(module, "_orig_forward"):
        module.forward = module._orig_forward
        delattr(module, "_orig_forward")


@attr.s(auto_attribs=True)
class ProfileParser:
    prof: tprofiler.profile = attr.ib(
        repr=lambda p: f"{p.__class__}<{len(p.function_events)}>"
    )

    _raw_events: Optional[List[Event]] = attr.ib(init=False, default=None)
    _events: Optional[List[Event]] = attr.ib(init=False, default=None)
    _totals: Optional[Dict[str, float]] = attr.ib(init=False, default=None)

    @property
    def raw_events(self):
        if self._raw_events is None:
            self.parse()
        return self._raw_events

    @property
    def events(self):
        if self._events is None:
            self.parse()
        return self._events

    @property
    def totals(self):
        if self._totals is None:
            self.parse()
        return self._totals

    def parse(self):
        function_events: tprofiler.EventList = self.prof.function_events  # type: ignore[assignment]

        # populate_cpu_children() was made private (and unnecessary in pytorch 1.8)
        if hasattr(function_events, "populate_cpu_children"):
            function_events.populate_cpu_children()

        events: List[Event] = [
            Event.from_function_event(e)
            for e in tqdm(function_events, desc="Make Events")
        ]

        # populate ancestors
        for evt in tqdm(events, desc="Populate ancestors"):
            if evt.parent is None:
                self._populate_ancestors(evt, [])

        # group by path
        events_by_path: Dict[int, List[Event]] = defaultdict(list)
        for evt in tqdm(events, desc="Group"):
            events_by_path[evt.path_id].append(evt)

        # aggregate
        agg_events = []
        totals: Dict[str, float] = defaultdict(lambda: 0.0)
        for group in tqdm(events_by_path.values(), desc="Aggregate"):
            aevt = Event.from_group(group)
            agg_events.append(aevt)

            for a in ["self_cpu_time", "self_cuda_time"]:
                totals[a] += getattr(aevt, a)

        self._raw_events = events
        self._events = agg_events
        self._totals = totals

    def _populate_ancestors(self, evt: Event, path: List[int]):
        evt.ancestor_ids = path
        for c in evt.children:
            self._populate_ancestors(c, path + [evt.id])

    def display(
        self,
        filter: List[str] = [r"^torchprof_nn_module::", r"^region_profiler::"],
        min_pct=1,
        display_empty_rows: bool = False,
        sort_by: List[str] = ["cuda_time"],
        filter_roots: bool = False,
        color: bool = True,
        terminal_jupyter: bool = True,
    ):

        filter_res = [re.compile(p) for p in filter]

        filter_fn = lambda e: (
            True if (not filter_roots and e.is_root) else e.matches(filter_res)
        )

        sort_key = lambda e: tuple(getattr(e, s) for s in sort_by)

        headers = ["Node", "Self CPU", "CPU", "Self CUDA", "CUDA", "Count"]
        colalign = ("left", "right", "right", "right", "right")
        table = []

        cpu_time_total, cuda_time_total = (
            self.totals["self_cpu_time"],
            self.totals["self_cuda_time"],
        )

        colored_printer = ColoredPrinter(terminal_jupyter_hack=terminal_jupyter)

        for evt, label in tqdm(
            get_tree_labels(self.events, filter_fn, sort_key), desc="Make rows"
        ):

            formatted_cols = [
                format_time(evt.self_cpu_time, cpu_time_total, min_pct),
                format_time(evt.cpu_time, cpu_time_total, min_pct),
                format_time(evt.self_cuda_time, cuda_time_total, min_pct),
                format_time(evt.cuda_time, cuda_time_total, min_pct),
            ]

            if not display_empty_rows and not any(formatted_cols):
                continue

            cols = [label, *formatted_cols, evt.count]
            table.append(colored_printer.print(evt.level, *cols) if color else cols)

        print(f"\n\nCPU={format_us(cpu_time_total)}, CUDA={format_us(cuda_time_total)}")
        print(tabulate(table, headers=headers, tablefmt="psql", colalign=colalign))


def format_time(time, total, min_pct):
    pct = time * 100.0 / total
    if pct >= min_pct:
        return f"{format_us(time)} ({pct:.0f}%)"
    else:
        return ""


US_IN_S = 1000 * 1000
US_IN_MS = 1000


def format_us(v_us):
    if v_us > US_IN_S:
        return f"{(v_us / US_IN_S) :.1f}s"
    elif v_us > US_IN_MS:
        return f"{(v_us / US_IN_MS) :.1f}ms"
    else:
        return f"{(v_us ) : .1f}us"


def get_tree_labels(
    events: List[Event],
    filter_fn: Callable[[Event], bool],
    sort_key: Callable[[Event], Tuple[int, ...]],
) -> Generator[Tuple[Event, str], None, None]:

    roots = sorted(
        [e for e in events if filter_fn(e) and e.is_root], key=sort_key, reverse=True
    )

    for e in roots:
        yield (e, e.label)
        yield from _tree_labels(e, " ", filter_fn, sort_key)


def _tree_labels(
    root: Event,
    stems: str,
    filter_fn: Callable[[Event], bool],
    sort_key: Callable[[Event], Tuple[int, ...]],
) -> Generator[Tuple[Event, str], None, None]:
    vert_stem, middle_branch, end_branch, space = (
        "\u2502",  # │
        "\u251c\u2500\u2500",  # ├──
        "\u2514\u2500\u2500",  # └──
        " ",
    )
    children = sorted(
        [c for c in root.children if filter_fn(c)], key=sort_key, reverse=True
    )
    num_children = len(children)
    for i, c in enumerate(children):
        last_child = i == num_children - 1
        branch = end_branch if last_child else middle_branch
        yield (c, stems + branch + c.label)

        desc_stems = stems + (space if last_child else vert_stem) + space
        yield from _tree_labels(c, desc_stems, filter_fn, sort_key)
