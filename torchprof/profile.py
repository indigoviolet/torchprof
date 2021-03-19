from __future__ import annotations

import re
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

import torch.autograd.profiler as tprofiler
from tabulate import tabulate
from torch import nn
from tqdm import tqdm

from .event import Event
from .format import ColoredPrinter, format_time, format_us
from .trace import NN_MODULE_PREFIX, add_hook_trace, remove_hook_trace, walk_modules
from .tree import get_tree_labels


@contextmanager
def profile(
    model: nn.Module, enabled: bool = True, **kwargs
) -> Generator[Optional[ProfileParser], None, None]:
    if not enabled:
        with nullcontext():
            yield None
    else:
        Event.reset_registry()
        traces = list(walk_modules(model))
        try:
            for t in tqdm(traces, desc="Adding traces"):
                add_hook_trace(t)
            with tprofiler.profile(**kwargs) as prof:
                with tprofiler.record_function("Total"):
                    yield ProfileParser(prof)
        finally:
            for t in tqdm(traces, desc="Removing traces"):
                remove_hook_trace(t)


@dataclass
class ProfileParser:
    prof: tprofiler.profile = field(repr=False)

    _raw_events: Optional[List[Event]] = field(init=False, default=None)
    _events: Optional[List[Event]] = field(init=False, default=None)
    _totals: Optional[Dict[str, float]] = field(init=False, default=None)

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
        allow: List[str] = [f"^{NN_MODULE_PREFIX}", r"^region_profiler::"],
        block: List[str] = ["^aten::"],
        min_pct=1,
        display_empty_rows: bool = False,
        sort_by: List[str] = ["cuda_time"],
        filter_roots: bool = False,
        color: bool = True,
        terminal_jupyter: bool = True,
    ):

        allow_res = [re.compile(p) for p in allow]
        block_res = [re.compile(p) for p in block]

        def allow_fn(e):
            if not filter_roots and e.is_root:
                return True
            elif e.matches(block_res, False):
                return False
            else:
                return e.matches(allow_res, True)

        sort_key = lambda e: tuple(getattr(e, s) for s in sort_by)

        headers = ["Node", "Self CPU", "CPU", "Self CUDA", "CUDA", "#"]
        colalign = ("left", "right", "right", "right", "right")
        table = []

        cpu_time_total, cuda_time_total = (
            self.totals["self_cpu_time"],
            self.totals["self_cuda_time"],
        )

        colored_printer = ColoredPrinter(terminal_jupyter_hack=terminal_jupyter)

        for evt, label in tqdm(
            get_tree_labels(self.events, allow_fn, sort_key), desc="Make rows"
        ):

            formatted_cols = [
                format_time(evt.self_cpu_time, cpu_time_total, evt.count, min_pct),
                format_time(evt.cpu_time, cpu_time_total, evt.count, min_pct),
                format_time(evt.self_cuda_time, cuda_time_total, evt.count, min_pct),
                format_time(evt.cuda_time, cuda_time_total, evt.count, min_pct),
            ]

            if not display_empty_rows and not any(formatted_cols):
                continue

            cols = [label, *formatted_cols, evt.count]
            table.append(colored_printer.print(evt.level, *cols) if color else cols)

        print(f"\n\nCPU={format_us(cpu_time_total)}, CUDA={format_us(cuda_time_total)}")
        print(tabulate(table, headers=headers, tablefmt="psql", colalign=colalign))
