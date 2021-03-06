from __future__ import annotations

import re
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import (
    Callable,
    ContextManager,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Union,
)

import torch.autograd.profiler as tprofiler
from tabulate import tabulate
from torch import nn
from tqdm import tqdm

from .annotate import REGION_PREFIX, global_settings, region
from .event import Event
from .format import ColoredPrinter, format_time, format_us
from .trace import walk_modules
from .tree import get_tree_labels

TOP_REGION = "<torchprof>"


@contextmanager
def profile(
    model: nn.Module,
    enabled: bool = True,
    nvtx: bool = False,
    progress: bool = True,
    sync_cuda: bool = True,
    **profiler_kwargs,
) -> Iterator[Optional[ProfileParser]]:
    """

    :param model: nn.Module:
    :param enabled: bool:  (Default value = True)
    :param nvtx: bool:  (Default value = False) Emit nvtx events instead of torch profiler `record_function()`. Useful for Nsight Systems etc.
    :param progress: bool:  (Default value = True) Show progress bar.
    :param sync_cuda: bool:  (Default value = True) Call `torch.cuda.synchronize` on the current stream if necessary.
    :param **profiler_kwargs: Passed through to torch.profiler -- only relevant if `nvtx` is False

    """

    if not enabled:
        with nullcontext():
            yield None
    else:
        Event.reset_registry()
        traces = list(walk_modules(model))
        try:
            for t in tqdm(traces, desc="Adding traces", disable=not progress):
                t.add_hook()
            with global_settings(nvtx=nvtx, sync_cuda=sync_cuda):
                # invoking these functions will emit the region(), so wrap in lambda
                profile_cm: Union[
                    Callable[[], ContextManager[None]],
                    Callable[[], ContextManager[ProfileParser]],
                ] = (
                    _nvtx_profile
                    if nvtx
                    else (lambda: _torch_profile(**profiler_kwargs))  # type: ignore[return-value]
                )
                with profile_cm() as p:
                    yield p
        finally:
            for t in tqdm(traces, desc="Removing traces", disable=not progress):
                t.remove_hook()


@contextmanager
def _nvtx_profile() -> Iterator[None]:
    with tprofiler.emit_nvtx():
        with region(TOP_REGION):
            yield


@contextmanager
def _torch_profile(**profiler_kwargs) -> Iterator[ProfileParser]:
    with tprofiler.profile(**profiler_kwargs) as prof:
        with region(TOP_REGION):
            yield ProfileParser(prof)


@dataclass
class ProfileParser:
    prof: tprofiler.profile = field(repr=False)

    _raw_events: Optional[List[Event]] = field(init=False, default=None)
    _events: Optional[List[Event]] = field(init=False, default=None)
    _totals: Optional[Dict[str, float]] = field(init=False, default=None)

    @property
    def raw_events(self) -> List[Event]:
        if self._raw_events is None:
            self.parse()
        assert self._raw_events is not None
        return self._raw_events

    @property
    def events(self) -> List[Event]:
        if self._events is None:
            self.parse()
        assert self._events is not None
        return self._events

    @property
    def totals(self) -> Dict[str, float]:
        if self._totals is None:
            self.parse()
        assert self._totals is not None
        return self._totals

    def parse(self):
        function_events: tprofiler.EventList = self.prof.function_events  # type: ignore[assignment]

        # populate_cpu_children() was made private (and unnecessary in pytorch 1.8)
        if hasattr(function_events, "populate_cpu_children"):
            function_events.populate_cpu_children()  # type: ignore[attr-defined]

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
        allow: List[str] = [
            f"^{REGION_PREFIX}",
            r"^region_profiler::",
        ],
        block: List[str] = ["^aten::"],
        min_pct=1,
        display_empty_rows: bool = False,
        sort_by: List[str] = ["cuda_time"],
        filter_roots: bool = False,
        color: bool = True,
        terminal_jupyter: bool = True,
    ):
        """

        :param allow: List[str]:  (Default value = [f"^{REGION_PREFIX}", r"^region_profiler::"]): allowlist
        :param block: List[str]:  (Default value = ["^aten::"]): blocklist
        :param min_pct:  (Default value = 1)
        :param display_empty_rows: bool:  (Default value = False)
        :param sort_by: List[str]:  (Default value = ["cuda_time"])
        :param filter_roots: bool:  (Default value = False)
        :param color: bool:  (Default value = True)
        :param terminal_jupyter: bool:  (Default value = True): Handles using Jupyter in a terminal, where html colors aren't supported. Only relevant if color=True

        """

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
        print(f"\n\nCPU={format_us(cpu_time_total)}, CUDA={format_us(cuda_time_total)}")
        if not len(self.events):
            return

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

        print(tabulate(table, headers=headers, tablefmt="psql", colalign=colalign))
