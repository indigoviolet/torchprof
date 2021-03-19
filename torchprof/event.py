from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, Dict, Iterable, List, Optional

import torch.autograd.profiler as tprofiler

from .trace import NN_MODULE_PREFIX


@dataclass
class Event:
    name: str
    id: int
    parent_id: Optional[int]
    children_ids: Iterable[int]
    self_cpu_time: float
    self_cuda_time: float
    cpu_time: float
    cuda_time: float
    count: float
    ancestor_ids: Optional[List[int]] = None

    events_by_id: ClassVar[Dict[int, Event]] = {}

    def __post_init__(self):
        Event.register(self)

    @classmethod
    def reset_registry(cls):
        # This is useful so that two separate profile runs don't have collisions
        # in events_by_id
        cls.events_by_id = {}

    @classmethod
    def register(cls, instance):
        assert (
            instance.id not in cls.events_by_id
        ), f"{instance=}, {cls.events_by_id[instance.id]=}"
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
        assert self.ancestor_ids is not None, self
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
        # We deliberately don't use evt.id anywhere because with
        # use_kineto=True, it seems these are not actually unique
        return Event(
            name=evt.name,
            id=id(evt),
            parent_id=(id(evt.cpu_parent) if evt.cpu_parent is not None else None),
            children_ids=[id(e) for e in evt.cpu_children],
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
            children_ids=set([c.path_id for c in rep.children]),
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

    def matches(self, res: List[re.Pattern], default: bool) -> bool:
        if not len(res):
            return default
        return any(p.search(self.name) for p in res)
