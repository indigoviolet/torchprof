from typing import Callable, Generator, List, Tuple

from .event import Event


def get_tree_labels(
    events: List[Event],
    allow_fn: Callable[[Event], bool],
    sort_key: Callable[[Event], Tuple[int, ...]],
) -> Generator[Tuple[Event, str], None, None]:

    roots = sorted(
        [e for e in events if allow_fn(e) and e.is_root], key=sort_key, reverse=True
    )

    for e in roots:
        yield (e, e.label)
        yield from _tree_labels(e, " ", allow_fn, sort_key)


def _tree_labels(
    root: Event,
    stems: str,
    allow_fn: Callable[[Event], bool],
    sort_key: Callable[[Event], Tuple[int, ...]],
) -> Generator[Tuple[Event, str], None, None]:
    vert_stem, middle_branch, end_branch, space = (
        "\u2502",  # │
        "\u251c\u2500\u2500",  # ├──
        "\u2514\u2500\u2500",  # └──
        " ",
    )
    children = sorted(
        [c for c in root.children if allow_fn(c)], key=sort_key, reverse=True
    )
    num_children = len(children)
    for i, c in enumerate(children):
        last_child = i == num_children - 1
        branch = end_branch if last_child else middle_branch
        yield (c, stems + branch + c.label)

        desc_stems = stems + (space if last_child else vert_stem) + space
        yield from _tree_labels(c, desc_stems, allow_fn, sort_key)
