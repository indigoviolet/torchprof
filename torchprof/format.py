from dataclasses import dataclass
from functools import cached_property
from typing import List, cast

import colorama
from rich.console import Console


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


def format_time(
    time: float,
    total: float,
    count: int,
    min_pct: float,
) -> str:
    if not total:
        return ""

    pct = time * 100.0 / total
    if pct < min_pct:
        return ""

    time_str = format_us(time)
    pct_str = f"{pct:.0f}%"
    avg_time_str = f"/{format_us(time / count)}" if count > 1 else ""

    return f"{time_str}{avg_time_str} ({pct_str})"


US_IN_S = 1000 * 1000
US_IN_MS = 1000


def format_us(v_us):
    if v_us > US_IN_S:
        return f"{(v_us / US_IN_S) :.1f}s"
    elif v_us > US_IN_MS:
        return f"{(v_us / US_IN_MS) :.1f}ms"
    else:
        return f"{(v_us ) :.1f}µs"
