from torchprof.annotate import func, global_settings, iter, module, region
from torchprof.profile import profile

name = "torchprof"

__all__ = ["profile", "region", "func", "iter", "module", "global_settings"]

import snoop

snoop.install()

__version__ = "1.0.0"
