import jax
from typing import Any

RandomKey = jax.Array
Kernel = callable  # kernel(key, x) -> new_x
KernelParam = Any

__all__ = ["RandomKey", "Kernel", "KernelParam"]
