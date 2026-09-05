"""mini_autograd: a reverse-mode autodiff engine in pure NumPy."""

from .tensor import Tensor
from . import ops  # noqa: F401  (binds operators onto Tensor)
from .broadcast import unbroadcast

__all__ = ["Tensor", "ops", "unbroadcast"]
