"""Shape utilities for reverse-mode broadcasting.

Forward broadcasting is free: NumPy expands operands for us. The backward pass
has to undo that expansion. If the forward pass copied one value into N slots,
the backward pass must sum the N incoming gradients back into that one slot.
`unbroadcast` is the single place in this repo where that happens.
"""

from __future__ import annotations

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum `grad` back down to `shape`, inverting NumPy broadcasting.

    NumPy broadcasts in exactly two steps, so we undo exactly two steps:

      1. Rank expansion: a (4,) operand used against (2, 3, 4) is first treated
         as (1, 1, 4). Those leading axes are pure replication, so we sum them
         away entirely.
      2. Size-1 stretching: an axis of extent 1 is stretched to extent n. That
         is also replication, so we sum over it but keep the axis (keepdims)
         because the original operand still has a length-1 axis there.

    Anything else is a shape bug in the caller, and the final reshape will raise
    rather than silently returning the wrong thing.
    """
    grad = np.asarray(grad)
    if grad.shape == shape:
        return grad

    # (1) undo rank expansion
    extra_dims = grad.ndim - len(shape)
    if extra_dims > 0:
        grad = grad.sum(axis=tuple(range(extra_dims)))
    elif extra_dims < 0:
        raise ValueError(f"cannot unbroadcast {np.shape(grad)} down to {shape}")

    # (2) undo size-1 stretching
    axes = tuple(i for i, dim in enumerate(shape) if dim == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)

    # Defensive: shapes must now agree exactly. reshape raises if they don't.
    return np.asarray(grad).reshape(shape)
