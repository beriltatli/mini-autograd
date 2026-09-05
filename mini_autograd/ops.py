"""Primitive ops: forward value + local derivative.

Every op follows the same three-step shape:
    1. compute the forward value with NumPy
    2. wrap it in an output Tensor that remembers its inputs
    3. attach a closure that multiplies the incoming gradient by the local
       derivative and accumulates the result into each input

Notation in the comments: `g` is dL/d(out), the gradient flowing in from
downstream. Each op's job is to turn `g` into dL/d(input) for each input.

At the bottom of the file these functions are bound onto Tensor as operators
and methods. They are defined here rather than in tensor.py so that tensor.py
knows nothing about calculus and ops.py knows nothing about graph traversal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .broadcast import unbroadcast
from .tensor import Tensor


def _tensorify(x: Any) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _result(data: np.ndarray, parents: tuple[Tensor, ...], op: str) -> Tensor:
    """Build the output node. If nothing upstream needs a gradient we drop the
    edges entirely -- that is both faster and keeps inference from building a
    graph nobody will ever walk."""
    requires_grad = any(p.requires_grad for p in parents)
    return Tensor(
        data,
        requires_grad=requires_grad,
        _prev=parents if requires_grad else (),
        _op=op,
    )


def _norm_axes(axis: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    return tuple(a % ndim for a in axes)


# --------------------------------------------------------------------- add
# out = a + b        d out/d a = 1        d out/d b = 1
# The gradient passes through untouched -- except that broadcasting may have
# replicated a or b in the forward pass, so we sum those copies back down.
def add(a: Any, b: Any) -> Tensor:
    a, b = _tensorify(a), _tensorify(b)
    out = _result(a.data + b.data, (a, b), "add")

    if out.requires_grad:
        def _backward() -> None:
            g = out.grad
            if a.requires_grad:
                a.accumulate_grad(unbroadcast(g, a.shape))
            if b.requires_grad:
                b.accumulate_grad(unbroadcast(g, b.shape))

        out._backward = _backward
    return out


# --------------------------------------------------------------------- mul
# out = a * b        d out/d a = b        d out/d b = a
def mul(a: Any, b: Any) -> Tensor:
    a, b = _tensorify(a), _tensorify(b)
    out = _result(a.data * b.data, (a, b), "mul")

    if out.requires_grad:
        def _backward() -> None:
            g = out.grad
            if a.requires_grad:
                a.accumulate_grad(unbroadcast(g * b.data, a.shape))
            if b.requires_grad:
                b.accumulate_grad(unbroadcast(g * a.data, b.shape))

        out._backward = _backward
    return out


# ------------------------------------------------------------------ matmul
# out = A @ B        dL/dA = g @ B^T        dL/dB = A^T @ g
# For stacked (batched) matmul, NumPy broadcasts the leading batch dims; the
# matrix dims always line up exactly, so unbroadcast only ever folds batch axes.
def matmul(a: Any, b: Any) -> Tensor:
    a, b = _tensorify(a), _tensorify(b)
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError(
            "matmul requires both operands to be at least 2-D "
            f"(got {a.shape} @ {b.shape}); reshape a vector to (1, n) or (n, 1)"
        )
    out = _result(a.data @ b.data, (a, b), "matmul")

    if out.requires_grad:
        def _backward() -> None:
            g = out.grad
            if a.requires_grad:
                a.accumulate_grad(unbroadcast(g @ b.data.swapaxes(-1, -2), a.shape))
            if b.requires_grad:
                b.accumulate_grad(unbroadcast(a.data.swapaxes(-1, -2) @ g, b.shape))

        out._backward = _backward
    return out


# --------------------------------------------------------------------- sum
# out = sum(x, axis)   d out/d x_i = 1
# Backward is the exact mirror of forward: sum replicates one gradient over
# every element that fed it, i.e. a broadcast. The only fiddly part is that with
# keepdims=False the reduced axes were dropped, so we put them back before
# broadcasting.
def sum(t: Any, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
    t = _tensorify(t)
    out = _result(t.data.sum(axis=axis, keepdims=keepdims), (t,), "sum")

    if out.requires_grad:
        def _backward() -> None:
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, _norm_axes(axis, t.ndim))
            t.accumulate_grad(np.broadcast_to(g, t.shape))

        out._backward = _backward
    return out


# -------------------------------------------------------------------- relu
# out = max(x, 0)     d out/d x = 1 if x > 0 else 0
# At exactly x = 0 the derivative does not exist; we pick 0, which is what
# PyTorch does. The numeric gradient checker must not probe x = 0 for this op.
def relu(t: Any) -> Tensor:
    t = _tensorify(t)
    out = _result(np.maximum(t.data, 0.0), (t,), "relu")

    if out.requires_grad:
        def _backward() -> None:
            t.accumulate_grad(out.grad * (t.data > 0.0))

        out._backward = _backward
    return out


# ------------------------------------------------------- bind onto Tensor
Tensor.__add__ = lambda self, other: add(self, other)
Tensor.__radd__ = lambda self, other: add(other, self)
Tensor.__mul__ = lambda self, other: mul(self, other)
Tensor.__rmul__ = lambda self, other: mul(other, self)
Tensor.__matmul__ = lambda self, other: matmul(self, other)
Tensor.__rmatmul__ = lambda self, other: matmul(other, self)
Tensor.sum = sum
Tensor.relu = relu
