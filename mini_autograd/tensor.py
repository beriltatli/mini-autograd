"""The Tensor type and the reverse-mode backward sweep.

A Tensor is a NumPy array plus three pieces of bookkeeping:
  * `_prev`     : the tensors this one was computed from (edges of the DAG)
  * `_backward` : a closure that pushes this tensor's gradient into `_prev`
  * `grad`      : the accumulated dL/dself, filled in by `backward()`

The ops themselves live in ops.py; this file only knows how to walk the graph.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

ArrayLike = Any


def _as_array(data: ArrayLike) -> np.ndarray:
    """Coerce to a float ndarray. Integers become float64: gradients of an
    integer tensor are meaningless, and silently accumulating floats into an
    int array truncates."""
    arr = np.asarray(data)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr


class Tensor:
    """A node in the autodiff graph."""

    # __weakref__ is listed explicitly so tests can weakref intermediates and
    # prove the graph is actually released after backward().
    __slots__ = (
        "data", "grad", "requires_grad", "_prev", "_backward", "_op", "__weakref__",
    )

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _prev: tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        self.data: np.ndarray = _as_array(data)
        self.grad: np.ndarray | None = None
        self.requires_grad: bool = requires_grad
        self._prev: tuple[Tensor, ...] = _prev
        self._backward: Callable[[], None] | None = None
        self._op: str = _op

    # ---------------------------------------------------------------- basics
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    def __repr__(self) -> str:
        op = f", op={self._op}" if self._op else ""
        return (
            f"Tensor(shape={self.shape}, dtype={self.data.dtype}, "
            f"requires_grad={self.requires_grad}{op})"
        )

    def zero_grad(self) -> None:
        self.grad = None

    def accumulate_grad(self, grad: np.ndarray) -> None:
        """Add an incoming gradient. Accumulation (not assignment) is what makes
        a tensor with several consumers -- the diamond graph -- come out right."""
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = np.array(grad, dtype=self.data.dtype, copy=True)
        else:
            self.grad += grad

    def detach(self) -> "Tensor":
        """A view of the same data with no history. Used by optimizers/eval."""
        return Tensor(self.data, requires_grad=False)

    # ------------------------------------------------------------ graph walk
    def _toposort(self) -> list["Tensor"]:
        """Return every node reachable from self, inputs before consumers.

        Iterative post-order DFS: we push (node, expanded) pairs. The first time
        we pop a node we push it back marked `expanded` and then push all of its
        inputs on top of it; so the node is only appended to the output after
        every one of its inputs has been appended. Iterative rather than
        recursive because a deep net is a few hundred ops long and Python's
        default recursion limit is a poor reason for a framework to fall over.
        """
        topo: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._prev:
                if id(parent) not in visited:
                    stack.append((parent, False))
        return topo

    def backward(
        self,
        grad: ArrayLike | None = None,
        retain_graph: bool = False,
    ) -> None:
        """Populate `.grad` on every requires_grad tensor upstream of self.

        `grad` seeds dL/dself; it defaults to ones, which is only meaningful
        when self is a scalar loss, so we insist on that.
        Unless `retain_graph`, the DAG is dismantled on the way out (see below).
        """
        if not self.requires_grad:
            raise RuntimeError(
                "backward() called on a tensor that does not require grad"
            )

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() on a non-scalar tensor requires an explicit "
                    f"grad argument (this tensor has shape {self.shape})"
                )
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(grad, dtype=self.data.dtype)
            if seed.shape != self.shape:
                raise ValueError(
                    f"grad shape {seed.shape} != tensor shape {self.shape}"
                )

        topo = self._toposort()
        self.accumulate_grad(seed)

        # Reverse topological order: when we reach a node, every consumer of it
        # has already run, so its .grad is final and can be pushed upstream.
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward()

        if not retain_graph:
            _free_graph(topo)


def _free_graph(nodes: Iterable[Tensor]) -> None:
    """Drop each node's edges and closure.

    The closures capture their inputs and their output, so a live graph pins
    every intermediate activation in memory. Cutting `_prev`/`_backward` turns
    the whole DAG into garbage as soon as the loss goes out of scope, which is
    what keeps a 5000-step training loop flat. Leaf parameters survive: they are
    referenced by the model, and they keep their `.grad` for the optimizer.
    """
    for node in nodes:
        node._backward = None
        node._prev = ()
