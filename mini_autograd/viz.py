"""Visualisation of the autodiff graph and of the backward sweep.

The test suite proves that the gradients are right. This module exists to make
them visible: it walks the same DAG that `Tensor.backward()` walks, records what
happens at every step of the reverse sweep, and renders the result as a
terminal report, a Graphviz document, or a self-contained interactive page.

Nothing here is part of the engine. `record_backward` reproduces the sweep in
`tensor.py` rather than instrumenting it, so the hot path stays free of hooks
and the engine has no idea a visualiser exists. The one behavioural difference
is deliberate: the recorder never dismantles the graph, because a picture of a
graph that has already been freed would be a picture of nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .tensor import Tensor

# A tensor with more entries than this is summarised by its statistics alone;
# printing a 512x512 weight matrix into an HTML file helps nobody.
MAX_CELLS = 144


def _stats(arr: np.ndarray | None) -> dict[str, float] | None:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "norm": float(np.sqrt((a * a).sum())),
    }


def _cells(arr: np.ndarray | None) -> list[list[float]] | None:
    """A 2-D preview of the array, or None if it is too large to show."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.size > MAX_CELLS:
        return None
    return np.atleast_2d(a).tolist()


@dataclass
class Node:
    """One tensor in the recorded graph, plus where it sits on the canvas."""

    idx: int
    label: str
    op: str
    shape: tuple[int, ...]
    requires_grad: bool
    depth: int
    parents: list[int]
    value_stats: dict[str, float] | None
    value_cells: list[list[float]] | None
    grad_stats: dict[str, float] | None = None
    grad_cells: list[list[float]] | None = None
    x: float = 0.0
    y: float = 0.0


@dataclass
class Step:
    """One firing of the reverse sweep: a node pushes gradient to its inputs."""

    index: int
    node: int
    kind: str  # "seed" for the initial dL/dL, "backward" otherwise
    updates: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""


class Trace:
    """The recorded graph and the ordered list of backward steps."""

    def __init__(self, nodes: list[Node], steps: list[Step], root: int) -> None:
        self.nodes = nodes
        self.steps = steps
        self.root = root
        _layout(self.nodes)

    # ------------------------------------------------------------- terminal
    def to_text(self) -> str:
        """A report for people who are looking at a terminal, not a browser."""
        width = max(len(n.label) for n in self.nodes)
        lines = [
            f"graph: {len(self.nodes)} tensors, "
            f"{sum(len(n.parents) for n in self.nodes)} edges, "
            f"{len(self.steps)} backward steps",
            "",
            f"  {'tensor'.ljust(width)}  {'op':<7} {'shape':<12} "
            f"{'|value|':>10} {'|grad|':>10}",
            f"  {'-' * width}  {'-' * 7} {'-' * 12} {'-' * 10} {'-' * 10}",
        ]
        for n in self.nodes:
            v = f"{n.value_stats['norm']:.4g}" if n.value_stats else "-"
            g = f"{n.grad_stats['norm']:.4g}" if n.grad_stats else "-"
            lines.append(
                f"  {n.label.ljust(width)}  {(n.op or 'leaf'):<7} "
                f"{str(n.shape):<12} {v:>10} {g:>10}"
            )
        lines += ["", "backward sweep:"]
        for s in self.steps:
            src = self.nodes[s.node].label
            if not s.updates:
                lines.append(f"  {s.index:>3}. {src}: {s.note}")
                continue
            for u in s.updates:
                dst = self.nodes[u["node"]].label
                lines.append(
                    f"  {s.index:>3}. {src} -> {dst}"
                    f"   d|grad| = {u['delta_norm']:.4g}"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------- graphviz
    def to_dot(self) -> str:
        """Graphviz source, for anyone who would rather use `dot -Tpng`."""
        out = [
            "digraph autograd {",
            "  rankdir=LR;",
            '  node [shape=record, fontname="Helvetica", fontsize=10];',
        ]
        for n in self.nodes:
            g = f"{n.grad_stats['norm']:.3g}" if n.grad_stats else "-"
            colour = "#1F6B8F" if n.requires_grad else "#77848D"
            out.append(
                f'  n{n.idx} [color="{colour}", label="{{{n.label}'
                f"|{n.op or 'leaf'} {n.shape}|grad {g}}}\"];"
            )
        for n in self.nodes:
            for p in n.parents:
                out.append(f"  n{p} -> n{n.idx};")
        out.append("}")
        return "\n".join(out)

    # ------------------------------------------------------------------ web
    def to_json(self) -> str:
        return json.dumps(
            {
                "nodes": [
                    {
                        "idx": n.idx, "label": n.label, "op": n.op,
                        "shape": list(n.shape), "requiresGrad": n.requires_grad,
                        "depth": n.depth, "parents": n.parents,
                        "value": n.value_stats, "valueCells": n.value_cells,
                        "grad": n.grad_stats, "gradCells": n.grad_cells,
                        "x": round(n.x, 2), "y": round(n.y, 2),
                    }
                    for n in self.nodes
                ],
                "steps": [
                    {"index": s.index, "node": s.node, "kind": s.kind,
                     "updates": s.updates, "note": s.note}
                    for s in self.steps
                ],
                "root": self.root,
            },
            separators=(",", ":"),
        )

    def to_html(self, path: str | Path, title: str = "Backward pass",
                subtitle: str = "") -> Path:
        from .render import graph_page

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(graph_page(self, title, subtitle), encoding="utf-8")
        return path


def _layout(nodes: list[Node]) -> None:
    """Assign canvas coordinates: one column per depth, inputs on the left.

    Depth is the longest path from a leaf, which is what puts an operation to
    the right of every tensor that feeds it. Within a column the order is the
    order the nodes were created, so the picture is stable across runs.
    """
    NODE_W, NODE_H, GAP_X, GAP_Y, PAD = 188.0, 66.0, 74.0, 22.0, 28.0
    columns: dict[int, list[Node]] = {}
    for n in nodes:
        columns.setdefault(n.depth, []).append(n)
    tallest = max(len(c) for c in columns.values())
    full_h = tallest * NODE_H + (tallest - 1) * GAP_Y
    for depth, column in columns.items():
        h = len(column) * NODE_H + (len(column) - 1) * GAP_Y
        top = PAD + (full_h - h) / 2
        for row, n in enumerate(column):
            n.x = PAD + depth * (NODE_W + GAP_X)
            n.y = top + row * (NODE_H + GAP_Y)


def record_backward(
    loss: Tensor,
    names: Mapping[str, Tensor] | None = None,
    grad: Any | None = None,
) -> Trace:
    """Run the reverse sweep over `loss`, recording every gradient it writes.

    `names` labels tensors you care about (`{"W1": W1, "b1": b1}`); everything
    else is named after its operation. Existing gradients are cleared first so
    that the recording starts from a known state, and the graph is left intact
    afterwards -- unlike `Tensor.backward()`, which frees it.
    """
    if not loss.requires_grad:
        raise RuntimeError("record_backward() needs a tensor that requires grad")

    topo = loss._toposort()
    labels = {id(t): name for name, t in (names or {}).items()}

    index_of = {id(t): i for i, t in enumerate(topo)}
    counts: dict[str, int] = {}
    nodes: list[Node] = []
    for i, t in enumerate(topo):
        if id(t) in labels:
            label = labels[id(t)]
        else:
            kind = t._op or "input"
            counts[kind] = counts.get(kind, 0) + 1
            label = f"{kind}{counts[kind]}"
        parents = [index_of[id(p)] for p in t._prev]
        depth = 1 + max((nodes[p].depth for p in parents), default=-1)
        nodes.append(
            Node(
                idx=i, label=label, op=t._op, shape=t.shape,
                requires_grad=t.requires_grad, depth=depth, parents=parents,
                value_stats=_stats(t.data), value_cells=_cells(t.data),
            )
        )

    for t in topo:
        t.grad = None

    if grad is None:
        if loss.data.size != 1:
            raise RuntimeError(
                "record_backward() on a non-scalar tensor requires an explicit "
                f"grad argument (this tensor has shape {loss.shape})"
            )
        seed = np.ones_like(loss.data)
    else:
        seed = np.asarray(grad, dtype=loss.data.dtype)
        if seed.shape != loss.shape:
            raise ValueError(f"grad shape {seed.shape} != tensor shape {loss.shape}")

    loss.accumulate_grad(seed)
    root = index_of[id(loss)]
    steps = [
        Step(index=0, node=root, kind="seed", note="seed dL/dL = 1",
             updates=[{
                 "node": root, "delta_norm": float(np.sqrt((seed * seed).sum())),
                 "grad": _stats(loss.grad), "gradCells": _cells(loss.grad),
             }])
    ]

    for t in reversed(topo):
        if t._backward is None or t.grad is None:
            continue
        before = {
            id(p): (None if p.grad is None else p.grad.copy()) for p in t._prev
        }
        t._backward()
        updates = []
        for p in t._prev:
            if p.grad is None:
                continue
            prev = before[id(p)]
            delta = p.grad if prev is None else p.grad - prev
            if prev is not None and not np.any(delta):
                continue
            updates.append({
                "node": index_of[id(p)],
                "delta_norm": float(np.sqrt((delta * delta).sum())),
                "grad": _stats(p.grad),
                "gradCells": _cells(p.grad),
            })
        if updates:
            steps.append(Step(index=len(steps), node=index_of[id(t)],
                              kind="backward", updates=updates))

    for i, t in enumerate(topo):
        nodes[i].grad_stats = _stats(t.grad)
        nodes[i].grad_cells = _cells(t.grad)

    return Trace(nodes, steps, root)
