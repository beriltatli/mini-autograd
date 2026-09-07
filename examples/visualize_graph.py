"""Record and draw the backward pass of a small network.

Run:  python examples/visualize_graph.py

Writes an interactive page to site/graph.html, a Graphviz file to
site/graph.dot, and prints the same trace to the terminal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import _path  # noqa: F401  (sys.path shim; see examples/_path.py)
from mini_autograd import Tensor
from mini_autograd.viz import record_backward

ROOT = Path(__file__).resolve().parent.parent


def build() -> tuple[Tensor, dict[str, Tensor]]:
    """One layer of an MLP over a batch of four examples, plus a squared loss.

    Small on purpose: every tensor is under the display threshold, so the page
    shows real numbers rather than summary statistics.
    """
    rng = np.random.default_rng(0)

    x = Tensor(rng.normal(size=(4, 3)))
    y = Tensor(rng.normal(size=(4, 2)))
    W = Tensor(rng.normal(size=(3, 2)) * 0.8, requires_grad=True)
    b = Tensor(np.zeros(2), requires_grad=True)   # broadcast over the batch

    h = x @ W + b
    pred = h.relu()
    neg_y = (-1.0) * y
    err = pred + neg_y
    sq = err * err
    total = sq.sum()
    scale = Tensor(1.0 / y.size)       # mean over the 4x2 entries
    loss = total * scale

    names = {"x": x, "y": y, "W": W, "b": b, "h": h, "pred": pred,
             "neg_y": neg_y, "err": err, "sq": sq, "total": total,
             "scale": scale, "loss": loss}
    return loss, names


def main() -> None:
    loss, names = build()

    # Differentiate with the engine first, keeping the graph alive, and hold on
    # to the answer. The recorder clears every gradient before it starts, so
    # running it second cannot contaminate the reference.
    loss.backward(retain_graph=True)
    reference = {k: t.grad.copy() for k, t in names.items() if t.requires_grad}

    trace = record_backward(loss, names)
    for k, expected in reference.items():
        np.testing.assert_allclose(names[k].grad, expected)

    print(trace.to_text())

    page = trace.to_html(
        ROOT / "site" / "graph.html",
        title="One layer, one backward pass",
        subtitle=(
            "Every tensor produced by <code>x @ W + b &rarr; relu &rarr; "
            "squared error</code>, and the gradient each step of the reverse "
            "sweep writes into it. Click a tensor to read its values."
        ),
    )
    dot = ROOT / "site" / "graph.dot"
    dot.write_text(trace.to_dot(), encoding="utf-8")

    print(f"\nrecorded gradients match Tensor.backward() for "
          f"{', '.join(sorted(reference))}")
    print(f"wrote {page.relative_to(ROOT)}")
    print(f"wrote {dot.relative_to(ROOT)}  (render with: dot -Tsvg -O {dot.name})")


if __name__ == "__main__":
    main()
