"""`python -m mini_autograd` -- differentiate an expression and print the sweep.

A zero-setup way to see the engine work. The expression is taken from the
command line, so this doubles as a scratchpad:

    python -m mini_autograd
    python -m mini_autograd --expr "(x @ W).relu().sum()"
    python -m mini_autograd --html /tmp/graph.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .tensor import Tensor
from .viz import record_backward

DEFAULT_EXPR = "((x @ W + b).relu() * y).sum()"


def environment(seed: int) -> dict[str, Tensor]:
    """The tensors an expression may refer to. `x` and `y` are data (no
    gradient); `W` and `b` are parameters (gradient wanted)."""
    rng = np.random.default_rng(seed)
    return {
        "x": Tensor(rng.normal(size=(3, 4))),
        "y": Tensor(rng.normal(size=(3, 2))),
        "W": Tensor(rng.normal(size=(4, 2)), requires_grad=True),
        "b": Tensor(np.zeros(2), requires_grad=True),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mini_autograd",
                                     description=__doc__)
    parser.add_argument("--expr", default=DEFAULT_EXPR,
                        help=f"expression over x, y, W, b (default: {DEFAULT_EXPR})")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--html", type=Path, default=None,
                        help="also write an interactive page to this path")
    args = parser.parse_args(argv)

    env = environment(args.seed)
    # eval over a namespace of tensors: the expression is the user's own input
    # on their own machine, and restricting it further would only get in the way.
    loss = eval(args.expr, {"__builtins__": {}}, dict(env))
    if not isinstance(loss, Tensor):
        parser.error(f"expression produced {type(loss).__name__}, not a Tensor")

    env["loss"] = loss
    trace = record_backward(loss, env)
    print(f"expression: {args.expr}\n")
    print(trace.to_text())

    if args.html:
        path = trace.to_html(args.html, title="Backward pass",
                             subtitle=f"<code>{args.expr}</code>")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
