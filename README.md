# mini-autograd

A reverse-mode automatic differentiation engine written in NumPy, together with
a visualiser that draws what the engine does.

The library implements the mechanism underneath frameworks such as PyTorch: it
records the operations performed on a tensor into a directed acyclic graph, then
walks that graph backwards to compute the derivative of a scalar output with
respect to every leaf that asked for one. There are five primitive operations,
no optimizer, and no neural-network layer abstraction. What is present is the
graph machinery itself, documented at the level of detail a reader
reconstructing the algorithm would need.

The engine is correct — the test suite compares it against finite differences
and against PyTorch — but correctness is not visible. So the second half of the
repository exists to make the backward pass legible: `mini_autograd.viz` records
the reverse sweep step by step and renders it as a terminal report, a Graphviz
document, or an interactive page on which the gradients appear one at a time.

## Requirements

- Python 3.11 or newer
- NumPy 1.24 or newer

PyTorch and pytest are development dependencies. Nothing under `mini_autograd/`
imports either; PyTorch appears only in `tests/test_against_torch.py`, as a
second opinion on every gradient.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

An editable install puts the package on the path, which is what makes `pytest`
and the example scripts work from any directory. Without it, run pytest through
the interpreter (`python -m pytest`) so that the project root lands on
`sys.path`; the example scripts insert it themselves.

## Running it

Three entry points, in increasing order of how much they show.

**The engine, from the command line.** Differentiate an expression and print the
sweep. `x` and `y` are data, `W` and `b` are parameters:

```sh
python -m mini_autograd
python -m mini_autograd --expr "((x @ W).relu() * y).sum()"
python -m mini_autograd --html /tmp/graph.html
```

```
  tensor   op      shape           |value|     |grad|
  -------  ------- ------------ ---------- ----------
  W        leaf    (4, 2)            2.129      1.624
  matmul1  matmul  (3, 2)            2.773     0.9382
  ...
backward sweep:
    4. add1 -> matmul1   d|grad| = 0.9382
    4. add1 -> b         d|grad| = 1.096
    5. matmul1 -> W      d|grad| = 1.624
```

**The graph, drawn.** Builds one layer with a squared-error loss, records the
backward pass, and writes an interactive page:

```sh
python examples/visualize_graph.py     # -> site/graph.html, site/graph.dot
```

The page shows every tensor on the tape. Stepping forward fires one node of the
reverse sweep at a time: the firing node and the edges carrying gradient are
highlighted, the receiving tensors are shaded, and their gradient norms update.
Clicking a tensor prints its values and its current gradient as matrices. The
script also verifies, before drawing anything, that the recorded gradients equal
what `Tensor.backward()` produces.

**A model, trained.** Fits a 48-unit ReLU network to sin(3x) + 0.3x using only
the five primitives, and writes a report:

```sh
python examples/train_mlp.py           # -> site/training.html
```

The report plots the loss, the gradient norm of each parameter, and the function
the network represents at six checkpoints, so the fit can be watched assembling
itself out of ReLU kinks. Final training MSE is about 0.003.

Open `index.html` for the written walkthrough; it links to both generated
pages. The same three files are published with GitHub Pages from the repository
root, so `index.html` is the site's landing page.

## Testing

```sh
python -m pytest
```

92 tests, in four files:

| file | what it establishes |
| --- | --- |
| `test_graph.py` | topological order, accumulation across shared subgraphs, teardown |
| `test_broadcast.py` | `unbroadcast` against a reference and against finite differences |
| `test_against_torch.py` | every gradient against PyTorch, to 1e-10 |
| `test_viz.py` | the recorder agrees with the engine; the generated page is well formed |

The last file is doing something slightly unusual. Since a browser is not
available in the test environment, it asserts the properties a screenshot would
otherwise confirm: that the embedded trace is valid JSON, that every element the
page's script reaches for exists in the markup, that no template placeholder
survived, and that the layout puts every tensor to the left of the operations
consuming it.

## Usage

The public surface is a single class. A tensor is constructed from anything
NumPy can coerce to an array; integer inputs are promoted to `float64`, since
gradients accumulated into an integer buffer would be truncated.

```python
import numpy as np
from mini_autograd import Tensor

x = Tensor(np.random.randn(8, 4))                    # input, no gradient
W = Tensor(np.random.randn(4, 3), requires_grad=True)
b = Tensor(np.zeros(3), requires_grad=True)

loss = (x @ W + b).relu().sum()
loss.backward()

W.grad.shape   # (4, 3)
b.grad.shape   # (3,)
```

To record a sweep instead of merely running one:

```python
from mini_autograd.viz import record_backward

trace = record_backward(loss, {"x": x, "W": W, "b": b, "loss": loss})
print(trace.to_text())
trace.to_html("graph.html", title="One backward pass")
```

`record_backward` reproduces the sweep rather than instrumenting it, so the
engine's hot path carries no hooks and does not know a visualiser exists. It
differs from `Tensor.backward()` in two deliberate ways: it clears existing
gradients first, so the recording starts from a known state, and it leaves the
graph intact afterwards, because a picture of a graph that has already been
freed would be a picture of nothing.

Three properties of the interface are worth stating explicitly, because each
mirrors a decision made by PyTorch.

`backward()` may be called without an argument only on a scalar. The seed
gradient defaults to ones, which is meaningful for a loss and meaningless for
anything else, so a non-scalar tensor raises unless an explicit seed of matching
shape is supplied.

Gradients accumulate rather than overwrite. Two consecutive backward passes
leave `x.grad` at twice its single-pass value; clearing is the caller's
responsibility, via `zero_grad()`.

The graph is dismantled once traversed. Each node's edges and backward closure
are dropped on the way out, so intermediate activations become collectable as
soon as the loss goes out of scope — `test_graph.py` asserts this with a weak
reference. `retain_graph=True` suppresses the teardown.

## Structure

```
mini_autograd/
  tensor.py      Tensor, topological sort, backward sweep, graph teardown
  ops.py         forward values and local derivatives; binds operators onto Tensor
  broadcast.py   unbroadcast(): the adjoint of NumPy broadcasting
  viz.py         records a sweep; renders it as text, Graphviz, or a page
  render.py      the HTML and SVG the visualiser emits
  __main__.py    `python -m mini_autograd`
examples/
  visualize_graph.py   record and draw one backward pass
  train_mlp.py         train a network on the five primitives, and plot it
tests/                 see the table above
index.html             the written walkthrough (GitHub Pages entry point)
site/                  the generated pages
notes/bugs.md          log of defects encountered during development
```

Two separations are deliberate. `tensor.py` knows how to traverse a graph and
nothing about calculus; `ops.py` knows the derivative of each primitive and
nothing about traversal. And the visualiser depends on the engine while the
engine does not depend on the visualiser, in either direction of import.

## Implementation notes

**Topological sort.** `Tensor._toposort` performs an iterative post-order
depth-first search over `_prev`, pushing `(node, expanded)` pairs so that a node
is emitted only after all of its inputs. The iterative formulation is not
stylistic: a chain of a few thousand operations is ordinary in a deep network
and would exceed CPython's default recursion limit. `test_graph.py` exercises a
5000-operation chain for exactly this reason.

**Accumulation.** Reverse-mode differentiation sums contributions along every
path from an input to the output. A node with several consumers — the diamond
`y = x*x + x` is the smallest example — receives one gradient per outgoing edge,
and `accumulate_grad` adds rather than assigns. Assignment would silently return
the contribution of whichever path happened to run last. In the visualiser this
shows up as two arrows landing on the same tensor in a single step.

**Broadcasting.** Forward broadcasting is supplied by NumPy at no cost; its
adjoint is not. If the forward pass replicated one value into *n* slots, the
backward pass must sum *n* incoming gradients back into that slot. `unbroadcast`
inverts the two steps NumPy takes, in order: it sums away the leading axes
introduced by rank expansion, then sums with `keepdims=True` over the axes
stretched from extent one. A final `reshape` raises on any shape neither step
accounts for, rather than returning a plausible but wrong array.

**ReLU at zero.** The derivative of `max(x, 0)` does not exist at the origin.
The implementation returns zero, following PyTorch; a finite-difference check
must therefore avoid probing that point.

**Rendering.** Both generated pages are single files with no external
JavaScript, so they open from the filesystem and keep working offline. The
charts are emitted as hand-written SVG rather than produced by a plotting
library, which keeps the visualiser's only dependency the same as the engine's.

## Limitations

The operation set is `add`, `mul`, `matmul`, `sum`, and `relu`, with reflected
operators bound for the first three. There is no division, exponentiation,
indexing, reshaping, or reduction other than `sum`; adding one follows the
three-step pattern documented at the top of `ops.py`. Subtraction is written as
`a + (-1.0) * b` throughout.

There is no optimizer and no module system. `examples/train_mlp.py` writes both
by hand — four parameters in a dict, momentum and a cosine step size in the
training loop — which is verbose but keeps visible exactly what is being
updated. Plain undecayed descent is not a stylistic omission there: at a step
size large enough to converge in reasonable time it diverges to NaN on that
problem, and at a step size small enough to stay stable it stalls near MSE 0.05.

The visualiser lays nodes out in columns by depth and does not route edges
around them, so a wide graph will cross some lines. Tensors larger than 144
entries are summarised by their statistics rather than displayed cell by cell.
`test_graph.py` still refers to a `tests/test_memory.py` that has not been
written.
