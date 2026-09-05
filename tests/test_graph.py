"""Graph mechanics: topological order, gradient accumulation, teardown."""

import gc
import weakref

import numpy as np
import pytest

from mini_autograd import Tensor


# ------------------------------------------------------- topological order
def test_toposort_puts_inputs_before_consumers():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = a * b
    d = c + a
    e = (d * c).sum()

    topo = e._toposort()
    position = {id(t): i for i, t in enumerate(topo)}

    assert len(topo) == len({id(t) for t in topo}), "a node was emitted twice"
    for node in topo:
        for parent in node._prev:
            assert position[id(parent)] < position[id(node)], (
                f"{parent._op or 'leaf'} must come before {node._op}"
            )
    assert topo[-1] is e


def test_toposort_visits_a_shared_node_once():
    x = Tensor(np.ones(3), requires_grad=True)
    y = x * x + x + x  # x has three consumers
    topo = y.sum()._toposort()
    assert sum(1 for t in topo if t is x) == 1


def test_toposort_is_iterative_and_survives_a_deep_chain():
    """A 5000-op chain would blow Python's recursion limit if the sort recursed."""
    x = Tensor(1.0, requires_grad=True)
    t = x
    for _ in range(5000):
        t = t + 1.0
    t.backward()
    np.testing.assert_allclose(t.data, 5001.0)
    np.testing.assert_allclose(x.grad, 1.0)


# ------------------------------------------------------ diamond / accumulate
def test_diamond_graph():
    """y = x*x + x  =>  dy/dx = 2x + 1.

    This is the canonical accumulation test: x reaches y by two paths. If
    _backward assigned instead of accumulating, we would get 2x or 1, never both.
    """
    x_np = np.array([-3.0, 0.5, 2.0, 7.0])
    x = Tensor(x_np, requires_grad=True)
    y = (x * x + x).sum()
    y.backward()

    np.testing.assert_allclose(y.data, (x_np * x_np + x_np).sum())
    np.testing.assert_allclose(x.grad, 2 * x_np + 1)


def test_wide_fan_out_accumulates_every_path():
    x = Tensor(np.array([1.5, -2.0]), requires_grad=True)
    y = (x + x + x + x * x).sum()  # d/dx = 3 + 2x
    y.backward()
    np.testing.assert_allclose(x.grad, 3 + 2 * x.data)


def test_shared_subgraph_accumulates_once_per_edge():
    """h is consumed twice; its gradient must be the sum of both consumers."""
    x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    h = x * 2.0
    out = (h * h + h).sum()  # d/dh = 2h + 1, dh/dx = 2  =>  dx = 2*(2*(2x)+1)
    out.backward()
    np.testing.assert_allclose(x.grad, 2 * (2 * (2 * x.data) + 1))


def test_grad_accumulates_across_two_backward_calls():
    """Like PyTorch, .grad is not zeroed for you -- that is the optimizer's job."""
    x = Tensor(np.array([3.0]), requires_grad=True)
    (x * x).sum().backward()
    first = x.grad.copy()
    (x * x).sum().backward()
    np.testing.assert_allclose(x.grad, 2 * first)
    x.zero_grad()
    assert x.grad is None


def test_non_requires_grad_inputs_get_no_grad_and_no_edges():
    a = Tensor(np.ones(3), requires_grad=False)
    b = Tensor(np.ones(3), requires_grad=False)
    out = a * b
    assert out.requires_grad is False
    assert out._prev == ()
    assert out._backward is None
    with pytest.raises(RuntimeError):
        out.sum().backward()


def test_backward_on_non_scalar_requires_explicit_seed():
    x = Tensor(np.ones((2, 2)), requires_grad=True)
    y = x * 3.0
    with pytest.raises(RuntimeError, match="non-scalar"):
        y.backward()
    y.backward(np.ones((2, 2)))
    np.testing.assert_allclose(x.grad, 3.0)


# ------------------------------------------------------------- graph teardown
def test_backward_releases_the_graph():
    x = Tensor(np.ones(4), requires_grad=True)
    hidden = x * 2.0
    loss = (hidden * hidden).sum()

    ref = weakref.ref(hidden)
    del hidden

    assert ref() is not None, "graph should be alive before backward()"
    loss.backward()
    assert loss._prev == () and loss._backward is None

    del loss
    gc.collect()
    assert ref() is None, "intermediate tensor survived backward() -- graph leak"

    np.testing.assert_allclose(x.grad, 8.0)  # d/dx (2x)^2 = 8x


def test_retain_graph_keeps_the_edges():
    x = Tensor(np.ones(4), requires_grad=True)
    loss = (x * x).sum()
    loss.backward(retain_graph=True)
    assert loss._prev != () and loss._backward is not None


def test_live_tensor_count_is_flat_over_many_steps():
    """Short version of the 5000-step leak test (see tests/test_memory.py)."""
    w = Tensor(np.ones((4, 4)), requires_grad=True)

    def step():
        x = Tensor(np.ones((8, 4)))
        ((x @ w).relu().sum()).backward()
        w.zero_grad()

    for _ in range(20):
        step()
    gc.collect()
    before = sum(1 for o in gc.get_objects() if isinstance(o, Tensor))
    for _ in range(200):
        step()
    gc.collect()
    after = sum(1 for o in gc.get_objects() if isinstance(o, Tensor))
    assert after <= before, f"live tensors grew {before} -> {after}"
