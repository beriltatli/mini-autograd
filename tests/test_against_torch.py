"""Differential testing against PyTorch.

The finite-difference checks elsewhere confirm that the gradients are close to
the derivative of the forward function. This file confirms something narrower
and stricter: that they are the same numbers a mature implementation produces,
including in the corners where a convention has to be chosen -- the subgradient
of ReLU at zero, the reduction of a broadcast bias, the batched matmul adjoint.

PyTorch is a development dependency only. If it is absent the file is skipped;
nothing under mini_autograd/ imports it.
"""

import numpy as np
import pytest

from mini_autograd import Tensor

torch = pytest.importorskip("torch")

TOL = dict(rtol=1e-10, atol=1e-12)


def compare(build_ours, build_theirs, inputs):
    """Run the same expression through both engines and compare every gradient.

    `inputs` maps a name to a NumPy array; both builders receive a dict of their
    own leaf tensors under those names.
    """
    ours = {k: Tensor(v.copy(), requires_grad=True) for k, v in inputs.items()}
    build_ours(ours).backward()

    theirs = {
        k: torch.tensor(v.copy(), dtype=torch.float64, requires_grad=True)
        for k, v in inputs.items()
    }
    build_theirs(theirs).backward()

    for name in inputs:
        np.testing.assert_allclose(
            ours[name].grad, theirs[name].grad.numpy(),
            err_msg=f"gradient of {name} disagrees with torch", **TOL
        )


def test_linear_layer_with_broadcast_bias():
    rng = np.random.default_rng(0)
    inputs = {
        "x": rng.normal(size=(6, 4)),
        "W": rng.normal(size=(4, 3)),
        "b": rng.normal(size=(3,)),
    }
    compare(
        lambda t: ((t["x"] @ t["W"] + t["b"]).relu()).sum(),
        lambda t: ((t["x"] @ t["W"] + t["b"]).relu()).sum(),
        inputs,
    )


def test_two_layer_network_with_squared_error():
    rng = np.random.default_rng(1)
    inputs = {
        "x": rng.normal(size=(8, 3)),
        "W1": rng.normal(size=(3, 5)),
        "b1": rng.normal(size=(5,)),
        "W2": rng.normal(size=(5, 2)),
        "b2": rng.normal(size=(2,)),
        "y": rng.normal(size=(8, 2)),
    }

    def forward(t):
        h = (t["x"] @ t["W1"] + t["b1"]).relu()
        err = h @ t["W2"] + t["b2"] + (-1.0) * t["y"]
        return (err * err).sum() * 0.0625

    compare(forward, forward, inputs)


def test_shared_subgraph_accumulation():
    """h feeds three consumers; both engines must sum all three paths."""
    rng = np.random.default_rng(2)
    inputs = {"x": rng.normal(size=(4, 4)), "W": rng.normal(size=(4, 4))}

    def forward(t):
        h = (t["x"] @ t["W"]).relu()
        return (h * h + h * 3.0 + h).sum()

    compare(forward, forward, inputs)


def test_batched_matmul_shares_the_right_operand():
    rng = np.random.default_rng(3)
    inputs = {"a": rng.normal(size=(5, 3, 4)), "b": rng.normal(size=(4, 2))}
    compare(
        lambda t: (t["a"] @ t["b"]).sum(),
        lambda t: (t["a"] @ t["b"]).sum(),
        inputs,
    )


@pytest.mark.parametrize("axis,keepdims", [
    (0, False), (1, False), (0, True), (1, True),
    ((0, 2), False), ((0, 2), True), (-1, False),
])
def test_sum_over_axes(axis, keepdims):
    rng = np.random.default_rng(4)
    inputs = {"x": rng.normal(size=(3, 4, 5)), "w": rng.normal(size=(3, 4, 5))}

    def ours(t):
        return (t["x"] * t["w"]).sum(axis=axis, keepdims=keepdims).sum()

    def theirs(t):
        return (t["x"] * t["w"]).sum(dim=axis, keepdim=keepdims).sum()

    compare(ours, theirs, inputs)


def test_relu_subgradient_at_zero_matches_torch():
    """max(x, 0) has no derivative at the origin. Both engines choose 0; this
    test exists to notice if either ever stops choosing it."""
    inputs = {"x": np.array([-1.0, 0.0, 1.0])}
    compare(lambda t: t["x"].relu().sum(),
            lambda t: t["x"].relu().sum(), inputs)
    x = Tensor(np.array([0.0]), requires_grad=True)
    x.relu().sum().backward()
    np.testing.assert_array_equal(x.grad, np.zeros(1))


@pytest.mark.parametrize("small,big", [
    ((), (3, 4)), ((1,), (5,)), ((3, 1), (3, 4)), ((1, 4), (3, 4)),
    ((4,), (2, 3, 4)), ((2, 1, 4), (2, 3, 4)), ((1,), (2, 3, 4)),
])
def test_broadcast_reduction_matches_torch(small, big):
    rng = np.random.default_rng(5)
    inputs = {"a": rng.normal(size=small) if small else np.array(0.7),
              "b": rng.normal(size=big)}
    compare(lambda t: (t["a"] * t["b"] + t["a"]).sum(),
            lambda t: (t["a"] * t["b"] + t["a"]).sum(), inputs)
