"""Direct tests for unbroadcast(), plus end-to-end broadcasting through ops.

unbroadcast is tested two ways on purpose:
  * against a hand-written reference (sum over the axes that were replicated),
  * against a finite-difference gradient of a real broadcasting expression,
so a bug in my mental model of NumPy broadcasting cannot pass both.
"""

import numpy as np
import pytest

from mini_autograd import Tensor, unbroadcast

CASES = [
    # (operand shape, broadcast-result shape)
    ((), (3, 4)),            # scalar <-> matrix
    ((1,), (5,)),            # size-1 stretch, same rank
    ((3, 1), (3, 4)),        # column vector <-> matrix
    ((1, 4), (3, 4)),        # row vector <-> matrix
    ((4,), (2, 3, 4)),       # rank expansion by 2
    ((3, 4), (2, 3, 4)),     # rank expansion by 1
    ((2, 1, 4), (2, 3, 4)),  # stretch a middle axis
    ((1, 1, 1), (2, 3, 4)),  # stretch every axis
    ((2, 3, 4), (2, 3, 4)),  # identity: must be a no-op
    ((1,), (2, 3, 4)),       # rank expansion *and* stretch
]


@pytest.mark.parametrize("small,big", CASES)
def test_unbroadcast_shape(small, big):
    g = np.random.randn(*big)
    assert unbroadcast(g, small).shape == small


@pytest.mark.parametrize("small,big", CASES)
def test_unbroadcast_values(small, big):
    """Reference: broadcasting replicates, so the adjoint sums over exactly the
    replicated axes -- the leading new axes plus any axis that was size 1."""
    g = np.random.randn(*big)
    extra = len(big) - len(small)
    padded = (1,) * extra + small
    axes = tuple(i for i, d in enumerate(padded) if d == 1 and big[i] != 1)
    expected = g.sum(axis=axes, keepdims=True).reshape(small) if axes else g.reshape(small)
    # rtol, not exact: unbroadcast sums the replicated axes in two passes
    # (rank expansion, then size-1 stretch) while the reference does it in
    # one, and float addition is not associative.
    np.testing.assert_allclose(unbroadcast(g, small), expected, rtol=1e-12)


def test_unbroadcast_is_a_view_or_copy_but_never_mutates_input():
    g = np.ones((3, 4))
    unbroadcast(g, (3, 1))
    np.testing.assert_array_equal(g, np.ones((3, 4)))


def test_unbroadcast_rejects_rank_increase():
    with pytest.raises(ValueError):
        unbroadcast(np.ones((3,)), (2, 3))


def test_unbroadcast_rejects_incompatible_shape():
    with pytest.raises(ValueError):
        unbroadcast(np.ones((3, 4)), (3, 5))


# --------------------------------------------------------------------------
# The same cases, but exercised through real ops and checked numerically.
# --------------------------------------------------------------------------
def _numeric_grad(f, x, h=1e-6):
    """Central difference: (f(x+h) - f(x-h)) / 2h, elementwise."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        orig = x[i]
        x[i] = orig + h
        plus = f(x)
        x[i] = orig - h
        minus = f(x)
        x[i] = orig
        grad[i] = (plus - minus) / (2 * h)
        it.iternext()
    return grad


@pytest.mark.parametrize("small,big", CASES)
@pytest.mark.parametrize("op", ["add", "mul"])
def test_broadcasting_gradients_match_finite_differences(small, big, op):
    a_np = np.random.randn(*small) if small else np.array(0.7)
    b_np = np.random.randn(*big)

    def forward(a_val, b_val):
        a, b = Tensor(a_val, requires_grad=True), Tensor(b_val, requires_grad=True)
        out = (a + b) if op == "add" else (a * b)
        return out, a, b

    out, a, b = forward(a_np.copy(), b_np.copy())
    loss = out.sum()
    loss.backward()

    f_a = lambda v: (v + b_np).sum() if op == "add" else (v * b_np).sum()
    f_b = lambda v: (a_np + v).sum() if op == "add" else (a_np * v).sum()

    np.testing.assert_allclose(a.grad, _numeric_grad(f_a, a_np.copy()), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(b.grad, _numeric_grad(f_b, b_np.copy()), rtol=1e-6, atol=1e-6)


def test_matmul_broadcasts_batch_dims():
    """(2,3,4) @ (4,5): the right operand is shared across the batch, so its
    gradient must be the sum over the batch, not just the last slice."""
    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    b = Tensor(np.random.randn(4, 5), requires_grad=True)
    (a @ b).sum().backward()

    assert a.grad.shape == (2, 3, 4)
    assert b.grad.shape == (4, 5)
    expected_b = np.einsum("bij,bik->jk", a.data, np.ones((2, 3, 5)))
    np.testing.assert_allclose(b.grad, expected_b, rtol=1e-12)
