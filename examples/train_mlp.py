"""Fit a one-dimensional function with an MLP built on mini_autograd.

Run:  python examples/train_mlp.py

The point is not the model. The point is that the five primitives in ops.py are
enough to train something, and that the training can be watched: the report
written to site/training.html plots the loss, the gradient norm of every
parameter, and the shape of the function the network represents at a series of
checkpoints, so the fit can be seen forming out of ReLU kinks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import _path  # noqa: F401  (sys.path shim; see examples/_path.py)
from mini_autograd import Tensor
from mini_autograd.render import line_chart, training_page

ROOT = Path(__file__).resolve().parent.parent

HIDDEN = 48
STEPS = 6000
LR = 0.005
MOMENTUM = 0.9
CHECKPOINTS = (0, 40, 150, 600, 2000, STEPS)

FW, BW, OK, MUTED = "#1F6B8F", "#AE3346", "#2E6B4F", "#77848D"


def target(x: np.ndarray) -> np.ndarray:
    """The function to learn: smooth, non-monotonic, not fittable by a line."""
    return np.sin(3.0 * x) + 0.3 * x


class MLP:
    """x -> W1 -> relu -> W2. Written out rather than wrapped in a Module class,
    because the repository has no module system and pretending otherwise would
    hide what the optimizer is actually touching."""

    def __init__(self, hidden: int, rng: np.random.Generator,
                 span: float = 3.0) -> None:
        # He initialisation: variance 2/fan_in keeps the ReLU activations from
        # collapsing or exploding as the signal moves through the layer.
        w1 = rng.normal(size=(1, hidden)) * np.sqrt(2.0)

        # The first-layer bias is *not* zeroed. Each hidden unit contributes one
        # kink, located where w1*x + b1 crosses zero; with b1 = 0 every kink
        # sits at x = 0 and the network is a two-piece line no matter how wide
        # it is. Scattering the knots across the input range gives the fit
        # somewhere to start. Leaving this at zeros plateaus at MSE ~ 0.47.
        knots = rng.uniform(-span, span, size=hidden)
        self.W1 = Tensor(w1, requires_grad=True)
        self.b1 = Tensor(-w1.ravel() * knots, requires_grad=True)
        self.W2 = Tensor(rng.normal(size=(hidden, 1)) * np.sqrt(2.0 / hidden),
                         requires_grad=True)
        self.b2 = Tensor(np.zeros(1), requires_grad=True)

    @property
    def params(self) -> dict[str, Tensor]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def __call__(self, x: Tensor) -> Tensor:
        return (x @ self.W1 + self.b1).relu() @ self.W2 + self.b2


def mse(pred: Tensor, y: Tensor) -> Tensor:
    """Mean squared error, assembled from add/mul/sum alone.

    Subtraction is not a primitive here, so the difference is written as
    `pred + (-1)*y`; the gradient is identical, and adding a `sub` op would be
    four lines in ops.py following the same pattern.
    """
    err = pred + (-1.0) * y
    return (err * err).sum() * (1.0 / y.size)


def main() -> None:
    rng = np.random.default_rng(7)
    x_np = np.linspace(-3.0, 3.0, 128).reshape(-1, 1)
    y_np = target(x_np)

    x, y = Tensor(x_np), Tensor(y_np)
    model = MLP(HIDDEN, rng)
    velocity = {k: np.zeros_like(p.data) for k, p in model.params.items()}

    losses: list[tuple[float, float]] = []
    grad_norms: dict[str, list[tuple[float, float]]] = {k: [] for k in model.params}
    snapshots: list[tuple[int, np.ndarray]] = []

    for step in range(STEPS + 1):
        pred = model(x)
        loss = mse(pred, y)

        if step in CHECKPOINTS:
            snapshots.append((step, pred.data.ravel().copy()))

        loss.backward()          # frees the graph; nothing is retained
        losses.append((step, float(loss.data)))
        for name, p in model.params.items():
            grad_norms[name].append((step, float(np.sqrt((p.grad ** 2).sum()))))

        # The optimizer, written by hand: there is no optimizer module in this
        # repository. Momentum and a cosine step-size decay are here for a
        # concrete reason -- undecayed gradient descent at a step size large
        # enough to converge in reasonable time diverges to NaN on this
        # problem, and at a step size small enough to stay stable it stalls
        # around MSE 0.05.
        lr = LR * 0.5 * (1.0 + np.cos(np.pi * step / STEPS))
        for name, p in model.params.items():
            velocity[name] = MOMENTUM * velocity[name] + p.grad
            p.data -= lr * velocity[name]
            p.zero_grad()

        if step % 1000 == 0:
            print(f"step {step:>5}  loss {float(loss.data):.6f}")

    write_report(x_np, y_np, losses, grad_norms, snapshots, model)


def thin(points: list[tuple[float, float]], keep: int = 500) -> list:
    """Downsample a per-step series for plotting.

    6001 points per curve is far more than a 470-pixel-wide chart can show, and
    writing them all out quadruples the size of the page for no visible gain.
    The last point is always kept so the final value on the chart is the real
    one.
    """
    if len(points) <= keep:
        return points
    stride = len(points) // keep
    thinned = points[::stride]
    if thinned[-1] != points[-1]:
        thinned.append(points[-1])
    return thinned


def write_report(x_np, y_np, losses, grad_norms, snapshots, model) -> None:
    xs = x_np.ravel()

    fit_series = [{
        "label": "target", "color": MUTED, "width": 2.6, "opacity": 0.9,
        "points": list(zip(xs, y_np.ravel())),
    }]
    for i, (step, pred) in enumerate(snapshots):
        fade = 0.22 + 0.78 * (i / max(len(snapshots) - 1, 1))
        fit_series.append({
            "label": f"step {step}", "color": FW, "opacity": fade,
            "width": 1.4 + 1.4 * fade, "points": list(zip(xs, pred)),
        })

    grad_colours = {"W1": FW, "b1": BW, "W2": OK, "b2": MUTED}
    grad_series = [
        {"label": name, "color": grad_colours[name],
         "points": thin([p for p in pts if p[1] > 0])}
        for name, pts in grad_norms.items()
    ]

    figures = [
        {
            "title": "Loss",
            "subtitle": "mean squared error, log scale",
            "svg": line_chart([{"label": "loss", "color": BW,
                                "points": thin(losses)}],
                              log_y=True, x_label="step", y_label="MSE"),
            "keys": [{"label": "training loss", "color": BW}],
        },
        {
            "title": "Gradient norms",
            "subtitle": "‖dL/dp‖ per parameter, log scale",
            "svg": line_chart(grad_series, log_y=True, x_label="step",
                              y_label="norm"),
            "keys": [{"label": k, "color": grad_colours[k]} for k in grad_norms],
        },
        {
            "title": "What the network represents",
            "subtitle": "predictions at successive checkpoints, darkening with training",
            "svg": line_chart(fit_series, x_label="x", y_label="y"),
            "keys": ([{"label": "target", "color": MUTED}] +
                     [{"label": f"step {s}", "color": FW}
                      for s, _ in snapshots[-3:]]),
        },
    ]

    summary = [["parameter", "shape", "‖p‖", "final ‖dL/dp‖"]]
    for name, p in model.params.items():
        summary.append([
            name, str(p.shape),
            f"{np.sqrt((p.data ** 2).sum()):.4f}",
            f"{grad_norms[name][-1][1]:.3e}",
        ])

    out = ROOT / "site" / "training.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        training_page(
            "Training on five primitives",
            f"A {HIDDEN}-unit ReLU network fitting sin(3x) + 0.3x, trained for "
            f"{STEPS} steps of gradient descent with momentum {MOMENTUM} and a "
            f"cosine step size decaying from {LR}. Every "
            "derivative in this run came from mini_autograd; nothing here is "
            "PyTorch.",
            figures, summary,
        ),
        encoding="utf-8",
    )
    print(f"\nfinal loss {losses[-1][1]:.6f}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
