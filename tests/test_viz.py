"""Tests for the visualiser.

Two things are being checked here, and they are different in kind. The first is
that the recorded trace agrees with the engine: a picture of the backward pass
is worthless if it is a picture of different arithmetic. The second is that the
generated page is structurally sound -- the embedded data parses, the script
addresses elements that exist, no template placeholder survived. That is the
part a screenshot would normally catch, and these assertions stand in for one.
"""

import json
import re

import numpy as np
import pytest

from mini_autograd import Tensor
from mini_autograd.render import line_chart, training_page
from mini_autograd.viz import record_backward


def small_graph():
    rng = np.random.default_rng(3)
    x = Tensor(rng.normal(size=(4, 3)))
    W = Tensor(rng.normal(size=(3, 2)), requires_grad=True)
    b = Tensor(np.zeros(2), requires_grad=True)
    loss = ((x @ W + b).relu() * 2.0).sum()
    return loss, {"x": x, "W": W, "b": b, "loss": loss}


# ------------------------------------------------- agreement with the engine
def test_recorded_gradients_match_backward():
    loss, names = small_graph()
    loss.backward(retain_graph=True)
    reference = {k: t.grad.copy() for k, t in names.items() if t.requires_grad}

    record_backward(loss, names)
    for k, expected in reference.items():
        np.testing.assert_allclose(names[k].grad, expected)


def test_recording_clears_stale_gradients_first():
    """Two recordings in a row must give the same answer, not twice the answer.
    `Tensor.backward` accumulates by design; the recorder resets so that what is
    drawn is one sweep, not a sum of every sweep so far."""
    loss, names = small_graph()
    first = record_backward(loss, names).nodes
    second = record_backward(loss, names).nodes
    for a, b in zip(first, second):
        assert (a.grad_stats is None) == (b.grad_stats is None)
        if a.grad_stats:
            assert a.grad_stats["norm"] == pytest.approx(b.grad_stats["norm"])


def test_recording_keeps_the_graph_alive():
    loss, names = small_graph()
    record_backward(loss, names)
    assert loss._prev != () and loss._backward is not None
    loss.backward()  # still differentiable afterwards


def test_steps_respect_topological_order():
    """A node may only fire after every one of its consumers has fired: that is
    what makes its gradient final when it is used."""
    loss, names = small_graph()
    trace = record_backward(loss, names)
    fired_at = {}
    for s in trace.steps:
        fired_at.setdefault(s.node, s.index)
    for s in trace.steps:
        for u in s.updates:
            if u["node"] in fired_at and u["node"] != s.node:
                assert fired_at[u["node"]] > s.index


def test_diamond_shows_two_pushes_into_the_same_tensor():
    """y = x*x is one step writing into x twice -- the accumulation the engine
    relies on, made visible."""
    x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    trace = record_backward((x * x).sum(), {"x": x})
    into_x = [u for s in trace.steps for u in s.updates
              if trace.nodes[u["node"]].label == "x"]
    assert len(into_x) == 2
    np.testing.assert_allclose(x.grad, 2 * x.data)


def test_non_scalar_root_requires_a_seed():
    x = Tensor(np.ones((2, 2)), requires_grad=True)
    with pytest.raises(RuntimeError, match="non-scalar"):
        record_backward(x * 3.0)


def test_root_without_grad_is_rejected():
    with pytest.raises(RuntimeError):
        record_backward(Tensor(np.ones(3)) * 2.0)


# --------------------------------------------------------------- text / dot
def test_text_report_names_every_tensor():
    loss, names = small_graph()
    text = record_backward(loss, names).to_text()
    for label in ("x", "W", "b", "loss"):
        assert re.search(rf"\b{label}\b", text)
    assert "backward sweep:" in text


def test_dot_declares_every_node_and_edge():
    loss, names = small_graph()
    trace = record_backward(loss, names)
    dot = trace.to_dot()
    assert dot.startswith("digraph autograd {") and dot.rstrip().endswith("}")
    assert dot.count("[color=") == len(trace.nodes)
    edges = sum(len(n.parents) for n in trace.nodes)
    assert dot.count(" -> ") == edges


# --------------------------------------------------------------- the page
def test_json_payload_is_valid_and_complete():
    loss, names = small_graph()
    trace = record_backward(loss, names)
    data = json.loads(trace.to_json())
    assert len(data["nodes"]) == len(trace.nodes)
    assert data["root"] == trace.root
    assert all("x" in n and "y" in n for n in data["nodes"])


def test_layout_places_inputs_left_of_consumers():
    loss, names = small_graph()
    trace = record_backward(loss, names)
    for n in trace.nodes:
        for p in n.parents:
            assert trace.nodes[p].x < n.x


def test_graph_page_is_structurally_sound(tmp_path):
    loss, names = small_graph()
    trace = record_backward(loss, names)
    path = trace.to_html(tmp_path / "graph.html", "Title", "Subtitle")
    html = path.read_text(encoding="utf-8")

    assert html.startswith("<!doctype html>")
    assert "__DATA__" not in html and "__NW__" not in html
    assert html.count("<svg") == html.count("</svg>")

    payload = re.search(r"const D = (\{.*?\});\n", html, re.S)
    assert payload, "embedded trace not found"
    json.loads(payload.group(1))

    # Every element the script reaches for must exist in the markup.
    for element_id in set(re.findall(r"getElementById\('([^']+)'\)", html)):
        assert f'id="{element_id}"' in html, f"script targets missing #{element_id}"


def test_large_tensors_are_summarised_not_dumped():
    """A 200-element tensor exceeds the display threshold: statistics survive,
    the cell-by-cell preview does not."""
    big = Tensor(np.ones((20, 10)), requires_grad=True)
    trace = record_backward(big.sum(), {"big": big})
    node = next(n for n in trace.nodes if n.label == "big")
    assert node.value_cells is None
    assert node.value_stats["norm"] == pytest.approx(np.sqrt(200))


# --------------------------------------------------------------- the charts
def test_line_chart_emits_one_path_per_series():
    svg = line_chart(
        [{"label": "a", "color": "#000", "points": [(0, 1), (1, 2), (2, 3)]},
         {"label": "b", "color": "#111", "points": [(0, 3), (1, 1), (2, 2)]}],
        x_label="step", y_label="value",
    )
    assert svg.count("<path") == 2
    assert svg.startswith("<svg") and svg.endswith("</svg>")


def test_line_chart_handles_a_flat_series():
    """A constant series gives a zero-height range; the scale must not divide
    by zero."""
    svg = line_chart([{"label": "flat", "color": "#000",
                       "points": [(0, 5.0), (1, 5.0)]}])
    assert "NaN" not in svg


def test_log_scale_tolerates_zero():
    svg = line_chart([{"label": "z", "color": "#000",
                       "points": [(0, 0.0), (1, 1.0)]}], log_y=True)
    assert "NaN" not in svg and "Infinity" not in svg


def test_training_page_renders_figures_and_summary():
    svg = line_chart([{"label": "l", "color": "#000", "points": [(0, 1), (1, 2)]}])
    html = training_page(
        "Report", "subtitle",
        [{"title": "Loss", "subtitle": "mse", "svg": svg,
          "keys": [{"label": "loss", "color": "#000"}]}],
        [["parameter", "shape"], ["W", "(3, 2)"]],
    )
    assert "<figcaption>Loss</figcaption>" in html
    assert html.count("<figure>") == html.count("</figure>") == 1
    assert "<td>W</td>" in html and "<th>parameter</th>" in html
