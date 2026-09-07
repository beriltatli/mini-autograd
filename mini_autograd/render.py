"""HTML rendering for the visualiser.

Two pages are produced from here: the interactive graph (`graph_page`) and the
training report (`training_page`). Both are single files with no external
JavaScript, so they open from the filesystem and keep working offline; the only
network request is the webfont link, which degrades to the fallback stack.

The palette matches `index.html` on purpose. Throughout the project one
colour means "forward pass, a value" and another means "backward pass, a
gradient", and the pages are only readable if that stays consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .viz import Trace

NODE_W, NODE_H = 188.0, 66.0

_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&"
    "family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600&"
    'family=JetBrains+Mono:wght@400;700&display=swap">'
)

_BASE_CSS = """
:root{
  --paper:#F1F3F5; --surface:#FFFFFF; --sunk:#E7EBEE;
  --ink:#12171C; --ink-2:#4C5862; --ink-3:#77848D;
  --rule:#D3DAE0; --rule-soft:#E3E8EC;
  --fw:#1F6B8F; --bw:#AE3346; --ok:#2E6B4F;
  --fw-wash:#E4EEF3; --bw-wash:#F7E7E9;
  --display:"Bricolage Grotesque","Helvetica Neue",Arial,sans-serif;
  --body:"Newsreader",Georgia,"Times New Roman",serif;
  --mono:"JetBrains Mono",ui-monospace,"SF Mono",Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#101519; --surface:#171E24; --sunk:#0B1013;
  --ink:#E8EDF1; --ink-2:#A6B3BC; --ink-3:#77858F;
  --rule:#2A343C; --rule-soft:#212A31;
  --fw:#68B4D6; --bw:#E8848F; --ok:#79C4A0;
  --fw-wash:#16303C; --bw-wash:#392026;
}}
:root[data-theme="dark"]{
  --paper:#101519; --surface:#171E24; --sunk:#0B1013;
  --ink:#E8EDF1; --ink-2:#A6B3BC; --ink-3:#77858F;
  --rule:#2A343C; --rule-soft:#212A31;
  --fw:#68B4D6; --bw:#E8848F; --ok:#79C4A0;
  --fw-wash:#16303C; --bw-wash:#392026;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:var(--body);font-size:16px;line-height:1.6;
  -webkit-font-smoothing:antialiased}
header{padding:38px 26px 20px;max-width:1180px;margin:0 auto}
h1{font-family:var(--display);font-weight:800;font-size:2rem;margin:0 0 .3em;
  letter-spacing:-.02em}
header p{margin:0;color:var(--ink-2);max-width:62ch}
main{max-width:1180px;margin:0 auto;padding:0 26px 72px}
.mono{font-family:var(--mono);font-size:.8rem}
.legend{display:flex;gap:20px;flex-wrap:wrap;align-items:center;
  margin:18px 0 0;font-family:var(--mono);font-size:.74rem;color:var(--ink-2)}
.swatch{display:inline-block;width:11px;height:11px;border-radius:2px;
  margin-right:6px;vertical-align:-1px}
"""


def _document(title: str, body: str, css: str = "", script: str = "") -> str:
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n{_FONTS}\n"
        f"<style>{_BASE_CSS}{css}</style>\n</head>\n<body>\n{body}\n"
        f"<script>{script}</script>\n</body>\n</html>\n"
    )


# --------------------------------------------------------------- graph page
_GRAPH_CSS = """
.board{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
  overflow:auto;position:relative}
svg{display:block}
.edge{fill:none;stroke:var(--rule);stroke-width:1.4}
.edge.fwd{stroke:var(--fw);stroke-width:1.8;opacity:.55}
.edge.live{stroke:var(--bw);stroke-width:2.6;opacity:1}
.card{cursor:pointer}
.card rect.bg{fill:var(--surface);stroke:var(--rule);stroke-width:1.3;rx:7}
.card.leaf rect.bg{fill:var(--sunk)}
.card.param rect.bg{stroke:var(--fw);stroke-width:1.8}
.card.firing rect.bg{stroke:var(--bw);stroke-width:2.6}
.card.updated rect.bg{fill:var(--bw-wash)}
.card.selected rect.bg{stroke:var(--ink);stroke-width:2.4}
.card text{font-family:var(--mono);font-size:11px;fill:var(--ink)}
.card text.sub{font-size:9.5px;fill:var(--ink-3)}
.card text.val{font-size:10px;fill:var(--fw)}
.card text.grad{font-size:10px;fill:var(--bw)}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;
  margin:20px 0 14px}
button{font-family:var(--mono);font-size:.76rem;padding:7px 13px;
  border:1px solid var(--rule);background:var(--surface);color:var(--ink);
  border-radius:6px;cursor:pointer}
button:hover{border-color:var(--fw)}
button[disabled]{opacity:.4;cursor:default}
input[type=range]{flex:1;min-width:200px;accent-color:var(--bw)}
.status{font-family:var(--mono);font-size:.78rem;color:var(--ink-2);
  min-height:1.5em;margin:0 0 8px}
.status b{color:var(--bw);font-weight:700}
.panel{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
  gap:22px;margin-top:26px}
.panel section{background:var(--surface);border:1px solid var(--rule);
  border-radius:10px;padding:16px 18px}
.panel h2{font-family:var(--display);font-size:1rem;margin:0 0 4px}
.panel .shape{font-family:var(--mono);font-size:.73rem;color:var(--ink-3);
  margin:0 0 12px}
table.m{border-collapse:collapse;font-family:var(--mono);font-size:.7rem}
table.m td{border:1px solid var(--rule-soft);padding:3px 7px;text-align:right;
  min-width:52px}
.note{color:var(--ink-3);font-family:var(--mono);font-size:.72rem}
"""

_GRAPH_JS = r"""
const D = __DATA__;
const NW = __NW__, NH = __NH__;
const svg = document.getElementById('canvas');
const NS = 'http://www.w3.org/2000/svg';
let step = -1, selected = D.root, timer = null;

function fmt(x){
  if (x === null || x === undefined) return '-';
  const a = Math.abs(x);
  if (a === 0) return '0';
  if (a < 1e-3 || a >= 1e4) return x.toExponential(1);
  return x.toFixed(a < 1 ? 3 : 2);
}

/* Gradient state at a step is a replay: each update carries the parent's full
   gradient after that push, so the state is the last update at or before k. */
function gradAt(k){
  const state = {};
  for (let i = 0; i <= k; i++)
    for (const u of D.steps[i].updates) state[u.node] = u;
  return state;
}

function el(tag, attrs, parent){
  const n = document.createElementNS(NS, tag);
  for (const key in attrs) n.setAttribute(key, attrs[key]);
  if (parent) parent.appendChild(n);
  return n;
}

function draw(){
  svg.textContent = '';
  const state = gradAt(step);
  const fired = step >= 0 ? D.steps[step].node : null;
  const touched = new Set(step >= 0 ? D.steps[step].updates.map(u => u.node) : []);

  const edges = el('g', {}, svg);
  for (const n of D.nodes) for (const p of n.parents){
    const a = D.nodes[p];
    const x1 = a.x + NW, y1 = a.y + NH/2, x2 = n.x, y2 = n.y + NH/2;
    const mid = (x1 + x2) / 2;
    let cls = 'edge';
    if (step < 0) cls += ' fwd';
    else if (n.idx === fired && touched.has(p)) cls += ' live';
    el('path', {class: cls,
      d: `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`}, edges);
  }

  for (const n of D.nodes){
    let cls = 'card';
    if (!n.op) cls += ' leaf';
    if (n.requiresGrad && !n.op) cls += ' param';
    if (n.idx === fired) cls += ' firing';
    if (touched.has(n.idx)) cls += ' updated';
    if (n.idx === selected) cls += ' selected';
    const g = el('g', {class: cls, transform: `translate(${n.x},${n.y})`}, svg);
    g.addEventListener('click', () => { selected = n.idx; draw(); });
    el('rect', {class: 'bg', width: NW, height: NH}, g);
    const t = el('text', {x: 12, y: 21}, g); t.textContent = n.label;
    const s = el('text', {class: 'sub', x: 12, y: 36}, g);
    s.textContent = (n.op || 'input') + '  (' + n.shape.join(', ') + ')';
    const v = el('text', {class: 'val', x: 12, y: 53}, g);
    v.textContent = '|x| ' + fmt(n.value ? n.value.norm : null);
    const gr = el('text', {class: 'grad', x: 100, y: 53}, g);
    const cur = state[n.idx];
    gr.textContent = cur ? '|g| ' + fmt(cur.grad.norm) : (n.requiresGrad ? '|g| -' : '');
  }

  document.getElementById('slider').value = step;
  document.getElementById('prev').disabled = step < 0;
  document.getElementById('next').disabled = step >= D.steps.length - 1;
  status(step);
  detail(state);
}

function status(k){
  const box = document.getElementById('status');
  if (k < 0){
    box.innerHTML = 'Forward pass complete. ' + D.nodes.length +
      ' tensors on the tape, no gradients yet. Press <b>step</b> to seed dL/dL.';
    return;
  }
  const s = D.steps[k];
  const src = D.nodes[s.node];
  if (s.kind === 'seed'){
    box.innerHTML = `step ${k}: seed <b>${src.label}</b>.grad = 1`;
    return;
  }
  const parts = s.updates.map(u =>
    `<b>${D.nodes[u.node].label}</b> (+${fmt(u.delta_norm)})`);
  box.innerHTML = `step ${k}: <b>${src.label}</b>.${src.op} pushes into ` +
    parts.join(', ');
}

function matrix(cells){
  if (!cells) return '<p class="note">too large to display; see the norm above</p>';
  let peak = 0;
  for (const row of cells) for (const v of row) peak = Math.max(peak, Math.abs(v));
  let html = '<table class="m">';
  for (const row of cells){
    html += '<tr>';
    for (const v of row){
      const a = peak ? Math.abs(v) / peak * 0.5 : 0;
      const c = v < 0 ? `rgba(174,51,70,${a})` : `rgba(31,107,143,${a})`;
      html += `<td style="background:${c}">${fmt(v)}</td>`;
    }
    html += '</tr>';
  }
  return html + '</table>';
}

function detail(state){
  const n = D.nodes[selected];
  document.getElementById('sel-name').textContent = n.label;
  document.getElementById('sel-shape').textContent =
    (n.op || 'input') + '  shape (' + n.shape.join(', ') + ')' +
    (n.requiresGrad ? '  requires_grad=True' : '  requires_grad=False');
  document.getElementById('sel-value').innerHTML = matrix(n.valueCells);
  const cur = state[selected];
  document.getElementById('sel-grad').innerHTML = cur
    ? matrix(cur.gradCells)
    : '<p class="note">no gradient at this step</p>';
}

function go(k){
  step = Math.max(-1, Math.min(D.steps.length - 1, k));
  draw();
}
document.getElementById('prev').onclick = () => go(step - 1);
document.getElementById('next').onclick = () => go(step + 1);
document.getElementById('reset').onclick = () => { stop(); go(-1); };
document.getElementById('slider').oninput = e => go(+e.target.value);
function stop(){
  clearInterval(timer); timer = null;
  document.getElementById('play').textContent = 'play';
}
document.getElementById('play').onclick = () => {
  if (timer){ stop(); return; }
  if (step >= D.steps.length - 1) go(-1);
  document.getElementById('play').textContent = 'pause';
  timer = setInterval(() => {
    if (step >= D.steps.length - 1) stop(); else go(step + 1);
  }, 750);
};
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') go(step + 1);
  if (e.key === 'ArrowLeft') go(step - 1);
});
draw();
"""


def graph_page(trace: "Trace", title: str, subtitle: str) -> str:
    """Render a recorded trace as an interactive page."""
    width = max(n.x for n in trace.nodes) + NODE_W + 28
    height = max(n.y for n in trace.nodes) + NODE_H + 28
    body = f"""
<header>
  <h1>{title}</h1>
  <p>{subtitle}</p>
  <div class="legend">
    <span><i class="swatch" style="background:var(--fw)"></i>forward value</span>
    <span><i class="swatch" style="background:var(--bw)"></i>gradient</span>
    <span>{len(trace.nodes)} tensors</span>
    <span>{len(trace.steps)} backward steps</span>
    <span>arrow keys to step</span>
  </div>
</header>
<main>
  <div class="controls">
    <button id="reset">forward</button>
    <button id="prev">&larr; back</button>
    <button id="next">step &rarr;</button>
    <button id="play">play</button>
    <input type="range" id="slider" min="-1" max="{len(trace.steps) - 1}" value="-1">
  </div>
  <p class="status" id="status"></p>
  <div class="board" style="max-height:70vh">
    <svg id="canvas" width="{width:.0f}" height="{height:.0f}"
         viewBox="0 0 {width:.0f} {height:.0f}"></svg>
  </div>
  <div class="panel">
    <section>
      <h2 id="sel-name">&nbsp;</h2>
      <p class="shape" id="sel-shape">&nbsp;</p>
      <div id="sel-value"></div>
    </section>
    <section>
      <h2>gradient at this step</h2>
      <p class="shape">dL/d(selected tensor)</p>
      <div id="sel-grad"></div>
    </section>
  </div>
</main>"""
    script = (
        _GRAPH_JS.replace("__DATA__", trace.to_json())
        .replace("__NW__", str(NODE_W))
        .replace("__NH__", str(NODE_H))
    )
    return _document(title, body, _GRAPH_CSS, script)


# ------------------------------------------------------------ training page
_TRAIN_CSS = """
.charts{display:grid;gap:24px;grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
figure{margin:0;background:var(--surface);border:1px solid var(--rule);
  border-radius:10px;padding:18px 20px 12px}
figcaption{font-family:var(--display);font-weight:600;font-size:1rem;
  margin:0 0 2px}
figure p.sub{font-family:var(--mono);font-size:.72rem;color:var(--ink-3);
  margin:0 0 14px}
.axis{stroke:var(--rule);stroke-width:1}
.grid{stroke:var(--rule-soft);stroke-width:1}
.tick{font-family:var(--mono);font-size:9.5px;fill:var(--ink-3)}
.keys{display:flex;gap:16px;flex-wrap:wrap;font-family:var(--mono);
  font-size:.71rem;color:var(--ink-2);margin:10px 0 0}
table.summary{border-collapse:collapse;font-family:var(--mono);font-size:.76rem;
  margin-top:26px;width:100%;background:var(--surface);border:1px solid var(--rule)}
table.summary th{text-align:left;font-weight:700;color:var(--ink-2);
  border-bottom:1px solid var(--rule);padding:9px 14px}
table.summary td{border-top:1px solid var(--rule-soft);padding:8px 14px}
"""

_W, _H = 470.0, 260.0
_ML, _MR, _MT, _MB = 54.0, 14.0, 12.0, 34.0


def _ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


def _fmt(v: float) -> str:
    a = abs(v)
    if a == 0:
        return "0"
    if a < 1e-2 or a >= 1e4:
        return f"{v:.0e}"
    return f"{v:.3g}"


def line_chart(series: Sequence[dict], *, log_y: bool = False,
               x_label: str = "", y_label: str = "") -> str:
    """A minimal SVG line chart. Each series is {label, points, color, width?}.

    Written by hand rather than pulled from a plotting library so that the
    output stays a single self-contained file with no runtime dependency.
    """
    import math

    pts = [p for s in series for p in s["points"]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    if log_y:
        ys = [math.log10(max(y, 1e-12)) for y in ys]
    y0, y1 = min(ys), max(ys)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1
    pad = (y1 - y0) * 0.08
    y0, y1 = y0 - pad, y1 + pad

    def sx(x: float) -> float:
        return _ML + (x - x0) / (x1 - x0) * (_W - _ML - _MR)

    def sy(y: float) -> float:
        v = math.log10(max(y, 1e-12)) if log_y else y
        return _H - _MB - (v - y0) / (y1 - y0) * (_H - _MT - _MB)

    out = [f'<svg width="100%" viewBox="0 0 {_W:.0f} {_H:.0f}" role="img">']
    for t in _ticks(y0, y1):
        y = _H - _MB - (t - y0) / (y1 - y0) * (_H - _MT - _MB)
        label = 10 ** t if log_y else t
        out.append(f'<line class="grid" x1="{_ML}" y1="{y:.1f}" '
                   f'x2="{_W - _MR}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{_ML - 7}" y="{y + 3:.1f}" '
                   f'text-anchor="end">{_fmt(label)}</text>')
    for t in _ticks(x0, x1, 4):
        x = sx(t)
        out.append(f'<text class="tick" x="{x:.1f}" y="{_H - _MB + 16:.0f}" '
                   f'text-anchor="middle">{_fmt(t)}</text>')
    out.append(f'<line class="axis" x1="{_ML}" y1="{_MT}" x2="{_ML}" '
               f'y2="{_H - _MB}"/>')
    out.append(f'<line class="axis" x1="{_ML}" y1="{_H - _MB}" '
               f'x2="{_W - _MR}" y2="{_H - _MB}"/>')
    for s in series:
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(px):.1f},{sy(py):.1f}"
            for i, (px, py) in enumerate(s["points"])
        )
        out.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" '
                   f'stroke-width="{s.get("width", 1.9)}" '
                   f'opacity="{s.get("opacity", 1)}" stroke-linejoin="round"/>')
    if x_label:
        out.append(f'<text class="tick" x="{(_ML + _W - _MR) / 2:.0f}" '
                   f'y="{_H - 4:.0f}" text-anchor="middle">{x_label}</text>')
    if y_label:
        out.append(f'<text class="tick" x="12" y="{_H / 2:.0f}" '
                   f'transform="rotate(-90 12 {_H / 2:.0f})" '
                   f'text-anchor="middle">{y_label}</text>')
    out.append("</svg>")
    return "".join(out)


def training_page(title: str, subtitle: str, figures: Sequence[dict],
                  summary: Sequence[Sequence[str]] | None = None) -> str:
    """Assemble figures (each {title, subtitle, svg, keys}) into one report."""
    blocks = []
    for f in figures:
        keys = "".join(
            f'<span><i class="swatch" style="background:{k["color"]}"></i>'
            f'{k["label"]}</span>'
            for k in f.get("keys", [])
        )
        blocks.append(
            f'<figure><figcaption>{f["title"]}</figcaption>'
            f'<p class="sub">{f.get("subtitle", "")}</p>{f["svg"]}'
            f'<div class="keys">{keys}</div></figure>'
        )
    table = ""
    if summary:
        head = "".join(f"<th>{c}</th>" for c in summary[0])
        rows = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
            for row in summary[1:]
        )
        table = f'<table class="summary"><tr>{head}</tr>{rows}</table>'
    body = (
        f"<header><h1>{title}</h1><p>{subtitle}</p></header>"
        f'<main><div class="charts">{"".join(blocks)}</div>{table}</main>'
    )
    return _document(title, body, _BASE_CSS_EXTRA + _TRAIN_CSS)


_BASE_CSS_EXTRA = """
.swatch{display:inline-block;width:11px;height:11px;border-radius:2px;
  margin-right:6px;vertical-align:-1px}
"""
