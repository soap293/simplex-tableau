import json
from decimal import Decimal
from fractions import Fraction
import io
from contextlib import redirect_stdout

import streamlit as st

# Local solver
from src.simplex import LP, solve

st.set_page_config(page_title="Simplex Visualizer", layout="wide")
st.title("Simplex (Tableau) — Solve & Visualize")

# Sidebar options
with st.sidebar:
    st.header("Options")
    method = st.selectbox("Method", ["auto", "big_m", "two_phase"], index=0)
    is_min = st.checkbox("Minimize (default: Maximize)", value=False)
    show_graph = st.checkbox("Show graph (2 variables only)", value=True)

# Default JSON template
default_json = {
    "c": [800, 600],
    "A": [[250, 450], [250, 50]],
    "b": [9000, 5000],
    "senses": ["<=", "<="],
    "maximize": True
}

st.subheader("Model JSON")
json_text = st.text_area("Edit LP JSON here", json.dumps(default_json, indent=2), height=260)

col_run, col_reset = st.columns([1, 1])
run = col_run.button("Solve")
if col_reset.button("Reset to template"):
    st.experimental_rerun()

# Helpers

def F(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, Decimal):
        return Fraction(x)
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        return Fraction.from_float(x).limit_denominator(10**12)
    return Fraction(str(x))


def fmt_out(x):
    fr = F(x)
    if fr.denominator == 1:
        return str(fr.numerator)
    sign = '-' if fr.numerator * fr.denominator < 0 else ''
    return f"{sign}{abs(fr.numerator)}/{abs(fr.denominator)}"


def plot_2d(lp: LP, res):
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import combinations

    if len(lp.c) != 2:
        return None

    A = [[float(v) for v in row] for row in lp.A]
    b = [float(v) for v in lp.b]
    senses = lp.senses

    pts = []
    for (i, j) in combinations(range(len(A)), 2):
        a1, a2 = A[i]
        c1, c2 = A[j]
        det = a1*c2 - a2*c1
        if abs(det) < 1e-12:
            continue
        x = (b[i]*c2 - a2*b[j]) / det
        y = (a1*b[j] - b[i]*c1) / det
        pts.append((x, y))

    for i in range(len(A)):
        a1, a2 = A[i]
        if abs(a1) > 1e-12:
            pts.append((b[i]/a1, 0.0))
        if abs(a2) > 1e-12:
            pts.append((0.0, b[i]/a2))
    pts.append((0.0, 0.0))

    def feasible(p):
        x, y = p
        ok = True
        for (row, bi, s) in zip(A, b, senses):
            lhs = row[0]*x + row[1]*y
            if s == "<=":
                ok &= lhs <= bi + 1e-9
            elif s == ">=":
                ok &= lhs >= bi - 1e-9
            elif s == "=":
                ok &= abs(lhs - bi) <= 1e-9
        return ok and x >= -1e-9 and y >= -1e-9

    feas = [p for p in pts if feasible(p)]
    if not feas:
        return None

    xs = [p[0] for p in feas]
    ys = [p[1] for p in feas]
    xmin, xmax = min(0.0, min(xs)), max(xs)*1.2 + 1e-9
    ymin, ymax = min(0.0, min(ys)), max(ys)*1.2 + 1e-9

    grid_x = np.linspace(xmin, xmax, 400)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot constraints with distinct colors
    color_cycle = plt.rcParams.get('axes.prop_cycle', None)
    colors = color_cycle.by_key()['color'] if color_cycle else [f'C{i}' for i in range(10)]
    for i in range(len(A)):
        a1, a2 = A[i]
        c = colors[i % len(colors)]
        label = f"Constraint {i+1} [{c}]: {a1}x1 + {a2}x2 {senses[i]} {b[i]}"
        if abs(a2) < 1e-12:
            x0 = b[i]/a1 if abs(a1) > 1e-12 else 0
            ax.axvline(x0, color=c, alpha=0.7, label=label)
        else:
            y = (b[i] - a1*grid_x)/a2
            ax.plot(grid_x, y, color=c, alpha=0.7, label=label)

    # Shade feasible region
    X, Y = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    mask = np.ones_like(X, dtype=bool)
    for (row, bi, s) in zip(A, b, senses):
        lhs = row[0]*X + row[1]*Y
        if s == "<=":
            mask &= lhs <= bi + 1e-9
        elif s == ">=":
            mask &= lhs >= bi - 1e-9
        else:
            mask &= np.abs(lhs - bi) <= 1e-9
    mask &= (X >= -1e-12) & (Y >= -1e-12)
    ax.contourf(X, Y, mask, levels=[0.5, 1.5], colors=['#e8f7ff'], alpha=0.5)

    # Collect BFS (extreme points) including axes intersections
    def bfs_points():
        lines = []  # (a1,a2,b)
        for i in range(len(A)):
            a1, a2 = A[i]
            lines.append((a1, a2, b[i]))
        lines.append((1.0, 0.0, 0.0))  # x=0
        lines.append((0.0, 1.0, 0.0))  # y=0
        cand = []
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                a1, a2, bi = lines[i]
                c1, c2, bj = lines[j]
                det = a1*c2 - a2*c1
                if abs(det) < 1e-12:
                    continue
                x = (bi*c2 - a2*bj) / det
                y = (a1*bj - bi*c1) / det
                if feasible((x, y)):
                    cand.append((x, y))
        uniq = []
        for (x, y) in cand:
            if not any(abs(x-x2) < 1e-7 and abs(y-y2) < 1e-7 for (x2,y2) in uniq):
                uniq.append((x, y))
        return uniq

    bfs = bfs_points()

    # Iso-profit through optimum + highlight optimal edge if alternate optimal
    if res and res.status == 'optimal' and res.solution is not None:
        try:
            xopt = float(F(res.solution[0]))
            yopt = float(F(res.solution[1]))
            zopt = float(F(res.optimal_value))
            c1, c2 = float(F(lp.c[0])), float(F(lp.c[1]))
            if abs(c2) < 1e-12:
                x_iso = zopt / (c1 if abs(c1) > 1e-12 else 1)
                ax.axvline(x_iso, color='red', linestyle='--', label='iso-profit')
            else:
                y_iso = (zopt - c1*grid_x)/c2
                ax.plot(grid_x, y_iso, 'r--', label='iso-profit (through optimum)')
            ax.plot([xopt], [yopt], 'ro', label=f"optimal ({xopt:.3g}, {yopt:.3g})")
            ax.annotate(f"Z* = {zopt:.4g}", (xopt, yopt), textcoords="offset points", xytext=(8, 8))

            # Highlight optimal edge if alternate optimal
            alt = bool(getattr(res, 'details', {}) and res.details.get('alternate_optimal'))
            if alt and bfs:
                vals = [(x, y, c1*x + c2*y) for (x, y) in bfs]
                on_edge = [(x, y) for (x, y, val) in vals if abs(val - zopt) <= 1e-6]
                if len(on_edge) >= 2:
                    on_edge_sorted = sorted(on_edge)
                    x1, y1 = on_edge_sorted[0]
                    x2, y2 = on_edge_sorted[-1]
                    ax.plot([x1, x2], [y1, y2], color='red', linewidth=3, alpha=0.6, label='optimal edge (∞ solutions)')
        except Exception:
            pass

    # Plot BFS points
    if bfs:
        bx = [p[0] for p in bfs]
        by = [p[1] for p in bfs]
        ax.scatter(bx, by, s=25, color='#444444', alpha=0.9, label='BFS')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Constraints, Feasible Region, Iso-profit')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


if run:
    # Parse JSON
    try:
        cfg = json.loads(json_text, parse_float=Decimal)
        # Override maximize with UI toggle if provided
        if is_min:
            cfg["maximize"] = False
        elif "maximize" not in cfg:
            cfg["maximize"] = True
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
    else:
        try:
            lp = LP(
                c=cfg["c"],
                A=cfg["A"],
                b=cfg["b"],
                senses=cfg["senses"],
                maximize=cfg.get("maximize", True),
            )
        except Exception as e:
            st.error(f"Invalid LP fields: {e}")
        else:
            # Auto-select M for Big-M based on problem scale
            def auto_M(lp_obj: LP) -> float:
                vals = []
                vals += [abs(float(F(v))) for v in lp_obj.c]
                for row in lp_obj.A:
                    vals += [abs(float(F(x))) for x in row]
                vals += [abs(float(F(v))) for v in lp_obj.b]
                maxv = max([1.0] + vals)
                return 1000.0 * maxv

            # Capture solver verbose output
            buf = io.StringIO()
            with redirect_stdout(buf):
                M_arg = auto_M(lp) if method == "big_m" else 10**6
                res = solve(lp, method=method, verbose=True, M=M_arg)
            text_out = buf.getvalue()

            # Single-column layout: Iterations -> Result -> Graph
            st.subheader("Iterations / Tableaux")
            st.code(text_out)
            st.subheader("Result")
            st.json({
                "status": res.status,
                "optimal_value": fmt_out(res.optimal_value) if res.optimal_value is not None else None,
                "solution": [fmt_out(v) for v in (res.solution or [])],
                "iterations": res.iterations,
                "method": res.method,
            })
            # Infinite many solutions note
            if isinstance(res.details, dict) and res.details.get('alternate_optimal'):
                st.info("Infinite many optimal solutions along an edge (alternate optimal).")

            st.subheader("Graph")
            if show_graph and len(lp.c) == 2:
                fig = plot_2d(lp, res)
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.info("No feasible region to plot or numerical issue.")
            else:
                st.info("Graph available only for 2 variables.")
