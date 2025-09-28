# Simplex (Tableau) Solver

This project implements a Simplex tableau solver supporting Big-M and Two-Phase methods, with step-by-step tableau printing and exact fraction math.

## How to Run

Two ways to run (they are independent and can coexist):

### 1) CLI (no UI)

- Solve from JSON:

```bash
python -m src.simplex examples/max_leq.json --method big_m
```

- Show 2D graph (only for 2 variables):

```bash
python -m src.simplex examples/max_leq.json --method big_m --graph
```

Options:
- `--method {auto,big_m,two_phase}`
- `--sense {max|min}` set objective sense (default: follow JSON if present, else max)
- `--min` deprecated alias for `--sense min` (still supported for compatibility)
- `--no-verbose` hide iterations
- `--graph` display 2D plot (needs numpy & matplotlib; only for 2 variables)

Note about Big-M: the CLI accepts `--M`, but the UI auto-selects a large M based on problem scale; you usually don’t need to set it manually.

### 2) Streamlit UI

Interactive interface to edit JSON, pick method, and plot constraints/feasible region/iso-profit.

Sidebar controls:
- Method: auto / big_m / two_phase
- Minimize (default: Maximize)
- Show graph (2 variables only)

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

- Left pane shows the iteration table output and final result (printed as exact fractions).
- Right pane shows a 2D plot (when the model has exactly 2 variables):
  - Each constraint line (distinct colors)
  - Feasible region shading
  - All BFS (extreme points) marked
  - Iso-profit (or iso-cost) line through the optimum
  - Optimal point with Z* label
  - If there are infinitely many optimal solutions, the optimal edge segment is highlighted

Tip: To stop the UI server, press Ctrl+C in the terminal where it’s running.

## Input JSON format

```
{
  "c": [c1, c2, ...],
  "A": [[...], [...], ...],
  "b": [b1, b2, ...],
  "senses": ["<=", ">=", "="],
  "maximize": true  // optional; default is maximize if omitted
}
```

Maximize vs Minimize
- If `maximize` is omitted, the solver assumes maximize.
- Set `"maximize": false` in JSON to solve a minimization.
- CLI `--sense {max|min}` overrides JSON when provided. If not provided, CLI follows JSON; if JSON has no `maximize`, default is max.
- `--min` is a deprecated alias of `--sense min` and remains supported for compatibility.

Examples:
```bash
# Maximize (default: JSON omits maximize)
python -m src.simplex examples/max_leq.json --method auto

# Force minimization via CLI
python -m src.simplex examples/problem_4_4_2_min.json --method auto --sense min --graph

# Legacy flag (deprecated but supported)
python -m src.simplex examples/problem_4_4_2_min.json --method auto --min
```

## Run (quick reference)

```bash
# CLI, auto method (will choose two_phase if any ">=" or "="; otherwise big_m)
python -m src.simplex examples/max_leq.json --method auto

# CLI with 2D plot (shows constraints, feasible region, BFS, iso line, optimal point; highlights optimal edge if alternate optimal)
python -m src.simplex examples/max_leq.json --method big_m --graph

# UI
streamlit run app.py
```

## Examples

Maximization with `<=` constraints only (no artificials needed):
```
{
  "c": [3, 2],
  "A": [[1, 1], [1, 0], [0, 1]],
  "b": [4, 2, 3],
  "senses": ["<=", "<=", "<="]
}
```
Save as `examples/max_leq.json` and run:
```
python -m src.simplex examples/max_leq.json --method auto
```

A case with `>=`/`=` (needs artificials). Save as `examples/geq_eq.json`:
```
{
  "c": [1, 1],
  "A": [[1, -1], [1, 1]],
  "b": [1, 3],
  "senses": [">=", "="]
}
```
Run:
```
python -m src.simplex examples/geq_eq.json --method two_phase
```

Minimization example (from textbook Problem 4.4.2) with 2 variables so graphing is available:
```
{
  "c": [1, 5],
  "A": [[1, 2], [1, 1], [5, 1], [1, 0]],
  "b": [3, 2, 5, 4],
  "senses": [">=", ">=", ">=", "<="],
  "maximize": false
}
```
Run:
```
python -m src.simplex examples/problem_4_4_2_min.json --method auto --sense min --graph
```
Expected result: x* = (3, 0), Z* = 3.

## Notes
- Uses Bland's-like tie-breaking by stable Python sorting.
- Detects unbounded and infeasible cases.
- If you minimize, the solver internally flips the objective and flips the final value back.
- Internals use Fraction for exact arithmetic to avoid floating errors.
- The tableau prints the Z row at the top, with Big-M Z row showing symbolic M terms when applicable.
- Pivot element is highlighted, ratios column is displayed.
