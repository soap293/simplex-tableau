from __future__ import annotations

"""
Simplex (Tableau) Solver with Big-M and Two-Phase methods.
- Supports max/min by converting to max internally.
- Constraints: <=, >=, =
- Shows each tableau iteration with clear pivot info.
- Detects infeasible and unbounded cases.

Input contract (programmatic API):
- c: list[float] objective coefficients for original variables (length n)
- A: list[list[float]] constraints coefficients (m x n)
- b: list[float] RHS (length m)
- senses: list[str] with entries in {"<=", ">=", "="}
- maximize: bool (True for max, False for min)
- method: str in {"auto", "big_m", "two_phase"}

CLI expects a JSON file with the same structure.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union
import math
import json
from fractions import Fraction
from decimal import Decimal
from itertools import combinations

EPS = 1e-9

# --- Exact rational helpers ---
Num = Union[int, float, Fraction, Decimal]

def F(x: Num) -> Fraction:
    """Convert a number to Fraction exactly when possible.
    - Fraction -> as is
    - Decimal -> exact rational
    - int -> exact
    - float -> best rational approx (limit large denominator)
    """
    if isinstance(x, Fraction):
        return x
    if isinstance(x, Decimal):
        return Fraction(x)
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        # Use from_float to preserve the exact binary rational, then limit to a large denominator
        return Fraction.from_float(x).limit_denominator(10**12)
    # Fallback via str to avoid float artifacts
    return Fraction(str(x))

def is_zero(x: Union[Fraction, float, int]) -> bool:
    if isinstance(x, Fraction):
        return x == 0
    return abs(float(x)) < EPS

def fmt_out(x: Num) -> str:
    """Pretty-print numbers as integers or reduced fractions for final outputs."""
    fr = F(x)
    if fr.denominator == 1:
        return str(fr.numerator)
    sign = '-' if fr.numerator * fr.denominator < 0 else ''
    return f"{sign}{abs(fr.numerator)}/{abs(fr.denominator)}"

@dataclass
class LP:
    c: List[float]
    A: List[List[float]]
    b: List[float]
    senses: List[str]
    maximize: bool = True

@dataclass
class SimplexResult:
    status: str  # optimal | infeasible | unbounded
    optimal_value: Optional[float]
    solution: Optional[List[float]]  # values for original n variables
    iterations: int
    method: str
    details: Dict[str, object]

class Tableau:
    def __init__(self, mat: List[List[Num]], basis: List[int], var_names: List[str], obj_is_max=True):
        # Matrix layout per row: [b, var1, var2, ..., varN, zj-cj]
        # store everything as Fractions for exact arithmetic
        self.T = [[F(v) for v in row] for row in mat]  # (m+1) x (N+2)
        self.m = len(mat) - 1
        self.cols = len(mat[0])            # total columns including b and last (zj-cj)
        self.num_vars = self.cols - 2      # number of variable columns
        self.basis = basis[:]              # indices into variable columns (0..num_vars-1)
        self.var_names = var_names[:]
        self.obj_is_max = obj_is_max
        self.iter = 0
        # Big-M meta (optional). If set, enables symbolic Z row with M terms
        self.cnum = None  # list[Fraction] same length as num_vars
        self.cM = None    # list[int] (0/1) same length as num_vars

    def _fmt_M(self, coeff_M: Num, coeff_num: Num) -> str:
        # Render coeff_M*M + coeff_num, simplifying zeros and using fractions
        coeff_M = F(coeff_M)
        coeff_num = F(coeff_num)
        parts: List[str] = []
        if coeff_M != 0:
            sM = self._fmt_frac(coeff_M)
            if sM == "1":
                parts.append("M")
            elif sM == "-1":
                parts.append("-M")
            else:
                parts.append(f"{sM}M")
        if coeff_num != 0:
            s = self._fmt_frac(coeff_num)
            if parts and not s.startswith("-"):
                s = "+" + s
            parts.append(s)
        if not parts:
            return "0"
        return " ".join(parts)

    def _fmt_frac(self, x: Num) -> str:
        # Format a number as integer or reduced fraction without float artifacts
        # Prefer exact Fraction when available
        if isinstance(x, Fraction):
            fr = x
        elif isinstance(x, Decimal):
            fr = Fraction(x)
        elif isinstance(x, int):
            fr = Fraction(x)
        elif isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return str(x)
            if abs(x) < 1e-12:
                x = 0.0
            # convert float to rational approx
            fr = Fraction.from_float(x).limit_denominator(10**6)
        else:
            fr = Fraction(str(x))

        if fr == 0:
            return "0"
        if fr.denominator == 1:
            return str(fr.numerator)
        sign = '-' if fr.numerator * fr.denominator < 0 else ''
        return f"{sign}{abs(fr.numerator)}/{abs(fr.denominator)}"

    def print_tableau(self, header: str = "", enter_j: Optional[int] = None, leave_i: Optional[int] = None):
        # Print in format: Iteration k, then header: Z x1 x2 ... RHS BV Ratio
        title = header or f"Iteration {self.iter}"
        print(f"\n{title}")
        headers = ["Z"] + self.var_names + ["RHS", "BV", "Ratio"]

        def fmt_num(x: Num) -> str:
            return self._fmt_frac(x)

        # compute column width from data
        samples = headers[:]
        for i in range(self.m):
            samples.append(fmt_num(self.T[i][0]))
            for j in range(1, 1 + self.num_vars):
                samples.append(fmt_num(self.T[i][j]))
        samples.append(fmt_num(self.T[-1][0]))
        colw = max(6, max(len(s) for s in samples) + 2)

        # header and separator
        print(" ".join(f"{h:>{colw}}" for h in headers))
        print("-" * (len(headers) * (colw + 1)))

        # Z row (row 0)
        if self.cM is not None and any(self.cM):
            cb_num = [self.cnum[self.basis[i]] if 0 <= self.basis[i] < self.num_vars else F(0) for i in range(self.m)]
            cb_M = [self.cM[self.basis[i]] if 0 <= self.basis[i] < self.num_vars else 0 for i in range(self.m)]
            z_cells: List[str] = ["Z"]
            for j in range(1, 1 + self.num_vars):
                zj_num = sum(cb_num[i] * self.T[i][j] for i in range(self.m))
                zj_M = sum(F(cb_M[i]) * self.T[i][j] for i in range(self.m))
                cnum_j = self.cnum[j-1]
                cM_j = self.cM[j-1]
                coeff_M = cM_j - zj_M
                coeff_num = zj_num - cnum_j
                z_cells.append(self._fmt_M(coeff_M, coeff_num))
            ZM_rhs = sum(F(cb_M[i]) * self.T[i][0] for i in range(self.m))
            Znum_rhs = sum(cb_num[i] * self.T[i][0] for i in range(self.m))
            z_cells.extend([self._fmt_M(-ZM_rhs, Znum_rhs), "Z", ""])  # RHS, BV, Ratio
            print(" ".join(f"{c:>{colw}}" for c in z_cells))
        else:
            z_cells = ["Z"]
            z_cells += [fmt_num(self.T[-1][j]) for j in range(1, 1 + self.num_vars)]
            z_cells += [fmt_num(self.T[-1][0]), "Z", ""]
            print(" ".join(f"{c:>{colw}}" for c in z_cells))

        # constraint rows
        for i in range(self.m):
            row_cells: List[str] = [""]
            for j in range(1, 1 + self.num_vars):
                val_str = fmt_num(self.T[i][j])
                if enter_j is not None and leave_i is not None and i == leave_i and j == enter_j:
                    val_str = f"⭕{val_str}"
                row_cells.append(val_str)
            rhs = fmt_num(self.T[i][0])
            bv_name = self.var_names[self.basis[i]] if 0 <= self.basis[i] < len(self.var_names) else f"x{self.basis[i]+1}"
            if enter_j is not None:
                aij = self.T[i][enter_j]
                ratio_cell = fmt_num(self.T[i][0] / aij) if aij > 0 else ""
            else:
                ratio_cell = ""
            row_cells.extend([rhs, bv_name, ratio_cell])
            print(" ".join(f"{c:>{colw}}" for c in row_cells))

    def choose_entering(self) -> Optional[int]:
        obj_row = self.T[-1]
        best_j = None
        best_val: Fraction = F(0)
        for j in range(1, 1 + self.num_vars):
            rc = obj_row[j]
            if self.obj_is_max:
                if rc < best_val:
                    best_val = rc
                    best_j = j
            else:
                if rc > best_val:
                    best_val = rc
                    best_j = j
        return best_j

    def choose_leaving(self, enter_j: int) -> Optional[int]:
        ratios = []
        for i in range(self.m):
            aij = self.T[i][enter_j]
            if aij > 0:
                ratios.append((self.T[i][0] / aij, i))
        if not ratios:
            return None
        ratios.sort()
        return ratios[0][1]

    def pivot(self, row: int, col: int):
        piv = self.T[row][col]
        if is_zero(piv):
            raise RuntimeError("Zero pivot encountered")
        factor = piv
        for j in range(self.cols):
            self.T[row][j] /= factor
        for i in range(self.m+1):
            if i == row:
                continue
            coeff = self.T[i][col]
            if is_zero(coeff):
                continue
            for j in range(self.cols):
                self.T[i][j] -= coeff * self.T[row][j]
        self.basis[row] = col - 1

    def solve(self, verbose=True) -> Tuple[str, int]:
        if verbose:
            print("\nInitial tableau")
            self.print_tableau(header=f"Iteration {self.iter}")
        while True:
            enter_j = self.choose_entering()
            if enter_j is None:
                if verbose:
                    self.print_tableau(header=f"Final tableau (Iteration {self.iter})")
                return ("optimal", self.iter)
            leave_i = self.choose_leaving(enter_j)
            if leave_i is None:
                if verbose:
                    # show the ratio column with no leaving row picked
                    self.print_tableau(header=f"Final tableau (unbounded)", enter_j=enter_j, leave_i=None)
                return ("unbounded", self.iter)
            self.iter += 1
            if verbose:
                self.print_tableau(header=f"Iteration {self.iter}", enter_j=enter_j, leave_i=leave_i)
            self.pivot(leave_i, enter_j)
                # no 'After pivot' print; next loop prints next iteration state


def build_tableau_for_big_m(lp: LP, M: Num = 10**6) -> Tuple[Tableau, List[int], Dict[str, List[int]]]:
    # Returns tableau, map of indices
    m, n = len(lp.A), len(lp.c)
    # Convert to maximization
    c = [F(v) for v in lp.c[:]]
    if not lp.maximize:
        c = [F(-x) for x in c]

    # Start building columns: original x (n), slack (for <=), surplus (for >=), artificial (for = or >=), RHS, zj-cj
    slack_idx = []
    surplus_idx = []
    art_idx = []
    var_names = [f"x{j+1}" for j in range(n)]

    rows = []
    basis = [-1] * m

    for i in range(m):
        sense = lp.senses[i]
        row = [F(lp.b[i])] + [F(v) for v in lp.A[i][:]]  # start with b and x's
        # placeholders; will append after knowing lengths
        rows.append((sense, row))

    # Determine counts and build augmented rows
    # We'll append slack/surplus/artificial as we traverse; track mapping per row
    T_rows = []
    # dynamic column assembly; we keep vectors for each extra var
    extra_cols = []  # list of column vectors for extra vars (as Fractions)
    extra_names = []
    basic_per_row = []

    for i, (sense, row) in enumerate(rows):
        b_i = row[0]
        x_part = row[1:]
        if sense == "<=":
            # add slack s_i
            col = [F(0)]*m
            col[i] = F(1)
            extra_cols.append(col)
            extra_names.append(f"s{i+1}")
            basic_per_row.append(len(var_names) + len(extra_names) - 1)
        elif sense == ">=":
            # add surplus (-1) and artificial (+1)
            col_sur = [F(0)]*m
            col_sur[i] = F(-1)
            extra_cols.append(col_sur)
            extra_names.append(f"e{i+1}")
            # artificial
            col_art = [F(0)]*m
            col_art[i] = F(1)
            extra_cols.append(col_art)
            extra_names.append(f"a{i+1}")
            basic_per_row.append(len(var_names) + len(extra_names) - 1)
        elif sense == "=":
            # artificial only
            col_art = [F(0)]*m
            col_art[i] = F(1)
            extra_cols.append(col_art)
            extra_names.append(f"a{i+1}")
            basic_per_row.append(len(var_names) + len(extra_names) - 1)
        else:
            raise ValueError("sense must be one of <=, >=, =")

    # Assemble full matrix columns
    N = len(var_names) + len(extra_names)
    # Build A' with size m x N
    Aprime = [[F(0)]*N for _ in range(m)]
    # fill original x columns
    for i in range(m):
        # rows[i] = (sense, [b, Arow...])
        _, r = rows[i]
        Arow = r[1:]
        for j in range(n):
            Aprime[i][j] = F(Arow[j])
    # fill extra columns
    for k, col in enumerate(extra_cols):
        j = len(var_names) + k
        for i in range(m):
            Aprime[i][j] = F(col[i])

    # Build tableau matrix: (m+1) rows, (N+2) columns with b at col 0, variables 1..N, last is zj-cj
    T = [[F(0)]*(N+2) for _ in range(m+1)]
    for i in range(m):
        T[i][0] = F(rows[i][1][0])  # b
        for j in range(N):
            T[i][j+1] = F(Aprime[i][j])
        T[i][-1] = F(0)

    # Objective row for Big-M: z row with reduced costs
    # z = c^T x - M * sum(a)
    # Start with -c for reduced costs (since we keep zj-cj convention); then adjust for basic artificial variables
    M = F(M)
    c_full = [F(0)]*N
    for j in range(n):
        c_full[j] = F(c[j])
    # artificial columns get -M in objective (since we maximize, artificial variables have -M penalty)
    # Track Big-M symbolic parts: c_full[j] = cnum[j] + (-M) * cM[j]
    cnum = [F(0)]*N
    cM = [0]*N
    for j in range(n):
        c_full[j] = F(c[j])
        cnum[j] = F(c[j])
        cM[j] = 0
    for idx, name in enumerate(extra_names):
        j = len(var_names) + idx
        if name.startswith('a'):
            c_full[j] = -M
            cnum[j] = F(0)
            cM[j] = 1
        else:
            c_full[j] = F(0)
            cnum[j] = F(0)
            cM[j] = 0

    # Initialize basis as the chosen basic_per_row
    basis = basic_per_row[:]

    # Compute initial reduced costs: zj - cj. Build z row by summing basic rows * their c_b, then subtract c_j
    zrow = [F(0)]*(N+2)
    # z value (b part): sum c_b * b_i
    cb = [c_full[j] for j in basis]
    zrow[0] = sum(cb[i] * T[i][0] for i in range(m))
    # zj for each column
    for j in range(1, N+1):
        zrow[j] = sum(cb[i] * T[i][j] for i in range(m))
    # reduced costs zj - cj (store directly)
    for j in range(N):
        zrow[j+1] -= c_full[j]
    # last column keep 0 for now (zj-cj for RHS isn't used); we store it as 0
    T[-1] = zrow

    var_names_all = var_names + extra_names

    tab = Tableau(T, basis, var_names_all, obj_is_max=True)
    # attach Big-M meta for pretty Z printing
    tab.cnum = cnum
    tab.cM = cM

    return tab, basis, {"var_names": var_names_all, "N": N, "n": n, "M": int(M)}


def extract_solution(tableau: Tableau, meta: Dict[str, object], original_n: int) -> List[float]:
    # read basic variable values from RHS
    m, num_vars = tableau.m, tableau.num_vars
    values = [F(0)]*num_vars
    for i in range(m):
        col = tableau.basis[i]
        if 0 <= col < num_vars:
            values[col] = tableau.T[i][0]
    return values[:original_n]


def simplex_big_m(lp: LP, verbose=True, M: float = 1e6) -> SimplexResult:
    tab, basis, meta = build_tableau_for_big_m(lp, M=M)
    status, iters = tab.solve(verbose=verbose)

    # Feasibility check: any artificial variable in basis with positive value implies infeasible (if > EPS)
    art_indices = [j for j, name in enumerate(meta["var_names"]) if name.startswith('a')]
    infeasible = False
    for i in range(tab.m):
        col = tab.basis[i]
        if col in art_indices and tab.T[i][0] > 0:
            infeasible = True
            break

    if status == "unbounded":
        return SimplexResult(status="unbounded", optimal_value=None, solution=None, iterations=iters, method="big_m", details={})
    if infeasible:
        return SimplexResult(status="infeasible", optimal_value=None, solution=None, iterations=iters, method="big_m", details={})

    # Optimal
    # Objective value is z in T[-1][0]
    z = tab.T[-1][0]
    # If original was minimization, flip sign back
    if not lp.maximize:
        z = -z
    x = extract_solution(tab, meta, original_n=len(lp.c))
    # Detect alternate optimal solutions: any nonbasic variable with zero reduced cost
    basis_set = set(tab.basis)
    alt_vars = []
    for j in range(tab.num_vars):
        if j not in basis_set:
            rc = tab.T[-1][j+1]
            if rc == 0:
                alt_vars.append(meta["var_names"][j])
    details = {
        "alternate_optimal": len(alt_vars) > 0,
        "alt_zero_rc_vars": alt_vars,
        "var_names": meta["var_names"],
    }
    return SimplexResult(status="optimal", optimal_value=z, solution=x, iterations=iters, method="big_m", details=details)


def build_tableau_for_two_phase(lp: LP) -> Tuple[Tableau, Dict[str, object]]:
    # Phase I: minimize sum of artificials (equivalently maximize -sum a)
    m, n = len(lp.A), len(lp.c)
    # Convert to maximization for internal; objective for phase I is maximize -sum a

    var_names = [f"x{j+1}" for j in range(n)]
    rows = []
    types = []  # 0 slack, 1 surplus+artificial, 2 artificial

    for i in range(m):
        sense = lp.senses[i]
        row = [F(lp.b[i])] + [F(v) for v in lp.A[i][:]]
        rows.append(row)
        if sense == "<=":
            types.append(0)
        elif sense == ">=":
            types.append(1)
        elif sense == "=":
            types.append(2)
        else:
            raise ValueError("sense must be one of <=, >=, =")

    # Build extra columns
    extra_cols = []
    extra_names = []
    basic_per_row = []

    for i, t in enumerate(types):
        if t == 0:
            col = [F(0)]*m; col[i] = F(1)
            extra_cols.append(col); extra_names.append(f"s{i+1}")
            basic_per_row.append(len(var_names) + len(extra_names) - 1)
        elif t == 1:
            col_sur = [F(0)]*m; col_sur[i] = F(-1)
            extra_cols.append(col_sur); extra_names.append(f"e{i+1}")
            col_art = [F(0)]*m; col_art[i] = F(1)
            extra_cols.append(col_art); extra_names.append(f"a{i+1}")
            basic_per_row.append(len(var_names) + len(extra_names) - 1)
        elif t == 2:
            col_art = [F(0)]*m; col_art[i] = F(1)
            extra_cols.append(col_art); extra_names.append(f"a{i+1}")
            basic_per_row.append(len(var_names) + len(extra_names) - 1)

    N = len(var_names) + len(extra_names)
    Aprime = [[F(0)]*N for _ in range(m)]
    for i in range(m):
        for j in range(n):
            Aprime[i][j] = F(rows[i][1:][j])
    for k, col in enumerate(extra_cols):
        j = len(var_names) + k
        for i in range(m):
            Aprime[i][j] = F(col[i])

    # Build tableau for Phase I objective: maximize -sum a (i.e., c_full = -1 on artificial columns)
    T = [[F(0)]*(N+2) for _ in range(m+1)]
    for i in range(m):
        T[i][0] = F(rows[i][0])
        for j in range(N):
            T[i][j+1] = F(Aprime[i][j])
        T[i][-1] = F(0)

    var_names_all = var_names + extra_names

    # c_full for phase I
    c_full = [F(0)]*N
    for j, name in enumerate(var_names_all):
        if name.startswith('a'):
            c_full[j] = F(-1)

    basis = basic_per_row[:]
    cb = [c_full[j] for j in basis]

    zrow = [F(0)]*(N+2)
    zrow[0] = sum(cb[i] * T[i][0] for i in range(m))
    for j in range(1, N+1):
        zrow[j] = sum(cb[i] * T[i][j] for i in range(m))
    for j in range(N):
        zrow[j+1] -= c_full[j]
    T[-1] = zrow

    tab = Tableau(T, basis, var_names_all, obj_is_max=True)

    meta = {
        "var_names": var_names_all,
        "n": n,
        "artificial_indices": [j for j, name in enumerate(var_names_all) if name.startswith('a')],
        "original_obj": (lp.c[:] if lp.maximize else [-x for x in lp.c])
    }

    return tab, meta


def simplex_two_phase(lp: LP, verbose=True) -> SimplexResult:
    # Phase I
    tab, meta = build_tableau_for_two_phase(lp)
    print("\n=== Phase I ===")
    status, iters1 = tab.solve(verbose=verbose)

    # Check feasibility (objective value should be 0)
    z_phase1 = tab.T[-1][0]
    if z_phase1 < F(0):  # should be 0 if feasible (we maximize -sum a)
        return SimplexResult(status="infeasible", optimal_value=None, solution=None, iterations=iters1, method="two_phase", details={})

    # Remove artificial columns from tableau and set up Phase II objective
    N = len(meta["var_names"])
    art_set = set(meta["artificial_indices"])

    # Build mapping of remaining columns
    keep_cols = [j for j in range(N) if j not in art_set]
    new_var_names = [meta["var_names"][j] for j in keep_cols]
    newN = len(keep_cols)

    # Construct new tableau matrix
    m = tab.m
    T2 = [[F(0)]*(newN+2) for _ in range(m+1)]
    new_basis = []
    for i in range(m):
        T2[i][0] = tab.T[i][0]
        for newj, oldj in enumerate(keep_cols):
            T2[i][newj+1] = tab.T[i][oldj+1]
        T2[i][-1] = F(0)
    # Initialize basis mapping from old to new; rows that had artificial basis need re-basing
    for i in range(m):
        if tab.basis[i] in art_set:
            # Try to find a column among remaining vars to serve as basis by pivoting
            chosen = None
            # Prefer a unit column already
            for newj in range(newN):
                if T2[i][newj+1] == F(1) and all(T2[k][newj+1] == F(0) for k in range(m) if k != i):
                    chosen = newj
                    break
            if chosen is None:
                # pick any nonzero column and pivot to make it unit
                for newj in range(newN):
                    if T2[i][newj+1] != F(0):
                        chosen = newj
                        # Normalize row i
                        piv = T2[i][newj+1]
                        for col in range(newN+2):
                            T2[i][col] /= piv
                        # Eliminate other rows
                        for r in range(m+1):
                            if r == i:
                                continue
                            coeff = T2[r][newj+1]
                            if coeff != F(0):
                                for col in range(newN+2):
                                    T2[r][col] -= coeff * T2[i][col]
                        break
            new_basis.append(chosen if chosen is not None else 0)
        else:
            new_basis.append(keep_cols.index(tab.basis[i]))

    # Phase II objective: original max objective over remaining columns
    c_full = [F(0)]*newN
    for newj, name in enumerate(new_var_names):
        if name.startswith('x'):
            idx = int(name[1:]) - 1
            c_full[newj] = F(meta["original_obj"][idx])
        else:
            c_full[newj] = F(0)

    # Build reduced cost row from current basis
    cb = [c_full[j] if j is not None and 0 <= j < newN else F(0) for j in new_basis]
    zrow = [F(0)]*(newN+2)
    zrow[0] = sum(cb[i] * T2[i][0] for i in range(m))
    for j in range(1, newN+1):
        zrow[j] = sum(cb[i] * T2[i][j] for i in range(m))
    for j in range(newN):
        zrow[j+1] -= c_full[j]
    T2[-1] = zrow

    tab2 = Tableau(T2, new_basis, new_var_names, obj_is_max=True)
    print("\n=== Phase II ===")
    status2, iters2 = tab2.solve(verbose=verbose)

    if status2 == "unbounded":
        return SimplexResult(status="unbounded", optimal_value=None, solution=None, iterations=iters1+iters2, method="two_phase", details={})

    z = tab2.T[-1][0]
    if not lp.maximize:
        z = -z
    x = extract_solution(tab2, meta, original_n=len(lp.c))
    # Detect alternate optimal solutions on final tableau
    basis_set = set(tab2.basis)
    alt_vars = []
    for j in range(tab2.num_vars):
        if j not in basis_set:
            rc = tab2.T[-1][j+1]
            if rc == 0:
                name = tab2.var_names[j] if j < len(tab2.var_names) else f"x{j+1}"
                alt_vars.append(name)
    details = {
        "alternate_optimal": len(alt_vars) > 0,
        "alt_zero_rc_vars": alt_vars,
        "var_names": tab2.var_names,
    }
    return SimplexResult(status="optimal", optimal_value=z, solution=x, iterations=iters1+iters2, method="two_phase", details=details)


def solve(lp: LP, method: str = "auto", verbose=True, M: float = 1e6) -> SimplexResult:
    # auto: if any >= or =, prefer two-phase; else plain simplex via big-M without artificials equals no penalty
    if method == "auto":
        if any(s in (">=", "=") for s in lp.senses):
            method = "two_phase"
        else:
            method = "big_m"
    if method == "big_m":
        return simplex_big_m(lp, verbose=verbose, M=M)
    elif method == "two_phase":
        return simplex_two_phase(lp, verbose=verbose)
    else:
        raise ValueError("method must be one of auto, big_m, two_phase")


# CLI

def graph(lp: LP, res: SimplexResult):
    """Plot constraints and an iso-profit line for 2 variables.
    - Only supports 2 variables. Shades feasible region, shows optimal point,
      and marks all BFS. If infinite optimal solutions, highlight the optimal edge.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print("Graphing requires matplotlib and numpy.", e)
        return

    n = len(lp.c)
    if n != 2:
        print("Graph only supports 2 variables.")
        return

    # Build half-planes: a1 x1 + a2 x2 <= b  (convert >=, = appropriately)
    A = [[float(v) for v in row] for row in lp.A]
    b = [float(v) for v in lp.b]
    senses = lp.senses

    # Compute intersection points of all constraint lines treated as equalities
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

    # Add intercepts with axes from each constraint line
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
        print("No feasible region to plot (empty or numerical issues).")
        return

    xs = [p[0] for p in feas]
    ys = [p[1] for p in feas]
    xmin, xmax = min(0.0, min(xs)), max(xs)*1.2 + 1e-9
    ymin, ymax = min(0.0, min(ys)), max(ys)*1.2 + 1e-9

    plt.figure(figsize=(6, 6))
    grid_x = np.linspace(xmin, xmax, 400)

    # Plot constraint lines and half-planes borders with distinct colors
    color_cycle = plt.rcParams.get('axes.prop_cycle', None)
    colors = color_cycle.by_key()['color'] if color_cycle else [f'C{i}' for i in range(10)]
    for i in range(len(A)):
        a1, a2 = A[i]
        c = colors[i % len(colors)]
        label = f"Constraint {i+1}: {a1}x1 + {a2}x2 {senses[i]} {b[i]}"
        if abs(a2) < 1e-12:
            # vertical line a1 x = b
            x0 = b[i]/a1 if abs(a1) > 1e-12 else 0
            plt.axvline(x0, color=c, alpha=0.7, label=label)
        else:
            y = (b[i] - a1*grid_x)/a2
            plt.plot(grid_x, y, color=c, alpha=0.7, label=label)

    # Shade feasible region by sampling a grid
    X, Y = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    mask = np.ones_like(X, dtype=bool)
    for (row, bi, s) in zip(A, b, senses):
        lhs = row[0]*X + row[1]*Y
        if s == "<=":
            mask &= lhs <= bi + 1e-9
        elif s == ">=":
            mask &= lhs >= bi - 1e-9
        else:  # '='
            mask &= np.abs(lhs - bi) <= 1e-9
    mask &= (X >= -1e-12) & (Y >= -1e-12)
    plt.contourf(X, Y, mask, levels=[0.5, 1.5], colors=['#e8f7ff'], alpha=0.5)

    # Collect BFS points (extreme points) including axes intersections
    def bfs_points():
        # candidate lines: each constraint as equality plus x=0 and y=0
        lines = []  # each as (a1,a2,b)
        for i in range(len(A)):
            a1, a2 = A[i]
            lines.append((a1, a2, b[i]))
        lines.append((1.0, 0.0, 0.0))  # x = 0
        lines.append((0.0, 1.0, 0.0))  # y = 0
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
                p = (x, y)
                if feasible(p):
                    cand.append(p)
        # dedup
        uniq = []
        for (x, y) in cand:
            if not any(abs(x-x2) < 1e-7 and abs(y-y2) < 1e-7 for (x2, y2) in uniq):
                uniq.append((x, y))
        return uniq

    bfs = bfs_points()

    # Iso-profit line through optimal solution if available
    if res and res.status == 'optimal' and res.solution is not None:
        try:
            xopt = float(F(res.solution[0]))
            yopt = float(F(res.solution[1]))
            zopt = float(F(res.optimal_value))
            c1, c2 = float(F(lp.c[0])), float(F(lp.c[1]))
            if abs(c2) < 1e-12:
                # vertical iso line c1 x = z
                x_iso = zopt / (c1 if abs(c1) > 1e-12 else 1)
                plt.axvline(x_iso, color='red', linestyle='--', label='iso-profit')
            else:
                y_iso = (zopt - c1*grid_x)/c2
                plt.plot(grid_x, y_iso, 'r--', label='iso-profit')
            # optimal point
            plt.plot([xopt], [yopt], 'ro', label='optimal')
            # annotate coordinates and Z*
            coord_text = f"({fmt_out(res.solution[0])}, {fmt_out(res.solution[1])})"
            plt.annotate(coord_text, (xopt, yopt), textcoords="offset points", xytext=(8, -12))
            plt.annotate(f"Z* = {fmt_out(res.optimal_value)}", (xopt, yopt), textcoords="offset points", xytext=(8, 8))

            # If alternate optimal: highlight the optimal edge (BFS with same objective value)
            alt = bool(res.details.get('alternate_optimal')) if isinstance(res.details, dict) else False
            if alt and bfs:
                vals = [(x, y, c1*x + c2*y) for (x, y) in bfs]
                on_edge = [(x, y) for (x, y, val) in vals if abs(val - zopt) <= 1e-6]
                if len(on_edge) >= 2:
                    # sort and connect extreme two points
                    on_edge_sorted = sorted(on_edge)
                    x1, y1 = on_edge_sorted[0]
                    x2, y2 = on_edge_sorted[-1]
                    plt.plot([x1, x2], [y1, y2], color='red', linewidth=3, alpha=0.6, label='optimal edge (∞ solutions)')
        except Exception:
            pass

    # Plot BFS points
    if bfs:
        bx = [p[0] for p in bfs]
        by = [p[1] for p in bfs]
        plt.scatter(bx, by, s=25, color='#444444', alpha=0.8, label='BFS')

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Constraints and Iso-profit')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    import argparse
    p = argparse.ArgumentParser(description="Tableau Simplex with Big-M and Two-Phase (shows iterations)")
    p.add_argument("json", help="Path to JSON file describing the LP")
    p.add_argument("--method", choices=["auto", "big_m", "two_phase"], default="auto")
    p.add_argument("--sense", choices=["max", "min"], default=None, help="Objective sense (default: use JSON or max)")
    p.add_argument("--min", action="store_true", help="Deprecated: same as --sense min")
    p.add_argument("--no-verbose", action="store_true", help="Hide iteration printouts")
    p.add_argument("--M", type=float, default=1e6, help="Big-M value when method=big_m")
    p.add_argument("--graph", action="store_true", help="Plot constraints and iso-profit (2 variables only)")
    args = p.parse_args()

    with open(args.json, "r") as f:
        # Parse floats as Decimal to avoid binary float artifacts, we'll convert to Fractions later
        cfg = json.load(f, parse_float=Decimal)

    # Resolve sense priority: CLI (if provided) overrides JSON; else fallback to JSON->max.
    sense = args.sense
    if args.min:
        sense = "min"
    if sense is None:
        if "maximize" in cfg:
            maximize = bool(cfg["maximize"])
        else:
            maximize = True
    else:
        maximize = (sense == "max")

    lp = LP(
        c=cfg["c"],
        A=cfg["A"],
        b=cfg["b"],
        senses=cfg["senses"],
        maximize=maximize,
    )

    res = solve(lp, method=args.method, verbose=not args.no_verbose, M=args.M)

    print("\n=== Result ===")
    print("Status:", res.status)
    if res.status == "optimal":
        print("Optimal value:", fmt_out(res.optimal_value))
        if res.solution is not None:
            print("Solution x:", [fmt_out(v) for v in res.solution])
    print("Iterations:", res.iterations)
    print("Method:", res.method)
    # Infinite many solutions notice
    if res.status == "optimal" and isinstance(res.details, dict) and res.details.get('alternate_optimal'):
        print("Note: Infinite many optimal solutions (alternate optimal).")
        alt_vars = res.details.get('alt_zero_rc_vars') or []
        if alt_vars:
            print("Zero reduced-cost nonbasic vars:", alt_vars)
    if args.graph:
        graph(lp, res)

if __name__ == "__main__":
    main()
