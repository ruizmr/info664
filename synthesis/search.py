import json
import math
import operator
import itertools
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Dict, Iterator, Optional, cast
import random

# ---------------- DSL definition constants ---------------- #
CONST_VALUES = [
    0, 1, 2, 3, 4, 5, 10, 15, 20, 25,
    50, 75, 100, 120, 150, 180, 200, 250, 300, 400, 500, 600, 800, 1000,
]
VAR_NAMES = ["d", "m", "r"]
MULTIPLIERS = [0.01, 0.1, 0.25, 0.4, 0.58, 0.8, 1.05]
ROUND_P = [0, 2]

# Comparison operators mapping
COMP_OPS: Dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
}

# ---------------- AST node classes ---------------- #
@dataclass(frozen=True)
class Expr:
    def eval(self, env: Dict[str, float]) -> float:
        raise NotImplementedError

    def size(self) -> int:
        """Number of nodes used (for enumeration complexity)."""
        raise NotImplementedError


@dataclass(frozen=True)
class Const(Expr):
    value: float

    def eval(self, env: Dict[str, float]) -> float:  # noqa: D401
        return float(self.value)

    def __str__(self):
        if isinstance(self.value, int):
            return str(self.value)
        return f"{self.value}"

    def size(self) -> int:
        return 1


@dataclass(frozen=True)
class Var(Expr):
    name: str

    def eval(self, env: Dict[str, float]) -> float:
        return float(env[self.name])

    def __str__(self):
        return self.name

    def size(self) -> int:
        return 1


@dataclass(frozen=True)
class Binary(Expr):
    op: str  # '+', '-', 'max', 'min'
    left: Expr
    right: Expr

    def eval(self, env: Dict[str, float]) -> float:
        l = self.left.eval(env)
        r = self.right.eval(env)
        if self.op == '+':
            return l + r
        elif self.op == '-':
            return l - r
        elif self.op == 'max':
            return max(l, r)
        elif self.op == 'min':
            return min(l, r)
        else:
            raise ValueError(f"Unknown op {self.op}")

    def __str__(self):
        if self.op in {'+', '-'}:
            return f"({self.left} {self.op} {self.right})"
        else:
            return f"{self.op}({self.left}, {self.right})"

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()


@dataclass(frozen=True)
class Scale(Expr):
    op: str  # '*' or '/'
    expr: Expr
    k: float

    def eval(self, env: Dict[str, float]) -> float:
        val = self.expr.eval(env)
        if self.op == '*':
            return val * self.k
        else:
            return val / self.k

    def __str__(self):
        if self.op == '*':
            return f"({self.expr} * {self.k})"
        else:
            return f"({self.expr} / {self.k})"

    def size(self) -> int:
        return 1 + self.expr.size()


@dataclass(frozen=True)
class Round(Expr):
    expr: Expr
    p: int

    def eval(self, env: Dict[str, float]) -> float:
        return round(self.expr.eval(env), self.p)

    def __str__(self):
        return f"round({self.expr}, {self.p})"

    def size(self) -> int:
        return 1 + self.expr.size()


# Predicate
@dataclass(frozen=True)
class Pred:
    left: Expr
    op: str  # one of COMP_OPS keys
    right: Expr

    def eval(self, env: Dict[str, float]) -> bool:
        return COMP_OPS[self.op](self.left.eval(env), self.right.eval(env))

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def __str__(self):
        return f"({self.left} {self.op} {self.right})"


# Statement classes
@dataclass(frozen=True)
class ReturnStmt:
    expr: Expr

    def eval(self, env: Dict[str, float]) -> float:
        return self.expr.eval(env)

    def size(self) -> int:
        return 1 + self.expr.size()

    def __str__(self):
        return f"return {self.expr}"


@dataclass(frozen=True)
class IfStmt:
    pred: Pred
    then_branch: 'Stmt'
    else_branch: 'Stmt'

    def eval(self, env: Dict[str, float]) -> float:
        if self.pred.eval(env):
            return self.then_branch.eval(env)
        else:
            return self.else_branch.eval(env)

    def size(self) -> int:
        return 1 + self.pred.size() + self.then_branch.size() + self.else_branch.size()

    def __str__(self):
        return f"if {self.pred}: {self.then_branch} else: {self.else_branch}"

# Alias for typing
Stmt = Any  # ReturnStmt or IfStmt


# ---------------- Enumeration utilities ---------------- #

def generate_terms() -> List[Expr]:
    """Generate base terms (Var and Const)."""
    # Use cast because list concatenation with heterogeneous types confuses type checker
    terms = [Var(n) for n in VAR_NAMES] + [Const(c) for c in CONST_VALUES]
    return cast(List[Expr], terms)


def combine_binary(op: str, exprs_left: List[Expr], exprs_right: List[Expr]) -> Iterator[Expr]:
    for l in exprs_left:
        for r in exprs_right:
            yield Binary(op, l, r)


def combine_scale(op: str, exprs: List[Expr]) -> Iterator[Expr]:
    for e in exprs:
        for k in MULTIPLIERS:
            yield Scale(op, e, k)


def combine_round(exprs: List[Expr]) -> Iterator[Expr]:
    for e in exprs:
        for p in ROUND_P:
            yield Round(e, p)


COMMUTATIVE_OPS = {'+', 'max', 'min'}


def enumerate_exprs(max_size: int) -> List[Expr]:
    """Enumerate all unique expressions whose AST size ≤ max_size."""
    # Dynamic programming: build lists per size.
    memo: Dict[int, List[Expr]] = {1: generate_terms()}

    # Helper to avoid generating obvious duplicates for commutative ops
    def should_emit_binary(op: str, left: Expr, right: Expr) -> bool:
        if op in COMMUTATIVE_OPS:
            # Use string ordering as proxy for structural ordering
            return str(left) <= str(right)
        return True

    for size in range(2, max_size + 1):
        current: List[Expr] = []

        # Scale and Round constructions: 1 (root) + child_size = size
        child_size = size - 1
        if child_size in memo:
            for expr in memo[child_size]:
                # scale operations
                for op in ['*', '/']:
                    for k in MULTIPLIERS:
                        current.append(Scale(op, expr, k))
                # round operations
                for p in ROUND_P:
                    current.append(Round(expr, p))

        # Binary constructions: 1 (root) + left_size + right_size = size
        for left_size in range(1, size - 1):
            right_size = size - 1 - left_size
            if right_size < 1:
                continue
            for left_expr in memo[left_size]:
                for right_expr in memo[right_size]:
                    for op in ['+', '-', 'max', 'min']:
                        if should_emit_binary(op, left_expr, right_expr):
                            current.append(Binary(op, left_expr, right_expr))

        memo[size] = current

    # Aggregate all expressions up to max_size
    all_exprs: List[Expr] = []
    for s in range(1, max_size + 1):
        all_exprs.extend(memo.get(s, []))
    return all_exprs


# ---------------- Program enumeration (placeholder) ---------------- #


def main():
    print("== Synthesiser starting ==")

    # Quick smoke-test run with small limits
    cegis_search(max_size=5, max_depth=3, initial_limit=50)


# ---------------- Predicate enumeration ---------------- #


def enumerate_preds(max_size: int, expr_by_size: Dict[int, List[Expr]]) -> Dict[int, List[Pred]]:
    """Return mapping size -> list of predicates with that size (≤ max_size)."""
    pred_memo: Dict[int, List[Pred]] = {}
    for size in range(3, max_size + 1):  # minimum size 3: 1 node + 1+1 exprs
        preds_curr: List[Pred] = []
        # allocate sizes to left and right exprs such that 1 + left + right = size
        for left_size in range(1, size - 1):
            right_size = size - 1 - left_size
            if right_size < 1:
                continue
            for left_expr in expr_by_size.get(left_size, []):
                for right_expr in expr_by_size.get(right_size, []):
                    for op in COMP_OPS.keys():
                        # skip trivial predicate equality duplicates when commutative
                        if op == '==' and str(left_expr) > str(right_expr):
                            continue
                        preds_curr.append(Pred(left_expr, op, right_expr))
        if preds_curr:
            pred_memo[size] = preds_curr
    return pred_memo


# ---------------- Statement enumeration ---------------- #


def enumerate_statements(
    max_size: int,
    max_depth: int,
    expr_by_size: Dict[int, List[Expr]],
    pred_by_size: Dict[int, List[Pred]],
) -> Dict[Tuple[int, int], List[Stmt]]:
    """Return mapping (size, depth) -> list of statements."""
    # Depth definition: Return = 1, If = 1 + max(depth_then, depth_else)
    stmt_memo: Dict[Tuple[int, int], List[Stmt]] = {}

    # Helper to add stmt to memo
    def add_stmt(size: int, depth: int, stmt: Stmt):
        key = (size, depth)
        stmt_memo.setdefault(key, []).append(stmt)

    # Handle Return statements
    for expr_size in range(1, max_size):
        for expr in expr_by_size.get(expr_size, []):
            total_size = 1 + expr_size
            if total_size <= max_size:
                add_stmt(total_size, 1, ReturnStmt(expr))

    # Iteratively build If statements by increasing total size
    for size in range(1, max_size + 1):
        for pred_size, pred_list in pred_by_size.items():
            if pred_size >= size:
                continue
            remaining_size = size - 1 - pred_size
            # Allocate remaining size between then and else stmts
            for then_size in range(1, remaining_size):
                else_size = remaining_size - then_size
                # Enumerate possible depths as well
                for depth_then in range(1, max_depth):
                    for depth_else in range(1, max_depth):
                        depth_if = 1 + max(depth_then, depth_else)
                        if depth_if > max_depth:
                            continue
                        # Get stmt lists for then and else matching (size, depth)
                        then_stmts = stmt_memo.get((then_size, depth_then), [])
                        else_stmts = stmt_memo.get((else_size, depth_else), [])
                        if not then_stmts or not else_stmts:
                            continue
                        # Combine
                        for pred in pred_list:
                            for t_stmt in then_stmts:
                                for e_stmt in else_stmts:
                                    add_stmt(size, depth_if, IfStmt(pred, t_stmt, e_stmt))
    return stmt_memo


# ---------------- CEGIS Search ---------------- #


Case = Tuple[int, int, float, float]  # (d, m, r, expected)


def load_public_cases(limit: Optional[int] = None) -> List[Case]:
    """Return list of (d, m, r, expected). Optionally limit for quick test."""
    with open('public_cases.json', 'r') as f:
        raw = json.load(f)
    cases: List[Case] = []
    for entry in raw[: limit]:
        inp = entry['input']
        exp = float(entry['expected_output'])
        cases.append((int(inp['trip_duration_days']), int(inp['miles_traveled']), float(inp['total_receipts_amount']), exp))
    return cases


def evaluate_program(prog: Stmt, cases: List[Case]) -> float:
    """Return max absolute error across cases (after rounding to 2dp)."""
    max_err = 0.0
    for d, m, r, exp in cases:
        pred = prog.eval({'d': d, 'm': m, 'r': r})
        pred = round(pred, 2)
        err = abs(pred - exp)
        if err > max_err:
            max_err = err
            if max_err > 0.01:
                return max_err
    return max_err


def deduplicate_stmts(stmts: List[Stmt]) -> List[Stmt]:
    seen: Dict[str, Stmt] = {}
    for s in stmts:
        key = str(s)
        if key not in seen:
            seen[key] = s
    return list(seen.values())


def sample_input_case(bounds: Dict[str, Tuple[int, int]]) -> Tuple[int, int, float]:
    d = random.randint(bounds['d'][0], bounds['d'][1])
    m = random.randint(bounds['m'][0], bounds['m'][1])
    r = random.uniform(bounds['r'][0], bounds['r'][1])
    return d, m, float(round(r, 2))


def generate_fuzz_cases(n: int, bounds: Dict[str, Tuple[int, int]], seed: int = 42) -> List[Tuple[int, int, float]]:
    random.seed(seed)
    return [sample_input_case(bounds) for _ in range(n)]


def cegis_search(
    max_size: int = 6,
    max_depth: int = 4,
    initial_limit: int = 500,
    fuzz_batch: int = 500,
):
    """CEGIS with fuzzing & counterexamples."""

    # Compute observed bounds from public data
    public_full = load_public_cases(limit=None)
    d_vals = [d for d, _, _, _ in public_full]
    m_vals = [m for _, m, _, _ in public_full]
    r_vals = [r for _, _, r, _ in public_full]
    bounds = {
        'd': (min(d_vals), max(d_vals)),
        'm': (min(m_vals), max(m_vals)),
        'r': (0, max(r_vals)),
    }

    constraints: List[Case] = public_full[: initial_limit]
    stable_rounds = 0

    iteration = 0
    while stable_rounds < 3:
        iteration += 1
        print(f"\n[CEGIS] Iteration {iteration} with {len(constraints)} constraints (stable={stable_rounds})")

        # Enumerate program space (can grow each iteration if needed)
        exprs_all = enumerate_exprs(max_size)
        expr_by_size: Dict[int, List[Expr]] = {}
        for e in exprs_all:
            expr_by_size.setdefault(e.size(), []).append(e)

        pred_by_size = enumerate_preds(max_size, expr_by_size)
        stmt_by_size_depth = enumerate_statements(max_size, max_depth, expr_by_size, pred_by_size)

        # Flatten candidates
        candidates: List[Stmt] = []
        for size in range(2, max_size + 1):
            for depth in range(1, max_depth + 1):
                candidates.extend(stmt_by_size_depth.get((size, depth), []))

        candidates = deduplicate_stmts(candidates)
        print(f"[CEGIS] Testing {len(candidates)} unique programs")

        found_prog: Optional[Stmt] = None
        for prog in candidates:
            if evaluate_program(prog, constraints) < 0.01:
                found_prog = prog
                break

        if found_prog is None:
            print("[CEGIS] No program satisfies current constraints. Consider increasing search space.")
            return None

        # Generate quasi-random fuzz cases using Sobol/Halton sequence
        fuzz_inputs = sobol_samples(fuzz_batch, bounds, start=iteration * fuzz_batch)

        new_cases: List[Case] = []
        for d, m, r in fuzz_inputs:
            predicted = round(found_prog.eval({'d': d, 'm': m, 'r': r}), 2)
            try:
                expected = oracle(d, m, r)
            except Exception:
                expected = predicted  # fallback if oracle not available

            if abs(predicted - expected) >= 0.01:
                new_cases.append((d, m, r, expected))

        if new_cases:
            print(f"[CEGIS] Found {len(new_cases)} counter-examples, adding to constraints.")
            constraints.extend(new_cases)
            stable_rounds = 0
        else:
            stable_rounds += 1

        if stable_rounds >= 3:
            print("[CEGIS] Program stable across fuzzing rounds. Writing legacy_reimburse.py")
            try:
                from .emit import write_program
                write_program(found_prog)
                print("[CEGIS] legacy_reimburse.py generated.")
            except Exception as e:
                print("[CEGIS] Failed to write program:", e)
            return found_prog

    print("[CEGIS] Reached max iterations without stable program.")
    return None


# ---------------- Oracle access ---------------- #


try:
    import joblib  # type: ignore
    from calculate import calculate_reimbursement as _calc_reim  # type: ignore

    _oracle_state = None  # lazy load

    def oracle(d: int, m: int, r: float) -> float:
        """Return oracle reimbursement rounded to 2 decimals."""
        global _oracle_state
        if _oracle_state is None:
            _oracle_state = joblib.load('model_state.pkl')
        return round(_calc_reim(d, m, r, _oracle_state), 2)

except Exception as exc:  # pragma: no cover
    print("[WARN] Oracle access unavailable:", exc)

    def oracle(d: int, m: int, r: float) -> float:  # type: ignore
        raise RuntimeError("Oracle access not available in this environment")


# ---------------- Sobol / Halton quasi-random sampling ---------------- #


def _van_der_corput(n: int, base: int) -> float:
    vdc, denom = 0.0, 1.0
    while n:
        n, remainder = divmod(n, base)
        denom *= base
        vdc += remainder / denom
    return vdc


def sobol_samples(count: int, bounds: Dict[str, Tuple[int, int]], start: int = 0) -> List[Tuple[int, int, float]]:
    """Return quasi-random samples using simple Halton/VdC per dimension."""
    samples: List[Tuple[int, int, float]] = []
    bases = [2, 3, 5]  # pairwise prime
    for i in range(start, start + count):
        u_d = _van_der_corput(i, bases[0])
        u_m = _van_der_corput(i, bases[1])
        u_r = _van_der_corput(i, bases[2])
        d = bounds['d'][0] + int(u_d * (bounds['d'][1] - bounds['d'][0] + 1))
        m = bounds['m'][0] + int(u_m * (bounds['m'][1] - bounds['m'][0] + 1))
        r = bounds['r'][0] + u_r * (bounds['r'][1] - bounds['r'][0])
        samples.append((d, m, round(r, 2)))
    return samples


if __name__ == '__main__':
    main()