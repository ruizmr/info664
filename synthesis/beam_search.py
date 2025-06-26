import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np

# Import AST nodes from search
from .search import (
    Expr,
    Const,
    Var,
    Binary,
    Scale,
    Round,
    Pred,
    ReturnStmt,
    IfStmt,
    COMP_OPS,
    generate_terms,
    COMMUTATIVE_OPS,
    CONST_VALUES,
    VAR_NAMES,
    MULTIPLIERS,
    ROUND_P,
)


@dataclass
class Candidate:
    stmt: Any  # Stmt
    output: np.ndarray  # Vector predictions on samples
    error: float  # MAE wrt y_true


class BeamSynthesiser:
    def __init__(self, X: Dict[str, np.ndarray], y: np.ndarray, beam_width: int = 800):
        self.X = X  # dict with 'd','m','r'
        self.y = y.astype(np.float32)
        self.n = len(y)
        self.beam_width = beam_width

        # Maintain a working subset of indices for quick evaluation (CEGIS-style).
        self.eval_idx = np.random.choice(self.n, size=min(200, self.n), replace=False)

        self.expr_cache: Dict[str, np.ndarray] = {}

        # ---- Mine existing model for constants/expressions ----
        try:
            import joblib  # type: ignore
            from calculate import calculate_reimbursement  # noqa: F401

            state = joblib.load('model_state.pkl')
            surrogate_tree = state['surrogate_tree']
            formulas = state['formulas']
            final_formula_features = state['final_formula_features']

            # Extract split thresholds from the fitted decision tree. In scikit-learn
            # these live under `tree_.threshold`; values <= 0 or == -2 are sentinel
            # placeholders for non-splitting nodes, so we keep strictly positive ones.
            self.mined_thresholds = [round(float(t), 2) for t in surrogate_tree.tree_.threshold if t > 0]

            # leaf-level expressions
            self.mined_leaf_stmts: List[ReturnStmt] = []
            for leaf_id, obj in formulas.items():
                if isinstance(obj, float):
                    self.mined_leaf_stmts.append(ReturnStmt(Const(round(obj, 2))))
                else:
                    # assume sklearn Pipeline(poly,ridge)
                    ridge = obj.named_steps['ridge']
                    coefs = ridge.coef_
                    intercept = ridge.intercept_
                    # build expression: intercept + sum(coef_i * var_i)
                    expr: Expr = Const(round(intercept, 2))
                    for w, feat_name in zip(coefs, final_formula_features):
                        if abs(w) < 1e-4:
                            continue
                        term_expr = Var(feat_name) if feat_name in VAR_NAMES else None
                        if term_expr is None:
                            continue
                        expr = Binary('+', expr, Scale('*', term_expr, round(float(w), 2)))
                    self.mined_leaf_stmts.append(ReturnStmt(expr))
        except Exception as e:
            print('[Beam] mining failed', e)
            self.mined_thresholds = []
            self.mined_leaf_stmts = []

    # ---------------- expression evaluation -----------------
    def eval_expr(self, e: Expr) -> np.ndarray:
        key = str(e)
        if key in self.expr_cache:
            return self.expr_cache[key]

        if isinstance(e, Const):
            col = np.full(self.n, e.value, dtype=float)
        elif isinstance(e, Var):
            col = self.X[e.name]
        elif isinstance(e, Binary):
            l = self.eval_expr(e.left)
            r = self.eval_expr(e.right)
            if e.op == '+':
                col = l + r
            elif e.op == '-':
                col = l - r
            elif e.op == 'max':
                col = np.maximum(l, r)
            elif e.op == 'min':
                col = np.minimum(l, r)
            else:
                raise ValueError(e.op)
        elif isinstance(e, Scale):
            base = self.eval_expr(e.expr)
            col = base * e.k if e.op == '*' else base / e.k
        elif isinstance(e, Round):
            base = self.eval_expr(e.expr)
            col = np.round(base, e.p)
        else:
            raise TypeError(type(e))

        # Store as float32 to cut memory usage by half.
        self.expr_cache[key] = col.astype(np.float32)
        return col

    def eval_pred(self, p: Pred) -> np.ndarray:
        l = self.eval_expr(p.left)
        r = self.eval_expr(p.right)
        return np.asarray(COMP_OPS[p.op](l, r))  # type: ignore[arg-type]

    def eval_stmt(self, stmt: Any) -> np.ndarray:
        if isinstance(stmt, ReturnStmt):
            return self.eval_expr(stmt.expr)
        elif isinstance(stmt, IfStmt):
            cond = self.eval_pred(stmt.pred)
            t_val = self.eval_stmt(stmt.then_branch)
            e_val = self.eval_stmt(stmt.else_branch)
            return np.where(cond, t_val, e_val)
        else:
            raise TypeError(type(stmt))

    # -----------------------------------------------------------------
    def mae(self, col: np.ndarray) -> float:
        idx = self.eval_idx
        return float(np.mean(np.abs(col[idx] - self.y[idx])))

    # -----------------------------------------------------------------
    def initial_beam(self) -> List[Candidate]:
        # Include mined constants
        extra_consts = [Const(c) for c in self.mined_thresholds]
        terms = generate_terms() + extra_consts
        candidates: List[Candidate] = []
        for t in terms:
            stmt = ReturnStmt(t)
            out = self.eval_stmt(stmt)
            err = self.mae(out)
            candidates.append(Candidate(stmt, out, err))

        # seed with mined leaf expressions
        for stmt in self.mined_leaf_stmts:
            out = self.eval_stmt(stmt)
            err = self.mae(out)
            candidates.append(Candidate(stmt, out, err))
        candidates.sort(key=lambda c: c.error)
        return candidates[: self.beam_width]

    # -----------------------------------------------------------------
    def grow_beam(self, beam: List[Candidate], max_depth: int) -> List[Candidate]:
        new_candidates: List[Candidate] = []

        # Create binary ops between top beam expressions
        exprs = [c.stmt.expr for c in beam if isinstance(c.stmt, ReturnStmt)]
        for i, left in enumerate(exprs):
            for right in exprs[i:]:
                for op in ['+', '-', 'max', 'min']:
                    if op in COMMUTATIVE_OPS and str(left) > str(right):
                        continue
                    bin_expr = Binary(op, left, right)
                    stmt = ReturnStmt(bin_expr)
                    out = self.eval_expr(bin_expr)
                    err = self.mae(out)
                    new_candidates.append(Candidate(stmt, out, err))

        # scale and round expansions
        for cand in beam:
            expr = cand.stmt.expr if isinstance(cand.stmt, ReturnStmt) else None
            if expr is None:
                continue
            for mult in MULTIPLIERS:
                se = Scale('*', expr, mult)
                out = self.eval_expr(se)
                new_candidates.append(Candidate(ReturnStmt(se), out, self.mae(out)))
            for p in ROUND_P:
                re = Round(expr, p)
                out = self.eval_expr(re)
                new_candidates.append(Candidate(ReturnStmt(re), out, self.mae(out)))

        # combine then/else of top few into if using simple predicates (<,>)
        simple_preds_ops = ['<', '>', '>=', '<=']
        top_expr_terms = exprs[:20]
        for left in top_expr_terms:
            for right in top_expr_terms:
                if str(left) == str(right):
                    continue
                for op in simple_preds_ops:
                    pred = Pred(left, op, right)
                    cond_vec = self.eval_pred(pred)
                    # skip trivial all-true or all-false
                    if cond_vec.all() or (~cond_vec).all():
                        continue
                    then_stmt = ReturnStmt(left)
                    else_stmt = ReturnStmt(right)
                    if_stmt = IfStmt(pred, then_stmt, else_stmt)
                    out = self.eval_stmt(if_stmt)
                    new_candidates.append(Candidate(if_stmt, out, self.mae(out)))

        # merge & select best beam_width
        merged = beam + new_candidates
        merged.sort(key=lambda c: c.error)
        unique: Dict[str, Candidate] = {}
        for c in merged:
            k = str(c.stmt)
            if k not in unique:
                unique[k] = c
            if len(unique) >= self.beam_width:
                break
        return list(unique.values())

    # -----------------------------------------------------------------
    def search(self, max_iters: int = 10, target_error: float = 0.01):
        beam = self.initial_beam()
        best = beam[0]
        print(f"[Beam] Iter0 best MAE={best.error:.3f}")

        for it in range(1, max_iters + 1):
            beam = self.grow_beam(beam, max_depth=3)
            best = beam[0]
            print(f"[Beam] Iter{it} best MAE={best.error:.3f}")

            # Expand evaluation subset at milestones to refine accuracy
            if it in {5, 10, 15}:
                new_sz = min(self.n, len(self.eval_idx) * 2)
                self.expand_subset(new_sz)

            # Dynamically shrink beam once we have low error to save memory
            if best.error < 5 and self.beam_width > 400:
                self.beam_width = 400
            if best.error < 1 and self.beam_width > 200:
                self.beam_width = 200

            if best.error < target_error and len(self.eval_idx) == self.n:
                print("[Beam] Target reached.")
                return best.stmt
        return best.stmt

    # Allow caller to increase sample size progressively
    def expand_subset(self, new_size: int) -> None:
        if new_size >= self.n:
            self.eval_idx = np.arange(self.n)
        else:
            keep = set(self.eval_idx)
            # sample additional indices to reach new_size
            needed = new_size - len(self.eval_idx)
            if needed > 0:
                extra = np.random.choice([i for i in range(self.n) if i not in keep], size=needed, replace=False)
                self.eval_idx = np.concatenate([self.eval_idx, extra])


def load_public() -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    with open('public_cases.json', 'r') as f:
        raw = json.load(f)
    d = np.array([e['input']['trip_duration_days'] for e in raw], dtype=float)
    m = np.array([e['input']['miles_traveled'] for e in raw], dtype=float)
    r = np.array([e['input']['total_receipts_amount'] for e in raw], dtype=float)
    y = np.array([e['expected_output'] for e in raw], dtype=float)
    X = {'d': d, 'm': m, 'r': r}
    return X, y


def main():
    X, y = load_public()
    # Start with a broad beam so we don't prematurely prune good candidates.
    synth = BeamSynthesiser(X, y, beam_width=800)
    # Allow more search iterations now that we have richer constant space
    best_stmt = synth.search(max_iters=20)
    from .emit import write_program
    write_program(best_stmt)
    print('legacy_reimburse.py written')


if __name__ == '__main__':
    main()