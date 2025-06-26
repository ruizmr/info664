def legacy_reimburse(d: int, m: int, r: float) -> float:
    """Deterministic reimbursement model (synthesised).
    """
    if min(1676.48, r) > (720.63 + m):
        return round(min(1676.48, r), 2)
    else:
        return round((720.63 + m), 2)

