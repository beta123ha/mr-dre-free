"""Compatibility runner using the single-question pipeline."""
from mrdre_baseline.runtime.runner import FEEDBACK, main

def chunk_targets(n: int, batch: int) -> list[int]:
    """Retained for compatibility with tests and old scripts."""
    if not 1 <= n <= 65 or batch < 0:
        raise ValueError("count 1..65, batch nonnegative")
    return [n] if batch == 0 else [min(i, n) for i in range(batch, n + batch, batch)]

if __name__ == "__main__":
    raise SystemExit(main())
