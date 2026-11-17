"""Backward-compatible CLI import of the packaged MR DRE metric aggregator."""
from mrdre_baseline.runtime.metrics import (AGENT, DATASET, SETTING_NAMES, records,
                                            break_rate, incorporation_content, atomic_csv,
                                            collect, main)

if __name__ == "__main__":
    main()
