"""Compatibility entrypoint; shared setup works on Kaggle, Colab and local."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mrdre_baseline.runtime.bootstrap import main
if __name__ == "__main__":
    raise SystemExit(main(["--profile", "kaggle", *sys.argv[1:]]))
