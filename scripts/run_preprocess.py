"""Entry point — build the long-tailed train/val split from data/Images/.

Runs: data/Images/ (ImageFolder) -> data/preprocessed_dataset.zip

Execute from the repository root:
    python scripts/run_preprocess.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.preprocessing import main

if __name__ == "__main__":
    main()
