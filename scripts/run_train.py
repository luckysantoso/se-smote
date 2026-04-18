"""Entry point — Phase 1: train the Super-Encoder on the preprocessed split.

Runs: data/preprocessed_dataset.zip -> best_super_encoder_model.pth

Execute from the repository root:
    python scripts/run_train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from training.train import main

if __name__ == "__main__":
    main()
