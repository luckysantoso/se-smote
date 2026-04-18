"""Entry point — Phase 2: SMOTE in latent space + t-SNE + decoded synthetic images.

Runs: best_super_encoder_model.pth + data/preprocessed_dataset.zip
   -> latent_features_dataset.zip, decoded_synthetic_images.zip

Execute from the repository root:
    python scripts/run_oversample.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oversampling import main

if __name__ == "__main__":
    main()
