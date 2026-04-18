# config.py
from dataclasses import dataclass
from typing import Tuple
import torch
import random

# =========================
# PATH & PROJECT SETTINGS
# =========================
@dataclass
class PathConfig:
    # Root folder gambar untuk ImageFolder (hasil ekstraksi dataset)
    data_dir: str = "./data/Images"
    # Arsip hasil preprocessing (train/val subset, class_to_idx, referensi)
    preprocessed_zip: str = "./data/preprocessed_dataset.zip"
    # Lokasi penyimpanan model terbaik saat training
    output_model_path: str = "best_super_encoder_model.pth"


# =========================
# PREPROCESSING SETTINGS
# =========================
@dataclass
class PreprocessConfig:
    # Faktor ketidakseimbangan long-tailed (geometrik)
    imbalance_factor: float = 10.0
    # Label kelas yang dijadikan ekor (tail)
    tail_class_label: int = 2
    # Porsi validasi saat stratified split
    test_size: float = 0.2
    # Seed global untuk reproducibility
    random_state: int = 42
    # Tampilkan plot distribusi pada preprocessing
    show_plots: bool = True


# =========================
# TRAINING SETTINGS
# =========================
@dataclass
class TrainConfig:
    # Ukuran gambar yang masuk model (akan di-resize ke (image_size, image_size))
    image_size: int = 64
    # Epoh maksimum
    epochs: int = 200
    # Ukuran batch
    batch_size: int = 32
    # Worker DataLoader
    num_workers: int = 2
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    # Bobot loss
    alpha_sep: float = 0.1   # klasifikasi (CrossEntropy)
    beta_recon: float = 1.0  # rekonstruksi (MSE)
    # Early stopping
    early_stopping_patience: int = 15
    # Seed global
    seed: int = 42
    # Arsitektur SuperEncoder
    latent_dim: int = 4096
    dropout_prob: float = 0.5
    l2_normalize_latent: bool = True
    use_adaptive_pool: bool = True


# =========================
# GLOBAL SINGLETONS
# =========================
PATHS = PathConfig()
PRE = PreprocessConfig()
TRAIN = TrainConfig()


# =========================
# UTILS
# =========================
def get_device() -> str:
    """Return 'cuda' jika tersedia, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42) -> None:
    """Set global seed untuk reproducibility (CPU & CUDA)."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denormalize_to_unit(x_norm: torch.Tensor) -> torch.Tensor:
    """
    Ubah tensor dari [-1, 1] ke [0, 1] agar cocok dengan output Decoder (Sigmoid).
    Digunakan pada perhitungan reconstruction loss.
    """
    return (x_norm + 1.0) / 2.0


def log_core_configs() -> Tuple[str, int, int]:
    """
    Helper opsional untuk logging awal.
    Returns: (device, epochs, batch_size)
    """
    device = get_device()
    return device, TRAIN.epochs, TRAIN.batch_size
