import io
import os
import zipfile
import pickle
from collections import Counter
from typing import Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

from super_encoder import SuperEncoder
from config import (
    PATHS, TRAIN,
    get_device, set_seed
)

# ---------------------------
# Dataset wrapper sederhana
# ---------------------------
class TransformedSubset(Dataset):
    """Menerapkan transform ke Subset (hasil pickle dari preprocessing)."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


# ---------------------------
# G-SMOTE di ruang laten
# ---------------------------
def G_SM1(z_latent: np.ndarray, n_to_sample: int, cl: int, k_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Generative SMOTE sederhana (linear interpolation) di ruang laten."""
    if len(z_latent) <= k_neighbors:
        k_neighbors = max(0, len(z_latent) - 1)

    if k_neighbors < 1:
        base_idx = np.random.choice(len(z_latent), size=n_to_sample)
        return z_latent[base_idx], np.full(n_to_sample, cl, dtype=int)

    nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, n_jobs=-1).fit(z_latent)
    _, ind = nn_model.kneighbors(z_latent)
    base_idx = np.random.choice(len(z_latent), n_to_sample)
    neigh_offset = np.random.randint(1, k_neighbors + 1, n_to_sample)
    z_base = z_latent[base_idx]
    z_neigh = z_latent[ind[base_idx, neigh_offset]]
    z_synth = z_base + np.random.rand(n_to_sample, 1) * (z_neigh - z_base)
    return z_synth, np.full(n_to_sample, cl, dtype=int)


# ---------------------------
# Ekstraksi & Dekode fitur laten
# ---------------------------
def extract_latent_features(model: SuperEncoder, data_subset, transform, batch_size: int, device: str):
    """Ekstraksi fitur dari encoder (atau method encode) pada subset."""
    ds = TransformedSubset(data_subset, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=TRAIN.num_workers, pin_memory=True)

    feats, labels = [], []
    model.eval()
    with torch.inference_mode():
        for images, y in tqdm(loader, desc="Extracting Latent Features"):
            images = images.to(device, non_blocking=True)
            # pakai encode() jika tersedia (bisa L2-normalize tergantung super_encoder)
            if hasattr(model, "encode"):
                z = model.encode(images)
            else:
                z = model.encoder(images)
            feats.append(z.detach().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def decode_latent_vectors(model: SuperEncoder, latent_vectors: np.ndarray, batch_size: int, device: str) -> torch.Tensor:
    """Decode vektor laten menjadi citra (keluaran decoder ∈ [0,1] karena Sigmoid)."""
    outs = []
    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(0, len(latent_vectors), batch_size), desc="Decoding Images"):
            batch_z = torch.from_numpy(latent_vectors[i:i+batch_size]).float().to(device)
            x_hat = model.decoder(batch_z)  # [B, 3, H, W] in [0,1]
            outs.append(x_hat.cpu())
    return torch.cat(outs, dim=0)  # [N,3,H,W] in [0,1]


# ---------------------------
# Visualisasi t-SNE
# ---------------------------
def visualize_tsne(features, labels, class_names, title, markers_info=None):
    print(f"\n--- {title} ---")

    n_samples_for_plot = 7000
    if len(features) > n_samples_for_plot:
        print(f"Data terlalu besar ({len(features)}), mengambil {n_samples_for_plot} sampel acak untuk visualisasi.")
        indices = np.random.permutation(len(features))[:n_samples_for_plot]
        features_to_plot = features[indices]
        labels_to_plot = labels[indices]
        markers_to_plot = [markers_info[i] for i in indices] if markers_info else None
    else:
        features_to_plot = features
        labels_to_plot = labels
        markers_to_plot = markers_info

    print(f"Menjalankan t-SNE pada {len(features_to_plot)} sampel...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000, learning_rate='auto', init='pca')
    tsne_results = tsne.fit_transform(features_to_plot)
    print("t-SNE selesai.")

    df = pd.DataFrame({
        "tsne-2d-one": tsne_results[:, 0],
        "tsne-2d-two": tsne_results[:, 1],
        "class": [class_names[l] for l in labels_to_plot]
    })

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(16, 12))
    palette = sns.color_palette("bright", n_colors=len(class_names))

    plot_params = dict(data=df, x="tsne-2d-one", y="tsne-2d-two", hue="class", palette=palette, s=100, alpha=0.7)

    if markers_to_plot:
        df["data_type"] = markers_to_plot
        plot_params["style"] = "data_type"
        plot_params["markers"] = {"Original": "o", "Synthetic": "*"}

    sns.scatterplot(**plot_params)
    plt.title(title, fontsize=18)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Legenda", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ---------------------------
# Main
# ---------------------------
def main():
    # ---- Config & seed ----
    device = get_device()
    set_seed(TRAIN.seed)

    PREPROCESSED_DATA_ZIP = PATHS.preprocessed_zip
    MODEL_PATH = PATHS.output_model_path
    OUTPUT_FEATURES_ZIP = "latent_features_dataset.zip"
    OUTPUT_IMAGES_ZIP = "decoded_synthetic_images.zip"

    BATCH_SIZE = TRAIN.batch_size
    IMAGE_SIZE = TRAIN.image_size
    LATENT_DIM = TRAIN.latent_dim

    print(f"Device: {device}")
    print(f"Memuat data latih dari {PREPROCESSED_DATA_ZIP} ...")

    # ---- Load subset & class map dari zip preprocessing ----
    with zipfile.ZipFile(PREPROCESSED_DATA_ZIP, "r") as zf:
        with zf.open("train_subset.pkl", "r") as f:
            train_subset = pickle.load(f)
        with zf.open("class_to_idx.pkl", "r") as f:
            class_to_idx = pickle.load(f)

    class_names = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # ---- Build & load model ----
    print(f"Memuat model dari {MODEL_PATH} ...")
    model = SuperEncoder(
        latent_dim=LATENT_DIM,
        num_classes=num_classes,
        dropout_prob=TRAIN.dropout_prob,
        l2_normalize_latent=TRAIN.l2_normalize_latent,
        use_adaptive_pool=TRAIN.use_adaptive_pool
    ).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ---- Transform (samakan dengan training) ----
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # [-1,1]
    ])

    # ---- Ekstraksi fitur laten asli ----
    X_train_latent, y_train_original = extract_latent_features(
        model, train_subset, val_transform, BATCH_SIZE, device
    )

    visualize_tsne(
        X_train_latent, y_train_original, class_names,
        title="Visualisasi Ruang Laten SEBELUM Oversampling"
    )

    # ---- Oversampling G-SMOTE ----
    print("\nMelakukan oversampling dengan G-SMOTE pada fitur laten ...")
    train_class_counts = Counter(y_train_original)
    max_class_count = max(train_class_counts.values())

    X_resampled_list = [X_train_latent]
    y_resampled_list = [y_train_original]
    all_synth_latents = []

    for cls, count in train_class_counts.items():
        n_needed = max_class_count - count
        if n_needed <= 0:
            continue
        print(f"Kelas {cls} ({class_names[cls]}): Membuat {n_needed} sampel sintetis ...")
        Z_cls = X_train_latent[y_train_original == cls]
        z_synth, y_synth = G_SM1(Z_cls, n_needed, cls)
        X_resampled_list.append(z_synth)
        y_resampled_list.append(y_synth)
        if len(z_synth) > 0:
            all_synth_latents.append(z_synth)

    X_train_resampled = np.vstack(X_resampled_list)
    y_train_resampled = np.concatenate(y_resampled_list)

    markers = (['Original'] * len(y_train_original)
               + ['Synthetic'] * (len(y_train_resampled) - len(y_train_original)))
    title_after = "Visualisasi Ruang Laten SETELAH Oversampling\n(Titik=Asli, Bintang=Sintetis)"
    visualize_tsne(
        X_train_resampled, y_train_resampled, class_names,
        title=title_after, markers_info=markers
    )

    # ---- Simpan fitur laten ----
    print(f"\nMenyimpan fitur laten ke {OUTPUT_FEATURES_ZIP} ...")
    with zipfile.ZipFile(OUTPUT_FEATURES_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        with zf.open("X_train_resampled.npy", "w") as f:
            np.save(f, X_train_resampled)
        with zf.open("y_train_resampled.npy", "w") as f:
            np.save(f, y_train_resampled)
        with zf.open("X_train_original_latent.npy", "w") as f:
            np.save(f, X_train_latent)
        with zf.open("y_train_original.npy", "w") as f:
            np.save(f, y_train_original)

    # ---- Decode gambar sintetis (opsional) ----
    if all_synth_latents:
        synth_latents_all = np.vstack(all_synth_latents)  # [Ns, latent_dim]
        decoded_images = decode_latent_vectors(model, synth_latents_all, BATCH_SIZE, device)  # [Ns,3,H,W] in [0,1]

        print(f"Menyimpan gambar sintetis yang telah di-dekode ke {OUTPUT_IMAGES_ZIP} ...")
        with zipfile.ZipFile(OUTPUT_IMAGES_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            with zf.open("decoded_synthetic_images.pt", "w") as f:
                torch.save(decoded_images, f)  # simpan tensor [0,1]
    else:
        print("Tidak ada latent sintetis yang perlu didekode.")

    print("\nProses oversampling selesai.")

if __name__ == "__main__":
    main()
