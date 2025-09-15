import os
import torch
import numpy as np
import random
from torchvision import datasets
from torch.utils.data import Subset
from collections import Counter
from sklearn.model_selection import train_test_split
import zipfile
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PATHS, PRE, set_seed

def walk_through_dir(dir_path):
    """Mencetak struktur direktori."""
    print("\n--- Struktur Direktori Dataset Asli ---")
    for dirpath, dirnames, filenames in os.walk(dir_path):
        if dirpath != dir_path:
            print(f"Direktori '{os.path.basename(dirpath)}' memiliki {len(filenames)} gambar.")
    print("----------------------------------------")

def create_imbalanced_dataset(dataset, imbalance_factor=10, tail_class_label=2):
    """
    Membuat versi dataset yang tidak seimbang (long-tailed).
    Mengembalikan subset dan urutan kelas long-tailed.
    """
    class_indices = {i: [] for i in range(len(dataset.classes))}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    num_classes = len(dataset.classes)
    other_class_labels = [l for l in range(num_classes) if l != tail_class_label]
    other_class_counts = {l: len(class_indices[l]) for l in other_class_labels}
    sorted_other_classes = sorted(other_class_counts, key=other_class_counts.get, reverse=True)
    long_tail_order = sorted_other_classes + [tail_class_label]

    head_class_label = long_tail_order[0]
    n_max = len(class_indices[head_class_label])
    target_counts = {}
    for i, label in enumerate(long_tail_order):
        new_count = n_max * (imbalance_factor ** (-i / (num_classes - 1)))
        target_counts[label] = int(new_count)

    selected_indices = []
    for label, target_count in target_counts.items():
        count_to_sample = min(target_count, len(class_indices[label]))
        sampled_indices = np.random.choice(class_indices[label], size=count_to_sample, replace=False)
        selected_indices.extend(sampled_indices.tolist())

    return Subset(dataset, selected_indices), long_tail_order

def get_label_distribution(subset, original_dataset, class_names):
    """Mendapatkan distribusi label dari sebuah subset."""
    indices = subset.indices
    current_dataset = subset.dataset
    while isinstance(current_dataset, Subset):
        indices = [current_dataset.indices[i] for i in indices]
        current_dataset = current_dataset.dataset
    labels = [original_dataset.targets[i] for i in indices]
    counts = Counter(labels)
    dist_df = pd.DataFrame({
        'Kelas': [class_names.get(key, f"Unknown:{key}") for key in sorted(counts.keys())],
        'Jumlah': [counts[key] for key in sorted(counts.keys())]
    }).set_index('Kelas')
    return dist_df

def main():
    # --- Konfigurasi via config.py ---
    DATA_DIR = PATHS.data_dir
    OUTPUT_ZIP = PATHS.preprocessed_zip
    IMBALANCE_FACTOR = PRE.imbalance_factor
    TAIL_CLASS_LABEL = PRE.tail_class_label
    TEST_SIZE = PRE.test_size
    RANDOM_STATE = PRE.random_state

    # Seed
    set_seed(RANDOM_STATE)

    # --- Memuat Dataset Awal & Menampilkan Distribusi Asli ---
    print("Memuat dataset awal...")
    walk_through_dir(DATA_DIR)
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)
    class_to_idx = full_dataset.class_to_idx
    class_names = {v: k for k, v in class_to_idx.items()}
    print(f"\nPemetaan kelas ke indeks: {class_to_idx}")

    # Distribusi dataset asli
    print("\n--- Distribusi Dataset Asli (Sebelum Imbalancing) ---")
    original_counts = Counter(full_dataset.targets)
    original_dist_df = pd.DataFrame({
        'Kelas': [class_names.get(key) for key in sorted(original_counts.keys())],
        'Jumlah': [original_counts[key] for key in sorted(original_counts.keys())]
    }).set_index('Kelas')
    print(original_dist_df)
    print("--------------------------------------------------")

    # --- Membuat Dataset Tidak Seimbang ---
    print("\nMembuat dataset tidak seimbang (long-tailed)...")
    imbalanced_subset, long_tail_order = create_imbalanced_dataset(
        full_dataset, IMBALANCE_FACTOR, TAIL_CLASS_LABEL
    )
    print(f"Total sampel dalam subset tidak seimbang: {len(imbalanced_subset)}")
    print("Distribusi label pada subset tidak seimbang:")
    print(get_label_distribution(imbalanced_subset, full_dataset, class_names))

    # --- Pemisahan Data Latih dan Validasi (Stratified) ---
    print("\nMelakukan pemisahan data latih-validasi secara stratified...")
    # Penting: ambil label sesuai urutan elemen di imbalanced_subset (bukan urutan index global)
    labels = [full_dataset.targets[idx] for idx in imbalanced_subset.indices]
    indices = list(range(len(imbalanced_subset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE
    )
    train_subset = Subset(imbalanced_subset, train_idx)
    val_subset = Subset(imbalanced_subset, val_idx)

    # --- Visualisasi Distribusi ---
    print("\nMembuat visualisasi distribusi data...")
    train_dist_df = get_label_distribution(train_subset, full_dataset, class_names).reset_index()
    val_dist_df = get_label_distribution(val_subset, full_dataset, class_names).reset_index()
    train_dist_df['Set'] = 'Train'
    val_dist_df['Set'] = 'Validation'
    combined_df = pd.concat([train_dist_df, val_dist_df])

    long_tail_class_names = [class_names[i] for i in long_tail_order]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    sns.barplot(data=combined_df, x='Kelas', y='Jumlah', hue='Set', order=long_tail_class_names, palette='viridis')
    plt.title('Distribusi Data Latih & Validasi (Long-Tailed)', fontsize=16, pad=20)
    plt.xlabel('Kelas Kematangan', fontsize=12)
    plt.ylabel('Jumlah Sampel', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Tipe Set')
    plt.tight_layout()
    plt.show()

    # --- Menyimpan Data yang Telah Diproses ---
    print(f"\nMenyimpan data yang telah diproses ke {OUTPUT_ZIP}...")
    with zipfile.ZipFile(OUTPUT_ZIP, 'w') as zf:
        with zf.open('train_subset.pkl', 'w') as f: pickle.dump(train_subset, f)
        with zf.open('val_subset.pkl', 'w') as f: pickle.dump(val_subset, f)
        with zf.open('class_to_idx.pkl', 'w') as f: pickle.dump(class_to_idx, f)
        with zf.open('full_dataset_references.pkl', 'w') as f:
            pickle.dump({'samples': full_dataset.samples, 'targets': full_dataset.targets}, f)

    print("\nPreprocessing selesai.")

if __name__ == '__main__':
    main()
