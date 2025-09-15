import time
from typing import Dict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

from super_encoder import SuperEncoder
from dataset import make_loaders_from_zip
from tqdm.auto import tqdm

from config import (
    PATHS, TRAIN,
    get_device, set_seed, denormalize_to_unit
)

# ---------------------------
# Plot helper
# ---------------------------
def plot_history(history: Dict[str, list], title: str = "Riwayat Training Super-Encoder"):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    epochs_run = range(1, len(history['train_loss']) + 1)

    # Total Loss
    axes[0].plot(epochs_run, history['train_loss'], label="Train Total Loss")
    axes[0].plot(epochs_run, history['val_loss'], label="Val Total Loss")
    axes[0].set_title("Total Combined Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()

    # Reconstruction Loss
    axes[1].plot(epochs_run, history['train_recon_loss'], label="Train Recon Loss")
    axes[1].plot(epochs_run, history['val_recon_loss'], label="Val Recon Loss")
    axes[1].set_title("Reconstruction Loss (MSE)"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    # Separability (CE) Loss
    axes[2].plot(epochs_run, history['train_sep_loss'], label="Train Separability Loss")
    axes[2].plot(epochs_run, history['val_sep_loss'], label="Val Separability Loss")
    axes[2].set_title("Separability Loss (Cross-Entropy)"); axes[2].set_xlabel("Epoch"); axes[2].legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # ---- Config ----
    DEVICE = get_device()
    PREPROCESSED_DATA_ZIP = PATHS.preprocessed_zip
    OUTPUT_MODEL_PATH = PATHS.output_model_path

    IMAGE_SIZE = TRAIN.image_size
    EPOCHS = TRAIN.epochs
    BATCH_SIZE = TRAIN.batch_size
    NUM_WORKERS = TRAIN.num_workers
    LEARNING_RATE = TRAIN.learning_rate
    WEIGHT_DECAY = TRAIN.weight_decay
    ALPHA_SEP = TRAIN.alpha_sep
    BETA_RECON = TRAIN.beta_recon
    EARLY_STOPPING_PATIENCE = TRAIN.early_stopping_patience
    SEED = TRAIN.seed

    print(f"Device: {DEVICE}")
    set_seed(SEED)

    # ---- Data ----
    print(f"Memuat data dari {PREPROCESSED_DATA_ZIP} ...")
    train_loader, val_loader, class_to_idx = make_loaders_from_zip(
        zip_path=PREPROCESSED_DATA_ZIP,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        pin_memory=True,
        shuffle_train=True
    )

    num_classes = len(class_to_idx)

    # ---- Model & Optim ----
    model = SuperEncoder(
        latent_dim=TRAIN.latent_dim,
        num_classes=num_classes,
        dropout_prob=TRAIN.dropout_prob,
        l2_normalize_latent=TRAIN.l2_normalize_latent,
        use_adaptive_pool=TRAIN.use_adaptive_pool
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=7, factor=0.5, verbose=True)
    loss_fn_recon = nn.MSELoss()
    loss_fn_sep = nn.CrossEntropyLoss()

    # ---- Training Loop ----
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'train_recon_loss': [], 'train_sep_loss': [],
        'val_loss': [], 'val_recon_loss': [], 'val_sep_loss': []
    }

    print("\nMulai training ...")
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        val_loader = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]  ", leave=False)
        # ---- Train ----
        model.train()
        train_recon_accum = 0.0
        train_sep_accum = 0.0
        nsamples_train = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            x_hat, logits = model(images)

            # Rekonstruksi: target diubah ke [0,1] agar match decoder Sigmoid
            images_unit = denormalize_to_unit(images)
            loss_recon = loss_fn_recon(x_hat, images_unit)
            loss_sep = loss_fn_sep(logits, labels)

            total_loss = BETA_RECON * loss_recon + ALPHA_SEP * loss_sep

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            bsz = images.size(0)
            nsamples_train += bsz
            train_recon_accum += loss_recon.item() * bsz
            train_sep_accum += loss_sep.item() * bsz

        avg_train_recon = train_recon_accum / nsamples_train
        avg_train_sep = train_sep_accum / nsamples_train
        avg_train_loss = BETA_RECON * avg_train_recon + ALPHA_SEP * avg_train_sep

        # ---- Val ----
        model.eval()
        val_recon_accum = 0.0
        val_sep_accum = 0.0
        nsamples_val = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                x_hat, logits = model(images)

                images_unit = denormalize_to_unit(images)
                v_recon = loss_fn_recon(x_hat, images_unit)
                v_sep = loss_fn_sep(logits, labels)

                bsz = images.size(0)
                nsamples_val += bsz
                val_recon_accum += v_recon.item() * bsz
                val_sep_accum += v_sep.item() * bsz

        avg_val_recon = val_recon_accum / nsamples_val
        avg_val_sep = val_sep_accum / nsamples_val
        avg_val_loss = BETA_RECON * avg_val_recon + ALPHA_SEP * avg_val_sep

        history['train_recon_loss'].append(avg_train_recon)
        history['train_sep_loss'].append(avg_train_sep)
        history['train_loss'].append(avg_train_loss)

        history['val_recon_loss'].append(avg_val_recon)
        history['val_sep_loss'].append(avg_val_sep)
        history['val_loss'].append(avg_val_loss)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train: {avg_train_loss:.4f} (R {avg_train_recon:.4f} / S {avg_train_sep:.4f}) | "
            f"Val: {avg_val_loss:.4f} (R {avg_val_recon:.4f} / S {avg_val_sep:.4f})"
        )

        scheduler.step(avg_val_loss)

        # ---- Checkpoint & Early Stopping ----
        if avg_val_loss < best_val_loss:
            print(f"  ↑ Val improved {best_val_loss:.6f} → {avg_val_loss:.6f} | Save: {OUTPUT_MODEL_PATH}")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping (no improve for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    total_time = (time.time() - start_time) / 60.0
    print(f"\nTraining selesai dalam {total_time:.2f} menit. Best model: {OUTPUT_MODEL_PATH}")

    # ---- Plot ----
    plot_history(history)
