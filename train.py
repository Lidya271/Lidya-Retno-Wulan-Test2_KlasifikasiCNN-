"""
LANGKAH 2-6 - EDA, pra-pemrosesan, CNN, evaluasi, dan visualisasi.

Pakai:
    python train.py                      # default: gambar 64x64, 40 epoch
    python train.py --img-size 48 --epochs 30
Semua hasil disimpan ke folder outputs/ dan dibaca oleh app.py (Streamlit).
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, optimizers

from common import load_and_preprocess, make_gradcam, overlay_heatmap
from config import LABELED_DIR, MODEL_PATH, OUT_DIR, find_images

AUTOTUNE = tf.data.AUTOTUNE


# DATA
def collect_dataframe(data_dir, min_per_class=10):
    rows = []
    for cdir in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        for f in find_images(cdir):
            rows.append((str(f), cdir.name))
    df = pd.DataFrame(rows, columns=["path", "label"])
    if df.empty:
        raise SystemExit(f"Tidak ada gambar di {data_dir}. Jalankan label_dataset.py dulu.")
    # buang file rusak
    ok = []
    for p in df["path"]:
        try:
            with Image.open(p) as im:
                im.verify()
            ok.append(True)
        except Exception:
            ok.append(False)
    n_bad = int((~np.array(ok)).sum())
    df = df[ok].reset_index(drop=True)
    counts = df["label"].value_counts()
    small = counts[counts < min_per_class].index
    if len(small):
        print("Kelas dibuang (gambar terlalu sedikit):", list(small))
        df = df[~df["label"].isin(small)].reset_index(drop=True)
    return df, n_bad


# EDA (Langkah 2)
def run_eda(df, class_names, n_bad):
    counts = df["label"].value_counts().reindex(class_names)
    counts.rename_axis("class").rename("count").reset_index().to_csv(OUT_DIR / "class_counts.csv", index=False)

    plt.figure(figsize=(9, 4.5))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis", legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    plt.xticks(rotation=30, ha="right")
    plt.title("Distribusi jumlah gambar per kelas")
    plt.ylabel("Jumlah gambar")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_class_distribution.png", dpi=120)
    plt.close()

    # contoh visual tiap kelas
    ncol = 5
    fig, axes = plt.subplots(len(class_names), ncol, figsize=(ncol * 2.2, len(class_names) * 2.2))
    for r, c in enumerate(class_names):
        files = df[df.label == c]["path"].sample(min(ncol, (df.label == c).sum()), random_state=0).tolist()
        for k in range(ncol):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(files):
                ax.imshow(Image.open(files[k]).convert("RGB").resize((128, 128)))
            if k == 0:
                ax.set_title(c, fontsize=10, loc="left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_samples.png", dpi=100)
    plt.close()

    # statistik ukuran, mode warna, kecerahan
    sample = df.sample(min(600, len(df)), random_state=0)["path"].tolist()
    widths, heights, modes, bright, contrast = [], [], [], [], []
    for p in sample:
        with Image.open(p) as im:
            widths.append(im.size[0])
            heights.append(im.size[1])
            modes.append(im.mode)
            g = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
            bright.append(float(g.mean()))
            contrast.append(float(g.std()))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].hist(widths, bins=30, color="#2a9d8f")
    axes[0].set_title("Lebar gambar (px)")
    axes[1].hist(heights, bins=30, color="#e9c46a")
    axes[1].set_title("Tinggi gambar (px)")
    axes[2].hist(bright, bins=30, color="#e76f51")
    axes[2].set_title("Kecerahan rata-rata (0-1)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_image_stats.png", dpi=110)
    plt.close()

    stats = {
        "n_sampled": len(sample),
        "width": {"min": int(min(widths)), "mean": float(np.mean(widths)), "max": int(max(widths))},
        "height": {"min": int(min(heights)), "mean": float(np.mean(heights)), "max": int(max(heights))},
        "color_modes": pd.Series(modes).value_counts().to_dict(),
        "brightness_mean": float(np.mean(bright)),
        "contrast_mean": float(np.mean(contrast)),
        "corrupted_removed": n_bad,
        "imbalance_ratio": float(counts.max() / counts.min()),
        "largest_class": str(counts.idxmax()),
        "smallest_class": str(counts.idxmin()),
    }
    with open(OUT_DIR / "image_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("EDA selesai. Rasio ketidakseimbangan:", round(stats["imbalance_ratio"], 2))
    return stats


# ------------------------------------------------------------------ PIPELINE (Langkah 3)
def make_ds(paths, labels, n_classes, img_size, batch, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, y: (load_and_preprocess(p, img_size), tf.one_hot(y, n_classes)),
        num_parallel_calls=AUTOTUNE,
    )
    return ds.batch(batch).prefetch(AUTOTUNE)


# ------------------------------------------------------------------ MODEL (Langkah 4)
def build_cnn(img_size, n_classes, lr):
    inp = layers.Input((img_size, img_size, 3))
    # Data augmentation (hanya aktif saat training)
    x = layers.RandomFlip("horizontal")(inp)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)

    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inp, out, name="marine_cnn")
    model.compile(optimizer=optimizers.Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ------------------------------------------------------------------ VISUALISASI (Langkah 6)
def plot_history(hist):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["accuracy"], label="train")
    axes[0].plot(hist["val_accuracy"], label="validasi")
    axes[0].set_title("Akurasi")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(hist["loss"], label="train")
    axes[1].plot(hist["val_loss"], label="validasi")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_curves.png", dpi=120)
    plt.close()


def plot_confusion(cm, class_names):
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion matrix (jumlah)")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion matrix (dinormalisasi per kelas)")
    for ax in axes:
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Label asli")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=110)
    plt.close()


def plot_predictions(val_paths, y_true, y_pred, class_names, img_size):
    idx = np.random.RandomState(1).choice(len(val_paths), min(16, len(val_paths)), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(Image.open(val_paths[i]).convert("RGB").resize((img_size * 3, img_size * 3)))
        ok = y_true[i] == y_pred[i]
        ax.set_title(f"Asli: {class_names[y_true[i]]}\nPrediksi: {class_names[y_pred[i]]}",
                     fontsize=9, color="green" if ok else "red")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "prediction_samples.png", dpi=110)
    plt.close()


def plot_gradcam(model, val_paths, y_true, class_names, img_size, n=8):
    idx = np.random.RandomState(2).choice(len(val_paths), min(n, len(val_paths)), replace=False)
    fig, axes = plt.subplots(2, len(idx), figsize=(2.3 * len(idx), 5))
    axes = np.atleast_2d(axes)
    for col, i in enumerate(idx):
        x = load_and_preprocess(val_paths[i], img_size).numpy()
        cam, pred_idx, _ = make_gradcam(model, x[None])
        axes[0, col].imshow(x)
        axes[0, col].set_title(f"{class_names[y_true[i]]} -> {class_names[pred_idx]}", fontsize=8)
        axes[1, col].imshow(overlay_heatmap(x, cam))
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    axes[0, 0].text(-0.1, 0.5, "Asli", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right")
    axes[1, 0].text(-0.1, 0.5, "Grad-CAM", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "gradcam_samples.png", dpi=110)
    plt.close()


# ------------------------------------------------------------------ MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(LABELED_DIR))
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    tf.random.set_seed(a.seed)
    OUT_DIR.mkdir(exist_ok=True)

    df, n_bad = collect_dataframe(a.data_dir)
    class_names = sorted(df["label"].unique())
    n_classes = len(class_names)
    df["label_id"] = df["label"].map({c: i for i, c in enumerate(class_names)})
    print(f"{len(df)} gambar, {n_classes} kelas: {class_names}")

    # Langkah 2: EDA
    stats = run_eda(df, class_names, n_bad)

    # Langkah 3: split 80/20 (stratified), one-hot ada di pipeline
    tr, va = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=a.seed)
    train_ds = make_ds(tr["path"].values, tr["label_id"].values, n_classes, a.img_size, a.batch, True)
    val_ds = make_ds(va["path"].values, va["label_id"].values, n_classes, a.img_size, a.batch, False)

    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=tr["label_id"].values)
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    print("Class weight (mengatasi imbalance):", {class_names[i]: round(w, 2) for i, w in class_weight.items()})
    # Menyimpan dataset gambar yang sudah dinormalisasi
    normalized_data = []

    for path, label in zip(df["path"], df["label_id"]):
        img = load_and_preprocess(path, a.img_size).numpy()

        normalized_data.append({
        "path": path,
        "label": class_names[label],
        "normalized_pixel": img.flatten().tolist()
    })

    df_normalized = pd.DataFrame(normalized_data)

    df_normalized.to_csv(
        OUT_DIR / "Nama_Test2_KlasifikasiCNN.csv",
        index=False
    )

    print("Dataset normalisasi berhasil disimpan!")

    # Langkah 4: model + training
    model = build_cnn(a.img_size, n_classes, a.lr)
    model.summary()
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]
    h = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=cbs, class_weight=class_weight)
    hist = {k: [float(v) for v in vals] for k, vals in h.history.items()}
    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(hist, f)
    model.save(MODEL_PATH)
    plot_history(hist)

    # Langkah 5: evaluasi
    probs = model.predict(val_ds, verbose=0)
    y_pred = probs.argmax(1)
    y_true = va["label_id"].values
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    with open(OUT_DIR / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    table = pd.DataFrame(report).T
    table.to_csv(OUT_DIR / "metrics_table.csv")
    print("\n", classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # Langkah 6: visualisasi
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    np.save(OUT_DIR / "confusion_matrix.npy", cm)
    plot_confusion(cm, class_names)
    val_paths = va["path"].tolist()
    plot_predictions(val_paths, y_true, y_pred, class_names, a.img_size)
    plot_gradcam(model, val_paths, y_true, class_names, a.img_size)

    config = {
        "img_size": a.img_size,
        "class_names": class_names,
        "n_images": int(len(df)),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "epochs_requested": a.epochs,
        "epochs_run": len(hist["loss"]),
        "batch_size": a.batch,
        "learning_rate": a.lr,
        "optimizer": "Adam",
        "params": int(model.count_params()),
        "label_source": "CLIP zero-shot pseudo-label",
        "image_stats": stats,
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSelesai. Akurasi validasi = {report['accuracy']:.4f}. Jalankan: streamlit run app.py")


if __name__ == "__main__":
    main()
