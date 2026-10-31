"""
LANGKAH 1 - Pengambilan dataset & pelabelan otomatis.

Dataset 'oceanic-life-dataset' tidak punya label/kelas. Dataset SUDAH ada di komputer
(tidak diunduh lagi). Skrip ini:
  1. membaca semua gambar dari folder dataset (rekursif, termasuk subfolder),
  2. memberi label ke 8 kelas memakai CLIP zero-shot (tanpa training),
  3. menyalin gambar (dikonversi ke RGB JPEG) ke dataset_labeled/<kelas>/,
  4. membuat contact sheet per kelas agar label bisa dicek manual,
  5. mencetak jumlah gambar per kelas.

Pakai:
    python label_dataset.py --raw-dir path/ke/folder/dataset
    python label_dataset.py                       # default: folder ./dataset
    python label_dataset.py --min-conf 0.5        # buang label yang kurang yakin
    python label_dataset.py --from-csv outputs/labels.csv   # bangun ulang folder setelah labels.csv diedit manual
"""
import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import (CLASS_NAMES, CLASS_PROMPTS, CLIP_MODEL_ID, LABELED_DIR,
                    OUT_DIR, find_images)

DEFAULT_RAW_DIR = r"C:\Semester 7\Analilisis Big Data\Test_KlasifikasiCNN\Dataset"  # folder dataset yang sudah ada (ubah sesuai lokasimu)


@torch.no_grad()
def clip_label(paths, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    prompts, owner = [], []
    for ci, name in enumerate(CLASS_NAMES):
        for p in CLASS_PROMPTS[name]:
            prompts.append(p)
            owner.append(ci)
    owner = torch.tensor(owner, device=device)
    text_in = proc.tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    rows = []
    for start in range(0, len(paths), batch_size):
        batch_paths, imgs = [], []
        for p in paths[start:start + batch_size]:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                batch_paths.append(p)
            except Exception as e:  # gambar rusak -> lewati
                print("Lewati (rusak):", p, e)
        if not imgs:
            continue
        pix = proc.image_processor(imgs, return_tensors="pt")["pixel_values"].to(device)
        out = model(input_ids=text_in["input_ids"],
                    attention_mask=text_in["attention_mask"], pixel_values=pix)
        logits = out.logits_per_image  # (B, jumlah_prompt)
        cls_logits = torch.stack(
            [torch.logsumexp(logits[:, owner == ci], dim=1) for ci in range(len(CLASS_NAMES))], dim=1
        )
        probs = cls_logits.softmax(dim=1).cpu().numpy()
        for p, pr in zip(batch_paths, probs):
            order = np.argsort(-pr)
            rows.append({
                "path": str(p),
                "label": CLASS_NAMES[order[0]],
                "confidence": float(pr[order[0]]),
                "second_label": CLASS_NAMES[order[1]],
            })
        print(f"  {min(start + batch_size, len(paths))}/{len(paths)} gambar", end="\r")
    print()
    return pd.DataFrame(rows)


def build_folders(df, min_conf):
    if LABELED_DIR.exists():
        shutil.rmtree(LABELED_DIR)
    kept = df[df["confidence"] >= min_conf].reset_index(drop=True)
    for c in CLASS_NAMES:
        (LABELED_DIR / c).mkdir(parents=True, exist_ok=True)
    for i, r in kept.iterrows():
        try:
            Image.open(r["path"]).convert("RGB").save(
                LABELED_DIR / r["label"] / f"{i:05d}.jpg", quality=95)
        except Exception as e:
            print("Gagal menyalin:", r["path"], e)
    print(f"\n{len(kept)} dari {len(df)} gambar disalin ke {LABELED_DIR}/ (min_conf={min_conf})")
    return kept


def contact_sheets():
    out = OUT_DIR / "label_check"
    out.mkdir(parents=True, exist_ok=True)
    for c in CLASS_NAMES:
        files = find_images(LABELED_DIR / c)
        if not files:
            continue
        pick = list(np.random.RandomState(0).permutation(files)[:20])
        fig, axes = plt.subplots(4, 5, figsize=(12, 10))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, f in zip(axes.ravel(), pick):
            ax.imshow(Image.open(f))
        fig.suptitle(f"Contoh kelas: {c} ({len(files)} gambar)", fontsize=14)
        fig.tight_layout()
        fig.savefig(out / f"{c}.jpg", dpi=80)
        plt.close(fig)
    print("Contact sheet untuk pengecekan manual ada di", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR,
                    help=f"Folder dataset yang sudah ada (default: ./{DEFAULT_RAW_DIR})")
    ap.add_argument("--min-conf", type=float, default=0.4,
                    help="Ambang keyakinan minimum CLIP (0-1). Di bawah ini dibuang.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--from-csv", default=None, help="Bangun folder dari labels.csv yang sudah diedit")
    a = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    if a.from_csv:
        df = pd.read_csv(a.from_csv)
    else:
        raw = Path(a.raw_dir)
        if not raw.is_dir():
            raise SystemExit(f"Folder dataset '{raw}' tidak ditemukan. "
                             f"Gunakan: python label_dataset.py --raw-dir path/ke/dataset")
        paths = find_images(raw)
        print(f"Ditemukan {len(paths)} gambar di {raw}")
        if not paths:
            raise SystemExit("Tidak ada gambar ditemukan. Periksa --raw-dir.")
        df = clip_label(paths, a.batch_size)
        df.to_csv(OUT_DIR / "labels.csv", index=False)
        print("Label lengkap disimpan di", OUT_DIR / "labels.csv")

    kept = build_folders(df, a.min_conf)
    print("\nJumlah gambar per kelas:")
    print(kept["label"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int).to_string())
    contact_sheets()
    print("\nTIPS: buka outputs/label_check/*.jpg, lalu perbaiki outputs/labels.csv bila ada salah label "
          "dan jalankan ulang dengan --from-csv.")


if __name__ == "__main__":
    main()
