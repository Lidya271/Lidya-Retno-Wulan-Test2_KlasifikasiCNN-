"""
LANGKAH 7 - Dashboard Streamlit.   Jalankan:  streamlit run app.py
(Pastikan label_dataset.py dan train.py sudah dijalankan.)
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
from PIL import Image

from common import make_gradcam, overlay_heatmap, preprocess_array
from config import MODEL_PATH, OUT_DIR

st.set_page_config(page_title="Klasifikasi Biota Laut (CNN)", page_icon="🐠", layout="wide")


# ------------------------------------------------------------------ loader
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_json(name):
    with open(OUT_DIR / name) as f:
        return json.load(f)


if not (OUT_DIR / "config.json").exists() or not MODEL_PATH.exists():
    st.error("Hasil training belum ada. Jalankan `python label_dataset.py` lalu `python train.py` terlebih dahulu.")
    st.stop()

cfg = load_json("config.json")
hist = load_json("history.json")
report = load_json("classification_report.json")
counts = pd.read_csv(OUT_DIR / "class_counts.csv")
cm = np.load(OUT_DIR / "confusion_matrix.npy")
class_names = cfg["class_names"]
stats = cfg["image_stats"]

# ------------------------------------------------------------------ header
st.title("🐠 Klasifikasi Biota Laut dengan CNN")
st.markdown(
    "Dashboard hasil praktikum klasifikasi citra **Oceanic Life Dataset** (Kaggle) menggunakan "
    "*Convolutional Neural Network*. Dataset asli tidak memiliki label, sehingga 8 kelas "
    "(`" + "`, `".join(class_names) + "`) dibuat dengan pelabelan otomatis **CLIP zero-shot** lalu dipakai melatih CNN."
)
st.info(
    "Catatan: label bersifat *pseudo-label*, jadi akurasi di bawah mengukur seberapa konsisten CNN "
    "mengikuti label tersebut, bukan akurasi terhadap label manusia."
)

tab_data, tab_eda, tab_train, tab_eval, tab_pred = st.tabs(
    ["📦 Dataset", "🔍 EDA", "📈 Training", "✅ Evaluasi", "🖼️ Prediksi Gambar Baru"]
)

# ------------------------------------------------------------------ dataset
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah kelas", len(class_names))
    c2.metric("Total gambar", cfg["n_images"])
    c3.metric("Data latih", f"{cfg['n_train']} ({cfg['n_train'] / cfg['n_images']:.0%})")
    c4.metric("Data uji/validasi", f"{cfg['n_val']} ({cfg['n_val'] / cfg['n_images']:.0%})")

    st.subheader("Distribusi gambar per kelas")
    st.bar_chart(counts.set_index("class")["count"])
    big, small = stats["largest_class"], stats["smallest_class"]
    ratio = stats["imbalance_ratio"]
    st.markdown(
        f"**Analisis:** kelas terbanyak adalah **{big}** dan paling sedikit **{small}**, dengan rasio "
        f"{ratio:.1f}:1. "
        + ("Data **tidak seimbang** (rasio > 3), sehingga model cenderung bias ke kelas besar; "
           "karena itu training memakai *class weight* dan metrik F1 per kelas perlu diperhatikan."
           if ratio > 3 else "Data relatif seimbang sehingga akurasi cukup representatif.")
    )
    st.dataframe(counts, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ EDA
with tab_eda:
    st.subheader("Contoh visual tiap kelas")
    st.image(str(OUT_DIR / "eda_samples.png"), use_container_width=True)
    st.markdown(
        "**Analisis:** tiap baris adalah satu kelas. Perhatikan variasi sudut pandang, warna air, "
        "dan latar belakang; kelas seperti *coral*, *sponge*, dan *seaweed* sering bertekstur mirip "
        "sehingga berpotensi saling tertukar."
    )
    st.subheader("Ukuran, warna, dan kualitas gambar")
    st.image(str(OUT_DIR / "eda_image_stats.png"), use_container_width=True)
    w, h_ = stats["width"], stats["height"]
    st.markdown(
        f"**Analisis:** dari {stats['n_sampled']} gambar sampel, lebar berkisar {w['min']}–{w['max']} px "
        f"(rata-rata {w['mean']:.0f}) dan tinggi {h_['min']}–{h_['max']} px (rata-rata {h_['mean']:.0f}), "
        f"sehingga semua gambar di-*resize* ke {cfg['img_size']}×{cfg['img_size']} piksel. "
        f"Mode warna: {stats['color_modes']}. Kecerahan rata-rata {stats['brightness_mean']:.2f} "
        f"dan kontras rata-rata {stats['contrast_mean']:.2f}. "
        f"File rusak yang dibuang: {stats['corrupted_removed']}."
    )

# ------------------------------------------------------------------ training
with tab_train:
    st.subheader("Arsitektur & parameter")
    st.markdown(
        f"- Augmentasi: flip horizontal, rotasi, zoom, shift\n"
        f"- 4 blok (2×Conv+BatchNorm → MaxPool → Dropout) dengan filter 32/64/128/256, "
        f"GlobalAvgPool, Dense 256, Dropout 0.5, Softmax {len(class_names)} kelas\n"
        f"- Optimizer {cfg['optimizer']}, learning rate {cfg['learning_rate']}, batch {cfg['batch_size']}, "
        f"epoch dijalankan {cfg['epochs_run']} (maks. {cfg['epochs_requested']}), parameter {cfg['params']:,}"
    )
    st.subheader("Grafik akurasi dan loss")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, key, title in zip(axes, ["accuracy", "loss"], ["Akurasi", "Loss"]):
        ax.plot(hist[key], label="train")
        ax.plot(hist["val_" + key], label="validasi")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    st.pyplot(fig)
    best = int(np.argmin(hist["val_loss"]))
    gap = hist["accuracy"][best] - hist["val_accuracy"][best]
    st.markdown(
        f"**Analisis:** val loss terendah pada epoch {best + 1} (akurasi validasi "
        f"{hist['val_accuracy'][best]:.2%}, akurasi latih {hist['accuracy'][best]:.2%}). "
        + (f"Selisih {gap:.1%} cukup besar, menandakan **overfitting**; tambah augmentasi/regularisasi atau data."
           if gap > 0.10 else f"Selisih {gap:.1%} kecil, model **konvergen dengan baik tanpa overfitting berarti**.")
    )

# ------------------------------------------------------------------ evaluasi
with tab_eval:
    st.subheader("Metrik evaluasi (data uji)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{report['accuracy']:.2%}")
    m2.metric("Precision (macro)", f"{report['macro avg']['precision']:.2%}")
    m3.metric("Recall (macro)", f"{report['macro avg']['recall']:.2%}")
    m4.metric("F1-score (macro)", f"{report['macro avg']['f1-score']:.2%}")

    per = pd.DataFrame({c: report[c] for c in class_names}).T.rename(
        columns={"f1-score": "f1"})[["precision", "recall", "f1", "support"]]
    st.dataframe(per.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}", "support": "{:.0f}"}),
                 use_container_width=True)
    best_c, worst_c = per["f1"].idxmax(), per["f1"].idxmin()
    st.markdown(
        f"**Analisis:** akurasi {report['accuracy']:.1%} dan F1 macro {report['macro avg']['f1-score']:.1%}. "
        f"Kelas paling mudah dikenali: **{best_c}** (F1 {per.loc[best_c, 'f1']:.2f}); "
        f"paling sulit: **{worst_c}** (F1 {per.loc[worst_c, 'f1']:.2f}). "
        "Bila F1 macro jauh di bawah akurasi, performa dipengaruhi kelas minoritas yang lemah."
    )

    st.subheader("Confusion matrix")
    normed = st.checkbox("Tampilkan dinormalisasi per kelas", value=True)
    mat = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normed else cm
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(mat, annot=True, fmt=".2f" if normed else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label asli")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    if off.max() > 0:
        i, j = np.unravel_index(off.argmax(), off.shape)
        st.markdown(f"**Analisis:** kesalahan terbanyak adalah **{class_names[i]} diprediksi sebagai "
                    f"{class_names[j]}** ({off[i, j]} gambar), menandakan kemiripan visual kedua kelas.")

    st.subheader("Contoh prediksi vs label asli")
    st.image(str(OUT_DIR / "prediction_samples.png"), use_container_width=True)
    st.markdown("**Analisis:** judul hijau = prediksi benar, merah = salah. Kesalahan biasanya pada gambar "
                "buram, objek kecil, atau latar yang didominasi warna/tekstur kelas lain.")

    st.subheader("Feature importance (Grad-CAM)")
    st.image(str(OUT_DIR / "gradcam_samples.png"), use_container_width=True)
    st.markdown("**Analisis:** area merah/kuning adalah bagian gambar yang paling memengaruhi keputusan model. "
                "Bila fokus ada pada objek biota, model belajar fitur yang benar; bila pada latar, model "
                "kemungkinan memakai shortcut (misalnya warna air).")

# ------------------------------------------------------------------ prediksi
with tab_pred:
    st.subheader("Uji prediksi gambar baru")
    up = st.file_uploader("Unggah gambar (jpg/png/webp)", type=["jpg", "jpeg", "png", "webp"])
    if up is not None:
        img = Image.open(up).convert("RGB")
        x = preprocess_array(np.array(img), cfg["img_size"]).numpy()
        model = load_model()
        cam, idx, probs = make_gradcam(model, x[None])
        pred = class_names[idx]
        conf = float(probs[idx])

        c1, c2 = st.columns(2)
        c1.image(img, caption="Gambar yang diunggah", use_container_width=True)
        big = np.asarray(img.resize((256, 256)), dtype=np.float32) / 255.0
        c2.image(overlay_heatmap(big, cam), caption="Grad-CAM (area yang diperhatikan model)",
                 use_container_width=True)

        st.success(f"Prediksi: **{pred}** (keyakinan {conf:.1%})")
        prob_df = pd.DataFrame({"kelas": class_names, "probabilitas": probs}).set_index("kelas")
        st.bar_chart(prob_df)
        top2 = np.argsort(-probs)[:2]
        msg = (f"**Analisis:** model memilih **{pred}** dengan keyakinan {conf:.1%}; kandidat kedua "
               f"**{class_names[top2[1]]}** ({probs[top2[1]]:.1%}). ")
        msg += ("Keyakinan rendah (<50%), hasil perlu diragukan." if conf < 0.5
                else "Keyakinan tinggi; Grad-CAM menunjukkan area gambar yang mendasari keputusan ini.")
        st.markdown(msg)
