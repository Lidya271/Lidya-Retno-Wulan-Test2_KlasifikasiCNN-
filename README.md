# Klasifikasi Biota Laut (CNN + Streamlit)

Dataset: https://www.kaggle.com/datasets/cyanex1702/oceanic-life-dataset
(tidak berlabel -> dilabeli otomatis ke 8 kelas dengan CLIP zero-shot)

## Urutan menjalankan
```bash
pip install -r requirements.txt
python label_dataset.py --raw-dir path/ke/dataset   # Langkah 1: beri label 8 kelas + hitung gambar/kelas (dataset sudah ada, tidak diunduh)
# cek outputs/label_check/*.jpg, perbaiki outputs/labels.csv bila perlu, lalu:
# python label_dataset.py --from-csv outputs/labels.csv
python train.py                  # Langkah 2-6: EDA, preprocessing, CNN, evaluasi, visualisasi
streamlit run app.py             # Langkah 7: dashboard
```

## Pemetaan ke laporan (Langkah 8)
| Bagian laporan | Sumber |
|---|---|
| Deskripsi dataset, jumlah gambar/kelas | output `label_dataset.py`, `outputs/class_counts.csv` |
| EDA | `outputs/eda_*.png`, `outputs/image_stats.json` |
| Pra-pemrosesan | fungsi `make_ds` & `preprocess_array` (resize, normalisasi 0-1, split 80/20, augmentasi, one-hot) |
| Arsitektur CNN & parameter | `build_cnn` di `train.py`, `outputs/config.json` |
| Evaluasi | `outputs/metrics_table.csv`, `classification_report.json` |
| Visualisasi | `confusion_matrix.png`, `training_curves.png`, `gradcam_samples.png`, `prediction_samples.png` |
| Dashboard | screenshot `streamlit run app.py` |

## Catatan penting
- Label hasil CLIP adalah *pseudo-label*. Sebutkan ini di laporan dan cek `outputs/label_check/`.
- Ukuran gambar default 64x64 (lebih baik untuk biota laut); ubah ke 48 dengan `--img-size 48`.
- Di Google Colab, jalankan dengan `!python ...` lalu untuk Streamlit gunakan `pyngrok`/`localtunnel`.
