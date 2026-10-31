"""Fungsi bersama untuk training (train.py) dan dashboard (app.py)."""
import matplotlib
import numpy as np
import tensorflow as tf


def preprocess_array(arr_uint8, img_size):
    """Resize + normalisasi 0-1. Dipakai identik saat training dan saat prediksi di dashboard."""
    x = tf.image.resize(tf.convert_to_tensor(arr_uint8), (img_size, img_size))
    return tf.cast(x, tf.float32) / 255.0


def load_and_preprocess(path, img_size):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return preprocess_array(img, img_size)


def last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Tidak ada layer Conv2D pada model")


def make_gradcam(model, img_batch, class_idx=None):
    """Grad-CAM: area gambar yang paling memengaruhi keputusan model.
    Return: (heatmap 0-1, indeks kelas, vektor probabilitas)."""
    grad_model = tf.keras.Model(
        model.inputs, [model.get_layer(last_conv_layer_name(model)).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), class_idx, preds[0].numpy()


def overlay_heatmap(img01, cam, alpha=0.45):
    """Tempel heatmap Grad-CAM di atas gambar (img01: float 0-1, HxWx3)."""
    h, w = img01.shape[:2]
    cam_r = tf.image.resize(cam[..., None].astype("float32"), (h, w)).numpy()[..., 0]
    heat = matplotlib.colormaps["jet"](cam_r)[..., :3]
    return np.clip((1 - alpha) * img01 + alpha * heat, 0, 1)
