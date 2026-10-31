"""Konfigurasi bersama (tanpa import berat) untuk seluruh skrip praktikum."""
from pathlib import Path

# ---- 8 kelas yang didefinisikan manual (dataset asli TIDAK berlabel) ----
CLASS_NAMES = [
    "fish",
    "coral",
    "shark",
    "octopus",
    "crustacean",
    "seaweed",
    "sponge",
    "other_marine_organisms",
]

# Prompt teks untuk pelabelan zero-shot CLIP. Skor kelas = logsumexp dari semua prompt-nya.
CLASS_PROMPTS = {
    "fish": [
        "a photo of a fish underwater",
        "a photo of a school of fish",
        "a colorful tropical fish swimming in the sea",
    ],
    "coral": [
        "a photo of coral reef",
        "a close-up photo of hard coral underwater",
        "a photo of soft coral on a reef",
    ],
    "shark": [
        "a photo of a shark",
        "a shark swimming underwater",
        "a great white shark in the ocean",
    ],
    "octopus": [
        "a photo of an octopus",
        "a photo of a squid",
        "a photo of a cuttlefish",
    ],
    "crustacean": [
        "a photo of a crab",
        "a photo of a lobster",
        "a photo of a shrimp",
    ],
    "seaweed": [
        "a photo of seaweed",
        "a photo of a kelp forest underwater",
        "a photo of marine algae",
    ],
    "sponge": [
        "a photo of a sea sponge",
        "a photo of a tube sponge on a reef",
    ],
    "other_marine_organisms": [
        "a photo of a jellyfish",
        "a photo of a sea turtle",
        "a photo of a starfish",
        "a photo of a sea urchin",
        "a photo of a dolphin or whale",
        "a photo of a sea anemone",
        "a photo of a seahorse",
        "a photo of a ray or manta",
    ],
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RAW_DATASET = "cyanex1702/oceanic-life-dataset"
LABELED_DIR = Path("dataset_labeled")
OUT_DIR = Path("outputs")
MODEL_PATH = OUT_DIR / "marine_cnn.keras"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def find_images(root):
    """Cari semua file gambar secara rekursif."""
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)
