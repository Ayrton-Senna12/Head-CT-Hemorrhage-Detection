"""Proje kökü ve veri yolları."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT
HEAD_CT_DIR = DATA_DIR / "head_ct"
LABELS_CSV = DATA_DIR / "labels.csv"

IMAGE_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ("Normal", "Hemorrhage")

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "plots"

for d in (OUTPUT_DIR, CHECKPOINTS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
