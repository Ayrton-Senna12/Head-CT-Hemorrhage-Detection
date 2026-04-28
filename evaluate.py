"""
Test seti metrikleri ve karışıklık matrisi.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

import config
from data_pipeline import load_labels_dataframe, make_dataloaders, split_train_val_test_by_group
from models import build_model


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys: List[int] = []
    preds: List[int] = []
    probs: List[float] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pr = torch.softmax(logits, dim=1)
        p = logits.argmax(dim=1)
        ys.extend(y.detach().cpu().numpy().tolist())
        preds.extend(p.cpu().numpy().tolist())
        probs.extend(pr.max(dim=1).values.cpu().numpy().tolist())
    return np.array(ys), np.array(preds), np.array(probs)


def evaluate_on_test(
    checkpoint_path: Path,
    output_report: Optional[Path] = None,
    confusion_image: Optional[Path] = None,
    batch_size: int = 16,
    random_state: int = 42,
    group_column: Optional[str] = None,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    model_type = ckpt["model_type"]
    num_classes = int(ckpt.get("num_classes", config.NUM_CLASSES))

    model = build_model(model_type, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    full_df = load_labels_dataframe(group_column=group_column)
    train_df, val_df, test_df = split_train_val_test_by_group(full_df, random_state=random_state)
    _, _, test_loader = make_dataloaders(train_df, val_df, test_df, batch_size=batch_size)

    y_true, y_pred, _probs = collect_predictions(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    prec = precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)

    lines = [
        "=== Test Seti Özeti ===",
        f"Accuracy : {acc:.4f}",
        f"Recall   : {rec:.4f}  (pozitif sınıf: Hemorrhage=1)",
        f"Precision: {prec:.4f}",
        "",
    ]
    text = "\n".join(lines)
    print(text)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = list(ckpt.get("class_names", config.CLASS_NAMES))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix (Test)")
    fig.tight_layout()

    if confusion_image is None:
        confusion_image = config.PLOTS_DIR / "confusion_matrix_test.png"
    confusion_image = Path(confusion_image)
    confusion_image.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(confusion_image, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix kaydı: {confusion_image}")

    if output_report is None:
        output_report = config.OUTPUT_DIR / "test_metrics.txt"
    output_report = Path(output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(text + f"\nConfusion matrix (satır=gerçek, sütun=tahmin):\n{cm}\n")
    print(f"Rapor: {output_report}")

    return {"accuracy": acc, "recall": rec, "precision": prec}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINTS_DIR / "best_model.pth"))
    p.add_argument("--report", type=str, default=str(config.OUTPUT_DIR / "test_metrics.txt"))
    p.add_argument("--cm-out", type=str, default=str(config.PLOTS_DIR / "confusion_matrix_test.png"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--group-col", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_on_test(
        Path(args.checkpoint),
        output_report=Path(args.report),
        confusion_image=Path(args.cm_out),
        batch_size=args.batch_size,
        group_column=args.group_col,
    )


if __name__ == "__main__":
    main()
