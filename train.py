"""
Eğitim döngüsü, erken durdurma, Optuna hiperparametre araması ve eğitim grafikleri.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from data_pipeline import load_labels_dataframe, make_dataloaders, split_train_val_test_by_group
from models import build_model


@dataclass
class History:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    losses, accs = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits.detach(), y))
    return float(np.mean(losses)), float(np.mean(accs))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    losses, accs = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits, y))
    return float(np.mean(losses)), float(np.mean(accs))


def fit_with_early_stopping(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
    patience: int = 7,
    min_delta: float = 1e-4,
    scheduler_t_max: Optional[int] = None,
) -> Tuple[History, Dict[str, Any]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tmax = scheduler_t_max if scheduler_t_max is not None else max_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=tmax)

    history = History()
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.train_loss.append(tr_loss)
        history.val_loss.append(va_loss)
        history.train_acc.append(tr_acc)
        history.val_acc.append(va_acc)

        if va_loss < best_val - min_delta:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    info = {"best_val_loss": best_val, "stopped_epoch": epoch, "best_state": best_state}
    return history, info


def plot_history(history: History, out_path: Path, title: str = "Eğitim özeti") -> None:
    """Train/Val loss ve accuracy grafiklerini kaydeder."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history.train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history.train_loss, label="Train Loss")
    axes[0].plot(epochs, history.val_loss, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Kayıp")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.train_acc, label="Train Acc")
    axes[1].plot(epochs, history.val_acc, label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Doğruluk")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(
    path: Path,
    model_name: str,
    state_dict: Dict[str, torch.Tensor],
    history: History,
    meta: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": model_name,
        "num_classes": config.NUM_CLASSES,
        "class_names": list(config.CLASS_NAMES),
        "image_size": config.IMAGE_SIZE,
        "state_dict": state_dict,
        "history": {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "train_acc": history.train_acc,
            "val_acc": history.val_acc,
        },
        "meta": meta,
    }
    torch.save(payload, path)


def run_optuna(
    model_name: str,
    n_trials: int = 12,
    patience: int = 7,
    random_state: int = 42,
    group_column: Optional[str] = None,
    study_name: str = "head_ct_study",
) -> optuna.Study:
    """
    Learning rate, batch size ve üst sınır epoch için Optuna optimizasyonu.
    Her deneme erken durdurma ile tamamlanır; amaç validasyon kaybını minimize etmek.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_df = load_labels_dataframe(group_column=group_column)
    train_df, val_df, test_df = split_train_val_test_by_group(full_df, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        max_epochs = trial.suggest_int("max_epochs", 15, 60)

        train_loader, val_loader, _ = make_dataloaders(
            train_df, val_df, test_df, batch_size=batch_size
        )
        model = build_model(model_name, pretrained=True).to(device)
        history, info = fit_with_early_stopping(
            model,
            train_loader,
            val_loader,
            device,
            max_epochs=max_epochs,
            lr=lr,
            patience=patience,
        )
        best_val = float(info["best_val_loss"])
        trial.set_user_attr("stopped_epoch", int(info["stopped_epoch"]))
        trial.set_user_attr("train_len", len(history.train_loss))
        return best_val

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", study_name=study_name, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    out = config.OUTPUT_DIR / f"{study_name}_best_params.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Optuna en iyi parametreler: {study.best_params}")
    print(f"Kayıt: {out}")
    return study


def train_final(
    model_name: str,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int = 7,
    random_state: int = 42,
    group_column: Optional[str] = None,
    out_name: str = "best_model.pth",
) -> Path:
    """Tam veri bölmesiyle final eğitim ve checkpoint kaydı."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_df = load_labels_dataframe(group_column=group_column)
    train_df, val_df, test_df = split_train_val_test_by_group(full_df, random_state=random_state)
    train_loader, val_loader, _ = make_dataloaders(train_df, val_df, test_df, batch_size=batch_size)

    model = build_model(model_name, pretrained=True).to(device)
    history, info = fit_with_early_stopping(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=max_epochs,
        lr=lr,
        patience=patience,
    )
    if info["best_state"] is not None:
        model.load_state_dict(info["best_state"])

    ckpt_path = config.CHECKPOINTS_DIR / out_name
    meta = {
        "lr": lr,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "best_val_loss": info["best_val_loss"],
        "stopped_epoch": info["stopped_epoch"],
    }
    save_checkpoint(ckpt_path, model_name, model.state_dict(), history, meta)

    plot_path = config.PLOTS_DIR / f"{Path(out_name).stem}_curves.png"
    plot_history(history, plot_path, title=f"{model_name} — eğitim / doğrulama")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Grafik: {plot_path}")
    return ckpt_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Head CT eğitimi")
    p.add_argument("--model", type=str, default="convnext_tiny", help="convnext_tiny veya custom")
    p.add_argument("--optuna", action="store_true", help="Optuna hiperparametre araması çalıştır")
    p.add_argument("--trials", type=int, default=10, help="Optuna deneme sayısı")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--group-col", type=str, default=None, help="CSV'de hasta/grup sütunu (ör. patient_id)")
    p.add_argument("--out", type=str, default="best_model.pth", help="Checkpoint dosya adı")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.optuna:
        study = run_optuna(
            args.model,
            n_trials=args.trials,
            patience=args.patience,
            group_column=args.group_col,
        )
        bp = study.best_params
        train_final(
            args.model,
            lr=bp["lr"],
            batch_size=bp["batch_size"],
            max_epochs=bp["max_epochs"],
            patience=args.patience,
            group_column=args.group_col,
            out_name=args.out,
        )
    else:
        train_final(
            args.model,
            lr=args.lr,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            patience=args.patience,
            group_column=args.group_col,
            out_name=args.out,
        )


if __name__ == "__main__":
    main()
