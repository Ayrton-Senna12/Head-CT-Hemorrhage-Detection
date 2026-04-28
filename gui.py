"""
Modern dark-themed dashboard GUI — CustomTkinter + Grad-CAM + Drag&Drop
+ Confidence Bars + History Panel.
"""
from __future__ import annotations

import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
from torchvision import transforms

import config
from gradcam import apply_gradcam_on_image
from models import build_model

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT = "#6C5CE7"
ACCENT_LIGHT = "#A29BFE"
CARD_BG = "#1E1E2E"
SIDEBAR_BG = "#181825"
HISTORY_BG = "#1A1A2E"
TEXT_PRIMARY = "#CDD6F4"
TEXT_SECONDARY = "#6C7086"
SUCCESS = "#A6E3A1"
DANGER = "#F38BA8"
CHART_BLUE = "#89B4FA"
WARNING_YELLOW = "#F9E2AF"

GRAYSCALE_TOLERANCE = 15.0
CT_MEAN_RANGE = (35.0, 180.0)
CT_STD_RANGE = (35.0, 115.0)

AVAILABLE_MODELS = {
    "ConvNeXt-Tiny (Transfer Learning)": {
        "file": "best_convnext.pth",
        "accuracy": "96.7",
        "recall": "93.3",
        "precision": "100",
    },
    "Custom CNN (From Scratch)": {
        "file": "best_custom.pth",
        "accuracy": "83.3",
        "recall": "80.0",
        "precision": "85.7",
    },
}

MAX_HISTORY = 5


def _is_likely_ct_scan(img: Image.Image) -> tuple[bool, str]:
    """
    Validates whether an image looks like a head CT scan.
    Returns (is_valid, reason) where reason explains the rejection.
    """
    import numpy as np
    rgb = np.array(img.convert("RGB"), dtype=np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    ch_diff = (np.abs(r - g).mean() + np.abs(r - b).mean() + np.abs(g - b).mean()) / 3.0
    if ch_diff >= GRAYSCALE_TOLERANCE:
        return False, "Color image detected"

    gray = np.array(img.convert("L"), dtype=np.float32)
    m, s = float(gray.mean()), float(gray.std())

    if not (CT_MEAN_RANGE[0] <= m <= CT_MEAN_RANGE[1]):
        return False, f"Brightness out of CT range (mean={m:.0f})"

    if not (CT_STD_RANGE[0] <= s <= CT_STD_RANGE[1]):
        return False, f"Contrast out of CT range (std={s:.0f})"

    return True, ""


def load_classifier(checkpoint_path: Path, device: torch.device):
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    model_type = ckpt["model_type"]
    num_classes = int(ckpt.get("num_classes", config.NUM_CLASSES))
    image_size = int(ckpt.get("image_size", config.IMAGE_SIZE))
    class_names = list(ckpt.get("class_names", config.CLASS_NAMES))
    model = build_model(model_type, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, model_type, tfm, class_names


class MetricCard(ctk.CTkFrame):
    def __init__(self, master, title: str, value: str, color: str, **kw):
        super().__init__(master, fg_color=CARD_BG, corner_radius=12, **kw)
        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self, text=title, font=("Segoe UI", 11),
            text_color=TEXT_SECONDARY, anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 0))
        self.val_label = ctk.CTkLabel(
            self, text=f"%{value}", font=("Segoe UI Semibold", 22),
            text_color=color, anchor="w",
        )
        self.val_label.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 10))

    def update_value(self, value: str, color: str | None = None):
        self.val_label.configure(text=f"%{value}")
        if color:
            self.val_label.configure(text_color=color)


class ConfidenceBar(ctk.CTkFrame):
    """Labeled progress bar for a single class probability."""

    def __init__(self, master, label: str, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.grid_columnconfigure(1, weight=1)

        self.name_label = ctk.CTkLabel(
            self, text=label, font=("Segoe UI Semibold", 12),
            text_color=TEXT_PRIMARY, width=100, anchor="w",
        )
        self.name_label.grid(row=0, column=0, sticky="w", padx=(0, 8))

        self.bar = ctk.CTkProgressBar(
            self, height=18, corner_radius=8,
            fg_color="#313244", progress_color=TEXT_SECONDARY,
        )
        self.bar.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.bar.set(0)

        self.pct_label = ctk.CTkLabel(
            self, text="0%", font=("Segoe UI Semibold", 12),
            text_color=TEXT_PRIMARY, width=52, anchor="e",
        )
        self.pct_label.grid(row=0, column=2, sticky="e")

    def update(self, value: float, color: str):
        self.bar.set(value / 100.0)
        self.bar.configure(progress_color=color)
        self.pct_label.configure(text=f"{value:.1f}%", text_color=color)


class HistoryItem(ctk.CTkFrame):
    """Single row in the history panel: thumbnail + result."""

    def __init__(self, master, pil_thumb: Image.Image, cls_name: str,
                 confidence: float, is_hemorrhage: bool, **kw):
        super().__init__(master, fg_color=CARD_BG, corner_radius=10, height=56, **kw)
        self.grid_columnconfigure(1, weight=1)
        self.grid_propagate(False)

        self._photo = ImageTk.PhotoImage(pil_thumb)
        thumb_label = ctk.CTkLabel(self, image=self._photo, text="", width=48)
        thumb_label.grid(row=0, column=0, padx=(6, 6), pady=4)

        color = DANGER if is_hemorrhage else SUCCESS
        icon = "⚠" if is_hemorrhage else "✓"
        ctk.CTkLabel(
            self, text=f"{icon} {cls_name}  {confidence:.1f}%",
            font=("Segoe UI Semibold", 11), text_color=color, anchor="w",
        ).grid(row=0, column=1, sticky="w", padx=(0, 6), pady=4)


class _DnDMixin(ctk.CTk, TkinterDnD.DnDWrapper if HAS_DND else object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if HAS_DND:
            self.TkdndVersion = TkinterDnD._require(self)


class App(_DnDMixin):
    def __init__(self):
        super().__init__()
        self.title("Head CT — Hemorrhage Detection Dashboard")
        self.geometry("1220x720")
        self.minsize(1050, 640)
        self.configure(fg_color="#11111B")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_type = ""
        self.tfm = None
        self.class_names = list(config.CLASS_NAMES)
        self.image_path: Path | None = None
        self._photo = None
        self.history: deque = deque(maxlen=MAX_HISTORY)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()
        self._build_right_panel()

        self.model_menu.set(list(AVAILABLE_MODELS.keys())[0])
        self._on_model_change(None)

    # ── Left Sidebar ─────────────────────────────────────────
    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=260, fg_color=SIDEBAR_BG, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(
            sidebar, text="  HEAD CT", font=("Segoe UI Black", 18),
            text_color=ACCENT_LIGHT, anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(20, 4))
        ctk.CTkLabel(
            sidebar, text="  Hemorrhage Detection", font=("Segoe UI", 11),
            text_color=TEXT_SECONDARY, anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 20))

        ctk.CTkFrame(sidebar, height=1, fg_color="#313244").grid(
            row=2, column=0, sticky="ew", padx=12, pady=4)

        ctk.CTkLabel(
            sidebar, text="Model Selection", font=("Segoe UI Semibold", 12),
            text_color=TEXT_PRIMARY, anchor="w",
        ).grid(row=3, column=0, sticky="w", padx=16, pady=(16, 4))

        self.model_menu = ctk.CTkOptionMenu(
            sidebar, values=list(AVAILABLE_MODELS.keys()),
            command=self._on_model_change,
            fg_color=CARD_BG, button_color=ACCENT,
            button_hover_color=ACCENT_LIGHT,
            dropdown_fg_color=CARD_BG, dropdown_hover_color=ACCENT,
            dropdown_text_color=TEXT_PRIMARY, text_color=TEXT_PRIMARY,
            width=228, font=("Segoe UI", 11),
        )
        self.model_menu.grid(row=4, column=0, padx=16, pady=(0, 8))

        self.status_label = ctk.CTkLabel(
            sidebar, text="No model loaded", font=("Segoe UI", 10),
            text_color=TEXT_SECONDARY, anchor="w",
        )
        self.status_label.grid(row=5, column=0, sticky="w", padx=16, pady=(0, 16))

        ctk.CTkFrame(sidebar, height=1, fg_color="#313244").grid(
            row=6, column=0, sticky="ew", padx=12, pady=4)

        ctk.CTkLabel(
            sidebar, text="Model Metrics", font=("Segoe UI Semibold", 12),
            text_color=TEXT_PRIMARY, anchor="w",
        ).grid(row=7, column=0, sticky="w", padx=16, pady=(12, 8))

        mf = ctk.CTkFrame(sidebar, fg_color="transparent")
        mf.grid(row=8, column=0, sticky="ew", padx=12)
        mf.grid_columnconfigure(0, weight=1)
        self.card_acc = MetricCard(mf, "Accuracy", "--", CHART_BLUE)
        self.card_acc.grid(row=0, column=0, sticky="ew", pady=3)
        self.card_rec = MetricCard(mf, "Recall", "--", SUCCESS)
        self.card_rec.grid(row=1, column=0, sticky="ew", pady=3)
        self.card_prec = MetricCard(mf, "Precision", "--", ACCENT_LIGHT)
        self.card_prec.grid(row=2, column=0, sticky="ew", pady=3)

        self.btn_select = ctk.CTkButton(
            sidebar, text="Select File", command=self._pick_file,
            fg_color=ACCENT, hover_color=ACCENT_LIGHT,
            text_color="white", font=("Segoe UI Semibold", 13),
            height=40, corner_radius=10,
        )
        self.btn_select.grid(row=11, column=0, padx=16, pady=(0, 16), sticky="sew")

    # ── Center Main ──────────────────────────────────────────
    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color="#11111B", corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)

        self.image_frame = ctk.CTkFrame(main, fg_color=CARD_BG, corner_radius=14)
        self.image_frame.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=(16, 8))
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)

        self.canvas = ctk.CTkLabel(
            self.image_frame,
            text="Drag & drop an image here\nor use the  Select File  button",
            font=("Segoe UI", 14), text_color=TEXT_SECONDARY,
            fg_color="transparent",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        if HAS_DND:
            self.image_frame.drop_target_register(DND_FILES)
            self.image_frame.dnd_bind("<<Drop>>", self._on_drop)

        # --- Result + Confidence Bars ---
        result_frame = ctk.CTkFrame(main, fg_color=CARD_BG, corner_radius=14, height=130)
        result_frame.grid(row=1, column=0, sticky="ew", padx=(16, 8), pady=(8, 16))
        result_frame.grid_columnconfigure(0, weight=1)
        result_frame.grid_propagate(False)

        top_row = ctk.CTkFrame(result_frame, fg_color="transparent")
        top_row.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 4))
        top_row.grid_columnconfigure(1, weight=1)

        self.result_icon = ctk.CTkLabel(
            top_row, text="◉", font=("Segoe UI", 26),
            text_color=TEXT_SECONDARY, width=40,
        )
        self.result_icon.grid(row=0, column=0, padx=(0, 8))

        self.result_class = ctk.CTkLabel(
            top_row, text="No prediction yet",
            font=("Segoe UI Semibold", 15), text_color=TEXT_PRIMARY, anchor="w",
        )
        self.result_class.grid(row=0, column=1, sticky="w")

        self.gradcam_hint = ctk.CTkLabel(
            top_row, text="", font=("Segoe UI", 10),
            text_color=ACCENT_LIGHT, anchor="e",
        )
        self.gradcam_hint.grid(row=0, column=2, sticky="e", padx=(8, 0))

        bars_frame = ctk.CTkFrame(result_frame, fg_color="transparent")
        bars_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=(4, 12))
        bars_frame.grid_columnconfigure(0, weight=1)

        self.bar_normal = ConfidenceBar(bars_frame, label="Normal")
        self.bar_normal.grid(row=0, column=0, sticky="ew", pady=2)
        self.bar_hemorrhage = ConfidenceBar(bars_frame, label="Hemorrhage")
        self.bar_hemorrhage.grid(row=1, column=0, sticky="ew", pady=2)

    # ── Right History Panel ──────────────────────────────────
    def _build_right_panel(self):
        panel = ctk.CTkFrame(self, width=210, fg_color=HISTORY_BG, corner_radius=0)
        panel.grid(row=0, column=2, sticky="nsew")
        panel.grid_propagate(False)
        panel.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            panel, text="  Recent Scans", font=("Segoe UI Semibold", 13),
            text_color=TEXT_PRIMARY, anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=12, pady=(16, 8))

        self.history_scroll = ctk.CTkScrollableFrame(
            panel, fg_color="transparent", corner_radius=0,
        )
        self.history_scroll.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 12))
        self.history_scroll.grid_columnconfigure(0, weight=1)

        self.history_widgets: list[HistoryItem] = []

    # ── Logic ────────────────────────────────────────────────
    def _on_drop(self, event):
        raw = event.data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        path = Path(raw)
        if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            self.image_path = path
            self._run_inference(path)

    def _on_model_change(self, _choice):
        name = self.model_menu.get()
        info = AVAILABLE_MODELS.get(name)
        if not info:
            return
        ckpt_path = config.CHECKPOINTS_DIR / info["file"]
        if not ckpt_path.is_file():
            self.status_label.configure(text="Checkpoint not found!", text_color=DANGER)
            self.model = None
            return
        try:
            self.model, self.model_type, self.tfm, self.class_names = load_classifier(
                ckpt_path, self.device
            )
            self.status_label.configure(text=f"✓ {info['file']}", text_color=SUCCESS)
            self.card_acc.update_value(info["accuracy"], CHART_BLUE)
            self.card_rec.update_value(info["recall"], SUCCESS)
            self.card_prec.update_value(info["precision"], ACCENT_LIGHT)
            if self.image_path and self.image_path.is_file():
                self._run_inference(self.image_path)
        except Exception as e:  # noqa: BLE001
            self.status_label.configure(text="Loading error!", text_color=DANGER)
            self.model = None

    def _pick_file(self):
        if self.model is None:
            return
        path = filedialog.askopenfilename(
            title="Select test image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        self.image_path = Path(path)
        self._run_inference(self.image_path)

    def _show_image(self, pil_img: Image.Image):
        disp = pil_img.copy()
        max_w = max(self.image_frame.winfo_width() - 20, 300)
        max_h = max(self.image_frame.winfo_height() - 20, 200)
        disp.thumbnail((max_w, max_h))
        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.configure(image=self._photo, text="")

    def _add_to_history(self, path: Path, cls_name: str, confidence: float, is_hemorrhage: bool):
        thumb = Image.open(path).convert("RGB")
        thumb.thumbnail((44, 44))
        self.history.appendleft((thumb, cls_name, confidence, is_hemorrhage))

        for w in self.history_widgets:
            w.destroy()
        self.history_widgets.clear()

        for i, (th, cn, cf, ih) in enumerate(self.history):
            item = HistoryItem(self.history_scroll, th, cn, cf, ih)
            item.grid(row=i, column=0, sticky="ew", pady=3)
            self.history_widgets.append(item)

    def _run_inference(self, path: Path):
        if self.model is None:
            return
        try:
            original = Image.open(path).convert("RGB")
            x = self.tfm(original).unsqueeze(0).to(self.device)
            logits = self.model(x)
            prob = F.softmax(logits, dim=1)[0]
            idx = int(prob.argmax().item())
            cls_name = self.class_names[idx]
            confidence = float(prob[idx].item()) * 100.0

            prob_normal = float(prob[0].item()) * 100.0
            prob_hemorrhage = float(prob[1].item()) * 100.0

            is_ct, reject_reason = _is_likely_ct_scan(original)
            if not is_ct:
                self.result_icon.configure(text="✕", text_color=WARNING_YELLOW)
                self.result_class.configure(
                    text="Invalid Image! Please upload a Head CT slice.",
                    text_color=WARNING_YELLOW,
                )
                self.bar_normal.update(prob_normal, TEXT_SECONDARY)
                self.bar_hemorrhage.update(prob_hemorrhage, TEXT_SECONDARY)
                self.gradcam_hint.configure(text=reject_reason)
                self._show_image(original)
                return

            is_hemorrhage = idx == 1
            if is_hemorrhage:
                color = DANGER
                icon = "⚠"
            else:
                color = SUCCESS
                icon = "✓"

            self.result_icon.configure(text=icon, text_color=color)
            self.result_class.configure(
                text=f"{cls_name}  —  {confidence:.1f}% confidence  |  {path.name}",
                text_color=color,
            )

            normal_bar_color = SUCCESS if not is_hemorrhage else TEXT_SECONDARY
            hemorrhage_bar_color = DANGER if is_hemorrhage else TEXT_SECONDARY
            self.bar_normal.update(prob_normal, normal_bar_color)
            self.bar_hemorrhage.update(prob_hemorrhage, hemorrhage_bar_color)

            if is_hemorrhage:
                x_cam = self.tfm(original).unsqueeze(0).to(self.device)
                overlay, _ = apply_gradcam_on_image(
                    self.model, self.model_type, x_cam, original, target_class=1
                )
                self._show_image(overlay)
                self.gradcam_hint.configure(text="Grad-CAM active  ◉")
            else:
                self._show_image(original)
                self.gradcam_hint.configure(text="")

            self._add_to_history(path, cls_name, confidence, is_hemorrhage)

        except Exception as e:  # noqa: BLE001
            self.result_class.configure(text="Error!", text_color=DANGER)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
