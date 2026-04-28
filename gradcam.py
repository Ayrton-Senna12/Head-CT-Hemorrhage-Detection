"""
Grad-CAM: Modelin hangi bölgeye bakarak karar verdiğini görselleştirir.
Kanama tespit edilen alanı ısı haritası + kırmızı dikdörtgen olarak gösterir.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Belirtilen katmanın aktivasyonlarından Grad-CAM haritası üretir."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output):
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_target_layer(model: torch.nn.Module, model_type: str) -> torch.nn.Module:
    """Model tipine göre Grad-CAM hedef katmanını döndürür."""
    name = model_type.lower().strip()
    if "convnext" in name:
        return model.features[-1]
    return model.conv4[0]


def apply_gradcam_on_image(
    model: torch.nn.Module,
    model_type: str,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    target_class: Optional[int] = None,
    threshold: float = 0.4,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Orijinal görüntü üzerine Grad-CAM ısı haritası + kırmızı dikdörtgen çizer.
    Döndürür: (çizimli PIL Image, cam haritası)
    """
    target_layer = get_target_layer(model, model_type)
    gc = GradCAM(model, target_layer)

    try:
        cam = gc.generate(input_tensor, target_class=target_class)
    finally:
        gc.remove_hooks()

    img_np = np.array(original_image.convert("RGB"))
    h, w = img_np.shape[:2]

    cam_resized = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * img_np + 0.45 * heatmap)

    binary = (cam_resized >= threshold).astype(np.uint8)
    if binary.sum() > 0:
        coords = np.column_stack(np.where(binary > 0))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        pad = 5
        y_min = max(0, y_min - pad)
        x_min = max(0, x_min - pad)
        y_max = min(h - 1, y_max + pad)
        x_max = min(w - 1, x_max + pad)

        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    result = Image.fromarray(overlay)
    return result, cam_resized
