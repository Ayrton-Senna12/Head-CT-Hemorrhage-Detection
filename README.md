# Head CT Hemorrhage Detection

Deep learning tabanlı beyin BT görüntülerinden kanama tespiti projesi. İki farklı model mimarisi ile eğitim, değerlendirme ve Grad-CAM görselleştirme içerir.

## Proje Yapısı

| Dosya | Açıklama |
|---|---|
| `config.py` | Proje yolları ve sabitler |
| `data_pipeline.py` | Veri yükleme, augmentasyon, grup bazlı train/val/test bölme |
| `models.py` | ConvNeXt-Tiny (transfer learning) ve Custom CNN modelleri |
| `train.py` | Eğitim döngüsü, erken durdurma, Optuna hiperparametre araması |
| `evaluate.py` | Test seti metrikleri ve confusion matrix |
| `gradcam.py` | Grad-CAM ısı haritası görselleştirmesi |
| `gui.py` | CustomTkinter tabanlı tahmin arayüzü (drag & drop destekli) |
| `labels.csv` | Görüntü etiketleri (0: Normal, 1: Hemorrhage) |

## Modeller

- **ConvNeXt-Tiny (Transfer Learning):** ImageNet ağırlıkları üzerine fine-tune. Test accuracy ~%96.7.
- **Custom CNN:** Sıfırdan eğitilen 4-katmanlı CNN (Conv + BN + ReLU + Pool). Test accuracy ~%83.3.

## Kurulum

```bash
pip install -r requirements.txt
```

## Veri Seti

[Head CT — Hemorrhage](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage) veri setini indirip proje kök dizinine `head_ct/` klasörü olarak yerleştirin:

```
Project2/
├── head_ct/        ← CT görüntüleri buraya
├── labels.csv
├── config.py
└── ...
```

## Kullanım

### Eğitim (Optuna ile hiperparametre araması)

```bash
python train.py --model convnext_tiny --optuna --trials 12
```

### Eğitim (manuel parametrelerle)

```bash
python train.py --model convnext_tiny --lr 3e-4 --batch-size 16 --epochs 40
python train.py --model custom --lr 3e-4 --batch-size 16 --epochs 40 --out best_custom.pth
```

### Değerlendirme

```bash
python evaluate.py --checkpoint outputs/checkpoints/best_model.pth
```

### GUI (Tahmin Arayüzü)

```bash
python gui.py
```

## Sonuçlar

| Model | Accuracy | Recall | Precision |
|---|---|---|---|
| ConvNeXt-Tiny | %96.7 | %93.3 | %100 |
| Custom CNN | %83.3 | %80.0 | %85.7 |

## Grup Üyeleri

- 211015064
- 211015051
- 211015044
- 211015024
