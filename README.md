# CNN for Flower Type Identification

A compact TensorFlow/Keras project that trains a Convolutional Neural Network (CNN) to classify five flower types, **daisy, dandelion, rose, sunflower, tulip** with strong validation performance and practical evaluation tooling (confusion matrix, per-class report, hard-case gallery), plus ablations with **L2 regularization** and **Batch Normalization**.

> ✅ Best validation accuracy: **75.31%** at epoch **12** (macro-F1 ≈ **0.7515**) on an 80/10/10 split.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Environment & Setup](#environment--setup)
- [Project Structure](#project-structure)
- [Training Pipeline](#training-pipeline)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Evaluate & Inspect Errors](#evaluate--inspect-errors)
- [Single-Image Inference](#single-image-inference)
- [Performance & Model Size](#performance--model-size)
- [Troubleshooting](#troubleshooting)
- [Reproducibility](#reproducibility)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview
This repo implements a straightforward image-classification workflow:
1. **Dataset audit** (counts per class, varied image sizes)
2. **Deterministic split** (80/10/10 using `split-folders`)
3. **Augmented training** + **class weighting** for imbalance
4. **Baseline CNN** + ablations (**L2**, **BatchNorm**)
5. **Evaluation** (accuracy, macro-F1, confusion matrix, hardest misclassifications)
6. **Single-image inference** example

---

## Dataset
- **Classes (5)**: `daisy`, `dandelion`, `rose`, `sunflower`, `tulip`
- **Total images**: **4317**
- **Distribution**:
  - daisy: **764** (17.7%)
  - dandelion: **1052** (24.4%)
  - rose: **784** (18.2%)
  - sunflower: **733** (17.0%)
  - tulip: **984** (22.8%)
- **Image sizes**: 386 unique (e.g., common W×H: 320×240, 500×333, 240×159)
- **Split (80/10/10) counts**:
  - **train**: 3019 (daisy 534, dandelion 736, rose 548, sunflower 513, tulip 688)  
  - **val**: 644 (daisy 114, dandelion 157, rose 117, sunflower 109, tulip 147)  
  - **test**: 654 (daisy 116, dandelion 159, rose 119, sunflower 111, tulip 149)

> Note: the code normalizes and resizes images to **224×224** before training.

---

## Environment & Setup

### Python & key packages
- Python 3.10–3.12
- TensorFlow **2.19.0**
- Keras **3.10.0**
- scikit-learn, Pillow, pandas, numpy, matplotlib, seaborn
- `split-folders==0.5.1`

### Quick start
```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
tensorflow==2.19.0
keras==3.10.0
split-folders==0.5.1
numpy>=1.26,<2.2
pandas>=2.0
matplotlib
seaborn
scikit-learn
Pillow
```

---

## Project Structure
```
.
├─ data/
│  └─ flowers/
│     ├─ daisy/...
│     ├─ dandelion/...
│     ├─ rose/...
│     ├─ sunflower/...
│     └─ tulip/...
├─ notebooks/
│  └─ cnn_flower_identification.ipynb
├─ src/
│  ├─ train.py
│  ├─ evaluate.py
│  └─ predict.py
├─ README.md
└─ requirements.txt
```

---

## Training Pipeline

**Image size**: `224×224` **Batch size**: `32` **Seed**: `42`  
**Augmentation (train only)**: rotation 20°, shifts 0.1, shear 0.1, zoom 0.1, horizontal flip  
**Rescaling**: all splits `1./255`  
**Loss**: `categorical_crossentropy` **Optimizer**: `adam`  
**Early stopping**: monitor `val_accuracy`, patience `5`, `restore_best_weights=True`  
**Class weights**: computed to offset imbalance

---

## Model Architectures

### Baseline CNN
```
Input(224,224,3)
→ Conv2D(32,3,relu) → MaxPool(2)
→ Conv2D(64,3,relu) → MaxPool(2)
→ Conv2D(128,3,relu) → MaxPool(2)
→ Flatten
→ Dense(256,relu)
→ Dense(5,softmax)
```

### Variants
- **L2 Regularization**
- **BatchNorm**
- **Lightweight (GAP + Dropout + Dense)**

---

## Results
- **Baseline val acc**: 0.7531 (macro-F1: 0.7515)  
- **Best epoch**: 12  
- **Per-class**: rose = hardest, sunflower = strongest

| Model     | Val Acc | Macro-F1 |
|-----------|---------|----------|
| baseline  | 0.7531  | 0.7515   |
| l2        | 0.7267  | 0.7252   |
| bn        | 0.6289  | 0.6154   |

---

## Evaluate & Inspect Errors
- Confusion matrix heatmap  
- Classification report  
- Misclassified samples gallery

---

## Single-Image Inference
```python
from PIL import Image, ImageOps
import numpy as np

img = Image.open("phone_rose.jpeg")
img = ImageOps.exif_transpose(img).convert("RGB").resize((224,224))
x = np.expand_dims(np.array(img)/255.0, axis=0)
probs = model.predict(x, verbose=0)[0]
print(dict(zip(labels, probs)))
```

---

## Performance & Model Size
- **Baseline params**: ~22.25M (~84.9 MB)
- **Lightweight alt**: ~110k params

---

## Troubleshooting
- Dataset path issues: ensure `data/flowers/...` exists
- macOS Metal TF warnings: harmless unless crash → run on CPU
- Keep `ImageOps.exif_transpose` for phone images

---

## Acknowledgements
- Flowers dataset (public demo/tutorial sources)  
- TensorFlow/Keras community 
