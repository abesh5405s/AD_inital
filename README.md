# Alzheimer's Disease Classification — Optimized CNN

A deep-learning project that classifies brain MRI images into **four Alzheimer's Disease (AD) stages** using an **EfficientNetB0 transfer-learning** model trained in Google Colab.

---

## Classes

| Index | Class Name |
|-------|------------|
| 0 | MildDemented |
| 1 | ModerateDemented |
| 2 | NonDemented |
| 3 | VeryMildDemented |

---

## Dataset

The dataset contains brain MRI images organised into the four class directories listed above.

**Google Drive folder:** <https://drive.google.com/drive/folders/1mpwG9EZBcD-tyJs9Fj1Rkq0O6eFQRQVX>

After mounting your Drive in Colab, update the `DATASET_PATH` variable in the notebook to point at the folder containing the four class subdirectories.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `AD_MAJOR_CNN_technique.ipynb` | Original CNN baseline model |
| `AD_Optimized_Model.ipynb` | **Optimized model** (EfficientNetB0 + fine-tuning) |

---

## Model Architecture (`AD_Optimized_Model.ipynb`)

```
Input (224 × 224 × 3)
    ↓
EfficientNetB0 (ImageNet weights, initially frozen)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(256, relu) → Dropout(0.4) → BatchNormalization
    ↓
Dense(128, relu) → Dropout(0.2)
    ↓
Dense(4, softmax)
```

### Key Optimizations Over the Baseline

| Feature | Detail |
|---------|--------|
| **Transfer Learning** | EfficientNetB0 pre-trained on ImageNet |
| **Two-phase training** | Phase 1 — frozen base; Phase 2 — top-30 layers unfrozen |
| **Augmentation** | Rotation, shift, shear, zoom, horizontal flip, brightness |
| **Class imbalance** | Computed class weights passed to `model.fit()` |
| **Learning rate** | `ReduceLROnPlateau` (factor 0.5, patience 3) |
| **Early stopping** | Monitors `val_accuracy` with `restore_best_weights=True` |
| **Regularisation** | Dropout (0.4 / 0.2) + BatchNormalization |
| **Model checkpoint** | Best model saved to Google Drive automatically |

---

## How to Run

1. Open `AD_Optimized_Model.ipynb` in **Google Colab** (set runtime to GPU).
2. Run **Cell 1** to install dependencies.
3. Run **Cell 2** (`drive.mount(...)`) and authorise Drive access.
4. In **Cell 3**, update `DATASET_PATH` to the location of the dataset folder in your Drive, e.g.:
   ```python
   DATASET_PATH = '/content/drive/MyDrive/AD_Dataset'
   ```
5. Run all remaining cells in order.

Best-model checkpoints and evaluation plots are saved to `/content/drive/MyDrive/AD_Models/`.

---

## Expected Output

- Training/validation accuracy & loss curves
- Confusion matrix for all 4 classes
- Per-class precision, recall, and F1-score (classification report)
- Sample predictions with actual vs predicted labels and confidence scores
- Best model saved as `ad_best_model.keras` on Google Drive

---

## Dependencies

| Package | Purpose |
|---------|---------|
| TensorFlow ≥ 2.12 | Model training & inference |
| Keras (bundled) | High-level API, `EfficientNetB0`, callbacks |
| scikit-learn | Class weights, confusion matrix, classification report |
| NumPy | Array operations |
| Matplotlib / Seaborn | Plotting |
| Pillow | Image I/O |

Install in Colab with:
```bash
pip install tensorflow scikit-learn matplotlib seaborn Pillow
```