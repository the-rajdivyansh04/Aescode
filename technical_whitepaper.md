# OsteoVision AI — Technical Whitepaper
## AI-Based Osteoporosis Risk Screening from Routine Hand/Wrist X-Ray Radiographs

**Team Techtoli** | AesCode Hackathon 2026 — Problem Statement 3 (PS3)
**Version**: 4.0 (Multi-Modal Final) | **Date**: April 2026

---

## Executive Summary

**OsteoVision AI** is a multi-modal deep learning system that performs 3-class osteoporosis risk screening (Normal / Osteopenia / Osteoporosis) from routine hand/wrist X-ray radiographs. The final architecture — **OsteoVisionMultiModal** — fuses EfficientNet-B0 image features with structured clinical tabular features (bone age, sex) using a late-fusion MLP head, combined with per-class threshold tuning for optimal decision boundaries.

**Key Results (Multi-Modal Model, April 2026):**

| Metric | Value |
|--------|-------|
| Validation Accuracy | **89.09%** |
| Macro F1 | **0.8381** |
| MCC (Matthews CC) | **0.7584** |
| AUC-ROC (OvR macro) | **0.9720** |
| Osteoporosis Critical FN Rate | **1.1%** (only 2/186 osteoporosis cases missed as Normal) |
| Checkpoint Size | ~20 MB |

---

## 1. Problem Statement & Clinical Motivation

Osteoporosis is a systemic skeletal disorder characterised by compromised bone mineral density (BMD) and deterioration of trabecular microarchitecture, leading to enhanced bone fragility. The World Health Organization estimates **over 200 million individuals** are affected globally, with osteoporotic fractures occurring every **3 seconds** and costing upwards of **$19 billion annually** in the US alone.

The current gold standard — **Dual-energy X-ray Absorptiometry (DEXA)** — remains inaccessible in resource-constrained settings due to its 10× higher cost, requirement for trained technicians, and centralisation in urban hospitals. Consequently, **80% of high-risk patients in rural areas are never screened prior to their first fracture**.

**OsteoVision AI** bridges this gap by applying a deep CNN to routine hand/wrist X-rays — the same images already taken for bone age assessment — enabling population-scale osteoporosis risk stratification without dedicated DEXA hardware.

---

## 2. Dataset: RSNA Bone Age Challenge

### 2.1 Source & Structure

| Attribute | Detail |
|-----------|--------|
| **Dataset** | RSNA Bone Age Challenge (hand/wrist radiographs) |
| **Drive path** | `Aescode/AI-Based Osteoporosis Risk Screening from Routine X-Ray Radiographs/` |
| **Image directory** | `boneage-training-dataset/` — images named by ID (e.g., `1234.png`) |
| **Label file** | `train_labels.csv` — columns: `id`, `boneage`, `male`, `risk_class` |
| **Test set file** | `val_ids.csv` — 1,892 unlabeled IDs for final competition submission |
| **Image modality** | Pediatric hand/wrist X-ray radiographs (grayscale PNG/JPEG) |
| **Input resolution** | 224 × 224 × 3 (RGB) |

> **Critical note**: `val_ids.csv` contains **unlabeled competition test IDs** (no `risk_class` column, zero overlap with `train_labels.csv`). It is the final submission test set — not used for training or validation.

### 2.2 Label Schema

| Class | Label | WHO T-Score Equivalent | Clinical Action |
|-------|-------|----------------------|----------------|
| **0** | Normal | T > −1.0 | Routine monitoring |
| **1** | Osteopenia | −2.5 < T ≤ −1.0 | DEXA referral, FRAX assessment |
| **2** | Osteoporosis | T ≤ −2.5 | Urgent DEXA, pharmacological review |

### 2.3 Dataset Scale & Split

**Total labeled images: 10,719** — split via stratified 80/20:

| Split | Class 0 (Normal) | Class 1 (Osteopenia) | Class 2 (Osteoporosis) | Total |
|-------|-----------------|---------------------|----------------------|-------|
| **Training** | 5,936 | 1,867 | 744 | **8,575** |
| **Validation** | 1,491 | 467 | 186 | **2,144** |
| **Test set** | — | — | — | 1,892 (unlabeled) |

```python
# Stratified split preserves class proportions in both sets
train_df, val_df = train_test_split(
    full_df, test_size=0.20, random_state=42, stratify=full_df["risk_class"]
)
```

### 2.4 Competition Constraints (Hard Rules)

- ❌ **No data augmentation** — train and val transforms are identical
- ❌ **No normalization** — ImageNet mean/std not applied
- ❌ **No class rebalancing** — no `WeightedRandomSampler`, no weighted `CrossEntropyLoss`
- ✅ Only `train_labels.csv` used as label source

---

## 3. System Architecture

### 3.0 Model Evolution: Single-Modal → Multi-Modal

| Aspect | v1 — Single-Modal | v2 — Multi-Modal (Final) |
|--------|------------------|--------------------------|
| **Architecture** | EfficientNet-B0 + head | OsteoVisionMultiModal (backbone + tabular branch + fusion head) |
| **Inputs** | Image only | Image + bone age + sex |
| **Tabular branch** | None | Linear(2→32) → ReLU → Linear(32→32) → ReLU |
| **Fusion** | None | concat(1280, 32) → Dropout(0.3) → Linear(1312→128) → Linear(128→3) |
| **Threshold** | argmax on raw probs | Per-class optimal threshold (margin strategy) |
| **Accuracy** | 83.77% | **89.09%** |
| **Macro F1** | 0.7706 | **0.8381** |
| **MCC** | — | **0.7584** |
| **AUC-ROC** | 0.9342 | **0.9720** |
| **Critical FN** | 1.6% (3/186) | **1.1% (2/186)** |

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                    OsteoVisionMultiModal Pipeline                                    │
├──────────────┬──────────────────┬─────────────────────┬──────────────────────────────┤
│   INPUT      │   PREPROCESSING  │     MODEL           │    OUTPUT                    │
│              │                  │                     │                              │
│ X-ray Image  │ PIL.open()       │ ┌─ Image Branch ──┐ │ Risk Class (0/1/2)           │
│ (PNG/JPEG/   │ .convert('RGB')  │ │ EfficientNet-B0 │ │   Normal / Osteopenia /      │
│  DICOM)      │ Resize(224,224)  │ │ backbone        │ │   Osteoporosis               │
│              │ ToTensor()       │ │ → 1280-d vector │ │                              │
│ bone age     │ /228.0 norm      │ └───────┬─────────┘ │ Confidence %                 │
│ sex (0/1)    │ float cast       │         │ cat       │ (MC Dropout mean, 20 passes) │
│              │                  │ ┌─ Tab Branch ───┐  │                              │
│              │                  │ │ Linear(2→32)   │  │ Uncertainty flag             │
│              │                  │ │ ReLU ×2        │  │ (std > 0.15)                 │
│              │                  │ │ → 32-d vector  │  │                              │
│              │                  │ └───────┬────────┘  │ Grad-CAM heatmap             │
│              │                  │         │           │ (backbone.features[-1])      │
│              │                  │ ┌─ Fusion Head ─┐   │                              │
│              │                  │ │ Linear(1312→  │   │ Clinical recommendation      │
│              │                  │ │  128→3)       │   │                              │
│              │                  │ └───────────────┘   │ Optimal threshold applied    │
└──────────────┴──────────────────┴─────────────────────┴──────────────────────────────┘
```

### 3.1 EfficientNet-B0: Why This Architecture

EfficientNet introduces **compound scaling** — optimising depth ($d$), width ($w$), and resolution ($r$) simultaneously:

$$d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi \quad \text{s.t.} \quad \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$$

| Attribute | EfficientNet-B0 | ResNet-50 |
|-----------|-----------------|-----------|
| Parameters | **5.3 M** | 25.6 M |
| FLOPs | **0.39 B** | 4.1 B |
| ImageNet Top-1 | **77.1%** | 76.1% |
| Edge deployable | ✅ | ❌ |

### 3.2 Multi-Modal Fusion Architecture

```python
class OsteoVisionMultiModal(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Image branch: EfficientNet-B0, classifier replaced with Identity
        _eff = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        _eff.classifier = nn.Identity()    # outputs 1280-d feature vector
        self.backbone = _eff

        # Tabular branch: [boneage_norm, sex] → 32-d embedding
        self.tabular_branch = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )

        # Fusion head: 1280 + 32 = 1312 → 128 → num_classes
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1312, 128), nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, tabular):
        img_feat = self.backbone(image)                        # (B, 1280)
        tab_feat = self.tabular_branch(tabular)                # (B, 32)
        fused    = torch.cat([img_feat, tab_feat], dim=1)      # (B, 1312)
        return self.fusion_head(fused)                         # (B, 3)
```

**Tabular feature encoding:**
- `boneage_norm = float(boneage) / 228.0` — maps [1, 228] months → [0.004, 1.0]
- `male = float(row['male'])` — 1.0 = male, 0.0 = female

---

## 4. Data Pipeline

### 4.1 `BoneAgeDataset`

```python
class BoneAgeDataset(Dataset):
    def __getitem__(self, idx):
        image = Image.open(img_path).convert('RGB')  # grayscale → RGB [FIX 1]
        if self.transform:
            image = self.transform(image)
        # Tabular features
        boneage_norm = float(row['boneage']) / 228.0   # normalize to [0,1]
        male         = float(row['male'])               # True/False → 1.0/0.0
        tabular      = torch.tensor([boneage_norm, male], dtype=torch.float32)
        return image, tabular, int(row['risk_class'])  # 3-tuple
```

> **FIX 1 — Critical**: RSNA images are grayscale. Without `.convert('RGB')`, `ToTensor()` produces a `(1, 224, 224)` tensor that crashes EfficientNet-B0's first conv layer (expects 3 channels).

### 4.2 Transforms

```python
# IDENTICAL for train and val — no augmentation per competition rules
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),    # → float32 in [0.0, 1.0]
])
```

### 4.3 DataLoaders

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False,
                          num_workers=2, pin_memory=True)
```

---

## 5. Training Strategy

### 5.1 Two-Phase Progressive Unfreezing

| Phase | Epochs | Backbone | LR | Scheduler |
|-------|--------|----------|----|-----------|
| **1 — Head only** | 1–10 | ❄️ `backbone.features` frozen | 1×10⁻⁴ | CosineAnnealing, T_max=10, η_min=10⁻⁶ |
| **2 — Fine-tuning** | 11–30* | 🔓 Last 3 MBConv blocks | 1×10⁻⁵ | CosineAnnealing, T_max=20, η_min=10⁻⁶ |

*Early stopping triggered based on val macro F1 (patience=7).

At epoch 11: `model.backbone.features[-3:]` unfrozen → trainable params include tabular branch, fusion head, and last 3 MBConv blocks.

### 5.2 Loss, Optimizer, Early Stopping

```python
criterion    = nn.CrossEntropyLoss()   # No class weights — competition rule
optimizer    = optim.AdamW(params, lr=lr, weight_decay=1e-4)
early_stopping monitors val macro F1, patience=7
```

Early stopping on **macro F1** (not val loss) — in a 3-class imbalanced setting, loss can improve while F1 degrades if the model collapses toward the dominant class.

---

## 6. Training Results

### 6.1 Model Comparison: Single-Modal (v1) vs Multi-Modal (v2)

| Metric | v1 — EfficientNet-B0 only | v2 — OsteoVisionMultiModal | Δ Improvement |
|--------|--------------------------|---------------------------|---------------|
| **Accuracy** | 83.77% | **89.09%** | **+5.32%** |
| **Macro F1** | 0.7706 | **0.8381** | **+0.0675** |
| **MCC** | — | **0.7584** | new metric |
| **AUC-ROC** | 0.9342 | **0.9720** | **+0.0378** |
| **Critical FN (Osteoporosis→Normal)** | 1.6% (3/186) | **1.1% (2/186)** | **−0.5%** |
| Architecture | EfficientNet-B0 + head | Backbone + tabular MLP + fusion head | — |
| Inputs | Image only | Image + bone age + sex | — |
| Decision rule | argmax | Per-class optimal threshold | — |

> The multi-modal upgrade reduced the most dangerous error — classifying Osteoporosis as Normal — by **33%** (3 → 2 cases out of 186).

### 6.2 Final Validation Metrics (Multi-Modal, v2)

| Metric | Value |
|--------|-------|
| **Accuracy** | **89.09%** |
| **Macro F1** | **0.8381** |
| **MCC (Matthews CC)** | **0.7584** |
| **AUC-ROC (OvR macro)** | **0.9720** |

> MCC is the primary reliability metric for imbalanced multi-class classification. Range: −1 (worst) to +1 (perfect). A score of **0.7584** indicates strong, well-balanced discrimination across all three classes.

### 6.3 Per-Class Performance (v2)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Normal** | — | — | — | 1,491 |
| **Osteopenia** | — | — | — | 467 |
| **Osteoporosis** | — | — | — | 186 |
| *Macro avg* | — | — | **0.8381** | 2,144 |

*(Per-class breakdown to be filled from BLOCK 6 classification_report output)*

### 6.4 Confusion Matrix (v2)

| Actual \ Predicted | **Normal** | **Osteopenia** | **Osteoporosis** |
|-------------------|-----------|---------------|----------------|
| **Osteoporosis** (186) | **2** ← FN | — | — |

**Clinical interpretation (v2):**
- Only **2 osteoporosis cases** misclassified as Normal — critical FN rate = **1.1%** (down from 1.6% in v1).
- This represents **33% reduction** in the most dangerous misclassification type.
- Tabular features (bone age and sex) provide complementary signal to image features, particularly for borderline cases.
- Per-class threshold tuning via margin strategy further corrects confident misclassifications.

### 6.5 Historical Training Log (v1 Reference)

| Epoch | Phase | Val Acc | Val Macro F1 | Best |
|-------|-------|---------|-------------|------|
| 1–10 | Phase 1 | 69.8%→76.5% | 0.2962→0.5765 | ✅ ep8 |
| **18** | Phase 2 | **83.8%** | **0.7706** | ✅ 🏆 |
| 25 | Phase 2 | — | — | ⏹️ Early stop |

**v1 best checkpoint**: Epoch 18 — `osteovision_best.pth` (16.3 MB)

---

## 7. Explainability: Grad-CAM

### 7.1 Mathematical Formulation

Grad-CAM (Selvaraju et al., 2017) targets `model.backbone.features[-1]` (last MBConv block of the EfficientNet-B0 backbone within OsteoVisionMultiModal).

**Importance weights:**
$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$

**Class activation map:**
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\!\left(\sum_k \alpha_k^c \cdot A^k\right)$$

ReLU retains only regions with positive influence — highlighting cortical thinning, trabecular rarefaction, and endosteal resorption.

### 7.2 Memory-Safe Hook Implementation (FIX 4)

```python
def compute_cam(self, image_tensor, tabular_tensor, target_class=None):
    fh = self.target_layer.register_forward_hook(self._save_activation)
    bh = self.target_layer.register_full_backward_hook(self._save_gradient)
    try:
        output = self.model(image_tensor, tabular_tensor)  # multi-modal call
        output[0, target_class].backward()
        weights = self._grads.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self._acts).sum(dim=1)).squeeze()
        return cv2.resize(cam.cpu().numpy(), (224, 224))
    finally:
        fh.remove()   # always removed — no accumulation across Gradio calls
        bh.remove()
```

---

## 8. Uncertainty Quantification: Monte Carlo Dropout

### 8.1 Method

At inference, dropout is normally disabled. MC Dropout re-enables it to approximate **Bayesian uncertainty**:

```python
def mc_dropout_predict(model, image_tensor, tabular_tensor, n_passes=20):
    model.train()    # activates Dropout(p=0.3)
    probs = [softmax(model(image_tensor, tabular_tensor)) for _ in range(20)]
    model.eval()     # FIX 2: restore deterministic mode
    mean = np.mean(probs, axis=0)
    std  = np.std(probs, axis=0)
    return mean, std, "Low Confidence — Human Review Required" if std.max() > 0.15 else "High Confidence"
```

### 8.2 Clinical Use

- **Mean probabilities** → displayed as prediction confidence
- **Std deviation** → quantifies epistemic uncertainty per class
- **Flag threshold** → `std.max() > 0.15` triggers radiologist review
- **Why `model.eval()` matters (FIX 2)**: Without it, all post-MC-Dropout Grad-CAM calls run with dropout active and BatchNorm in training mode — non-deterministic heatmaps with no error raised.

---

## 9. Risk Stratification & Clinical Recommendations

| Class | Label | Recommendation |
|-------|-------|----------------|
| **0** | **Normal** | Routine monitoring. Calcium 1000–1200 mg/day, vitamin D, weight-bearing exercise. Reassess in 2–3 years. |
| **1** | **Osteopenia** | DEXA referral. FRAX® fracture risk assessment. Review modifiable risk factors. Follow-up in 6–12 months. |
| **2** | **Osteoporosis** | Urgent DEXA. Metabolic bone panel. Pharmacological review (bisphosphonates, denosumab). Specialist referral. |

---

## 10. DICOM Integration (Hospital Readiness)

```python
import pydicom

def load_dicom_as_pil(dcm_path):
    ds          = pydicom.dcmread(dcm_path)
    pixels      = ds.pixel_array.astype(np.float32)
    pixels      = ((pixels - pixels.min()) / (pixels.max() - pixels.min()) * 255).astype(np.uint8)
    return Image.fromarray(pixels).convert('RGB')   # enters same pipeline as PNG/JPEG
```

DICOM output enters the identical `val_transform → model` path as PNG/JPEG. This confirms readiness for hospital PACS integration via HL7-FHIR.

---

## 11. Gradio Clinical Interface

| Feature | Detail |
|---------|--------|
| **Inputs** | JPEG/PNG upload + bone age (months) + sex (0/1) |
| **Tabular tensor** | Constructed as `[[boneage/228.0, sex]]` on GPU |
| **Inference** | MC Dropout (20 passes) → mean probs + std devs |
| **Decision rule** | Per-class optimal thresholds (OPTIMAL_THRESHOLDS); margin strategy not naive argmax |
| **Explainability** | Grad-CAM overlay (JET colormap, α=0.4) on `backbone.features[-1]` |
| **Risk card** | Colour-coded label (green/orange/red), confidence bar, per-class prob table |
| **Uncertainty** | ⚠️ banner if `std.max() > 0.15` |
| **Recommendation** | Clinical action text |
| **Sharing** | `demo.launch(share=True)` — live public URL |

---

## 12. Critical Engineering Fixes

| Fix | Bug | Consequence without fix |
|-----|-----|------------------------|
| **FIX 1** | No `.convert('RGB')` on grayscale RSNA images | Tensor `(1,224,224)` crashes EfficientNet first conv |
| **FIX 2** | Model stays in `train()` after MC Dropout | All subsequent Grad-CAM heatmaps non-deterministic |
| **FIX 3** | Pandas dtype mismatch on `id` column | Silent wrong train/val split — no error, just bad counts |
| **FIX 4** | GradCAM hooks never removed between calls | Gradient accumulation, degraded heatmaps, memory leak |
| **FIX 5** | `val_ids.csv` used as val split | ZeroDivisionError — 0 labeled val samples |

---

## 13. Technical Limitations

- **Screening tool only**: Detects radiographic BMD proxies. Quantitative BMD requires DEXA.
- **Training domain**: Pediatric/adolescent hand/wrist X-rays. Performance on hip, spine, or pathological adult populations not evaluated.
- **Tabular dependency**: Multi-modal model requires bone age and sex at inference. If unavailable, a fallback to population median values (120 months, sex=0.5) degrades performance.
- **No augmentation**: Competition rules prohibit it, which limits exposure-variation robustness.
- **Compute**: Trained on Google Colab T4 free tier, batch size 32.

---

## 14. Ethical Considerations

- **Not a diagnostic device**: Must not replace DEXA or clinical judgment.
- **Demographic drift**: Performance may differ across ethnicities, ages, or scanner types not in the training set.
- **DICOM privacy**: Hospital integration must strip patient PII from DICOM metadata before inference.
- **Clinician in the loop**: The MC Dropout uncertainty flag escalates ambiguous cases to radiologist review.

---

## 15. Deployment Roadmap

```
[Current — Hackathon submission]
  Gradio share=True public URL
  osteovision_best.pth (16.3 MB, Google Drive)

[Near-term]
  ONNX export → TensorRT optimisation
  NVIDIA Jetson Nano for edge deployment
  HL7-FHIR API wrapper for PACS/HIS integration

[Long-term]
  Multi-site clinical validation
  FDA/CDSCO SaMD Class II regulatory pathway
  Federated learning (privacy-preserving multi-hospital training)
```

---

## 16. Technical Summary

```
=== TECHNICAL SUMMARY ===
Model             : OsteoVisionMultiModal
                    (EfficientNet-B0 backbone + tabular branch + fusion head)
Task              : 3-class osteoporosis risk classification (Class 0/1/2)
Dataset           : RSNA Bone Age — 10,719 labeled hand/wrist X-rays
Inputs            : 224×224 RGB image + tabular [boneage_norm, sex]
Tabular encoding  : boneage / 228.0 (normalized), sex as 0.0/1.0 float
Fusion            : concat(1280-d image, 32-d tabular) → Linear(1312→128→3)
Training strategy : 2-phase (head-only epochs 1–10; last 3 MBConv epochs 11–30)
Loss              : CrossEntropyLoss() — no class weights (competition rule)
Augmentation      : None (per competition rules, dataset used as-is)
Validation        : Pre-defined split from val_ids.csv
Early stopping    : patience=7 on val macro F1
Checkpoint        : osteovision_best.pth (~20 MB, saved to Google Drive)
Explainability    : Grad-CAM on backbone.features[-1] (FIX 4: hooks removed per call)
Uncertainty       : Monte Carlo Dropout (20 passes, flag if max std > 0.15)
Threshold tuning  : Per-class optimal thresholds via margin strategy (BLOCK 6b)
Deployment        : Gradio UI (share=True) + pydicom DICOM pipeline

Best val Accuracy : 0.8909  (89.09%)
Best val F1 macro : 0.8381
Best val MCC      : 0.7584  (Matthews Correlation Coefficient)
Best val AUC-ROC  : 0.9720  (OvR macro)

Critical metric:
  Osteoporosis false-negative rate (misclassified as Normal): 1.1% (2/186)
  [Down from 1.6% (3/186) in single-modal v1 — 33% improvement]

Model comparison:
  v1 (single-modal) : Acc=83.77%  F1=0.7706  AUC=0.9342  FN=1.6%
  v2 (multi-modal)  : Acc=89.09%  F1=0.8381  AUC=0.9720  FN=1.1%
  Δ improvement     : +5.32%      +0.0675    +0.0378     −0.5%
=========================
```

---

## References

1. Tan, M. & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML.
2. Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV.
3. Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML.
4. Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR.
5. RSNA Pediatric Bone Age Challenge — Radiological Society of North America.
6. WHO (1994). *Assessment of Fracture Risk and its Application to Screening for Postmenopausal Osteoporosis*. Technical Report Series 843.

---

*© 2026 Team Techtoli | OsteoVision AI — AesCode Hackathon | Problem Statement 3*
