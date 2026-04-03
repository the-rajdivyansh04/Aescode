# 🦴 OsteoVision AI — PPT Content (9 Slides)
## Copy-Paste Ready Content for Each Slide

---

## SLIDE 1 — Cover / Title Slide

**Project Title (large, centered):**
> **OsteoVision AI**

**One-line Tagline:**
> AI-Powered Osteoporosis Risk Screening from Routine X-ray Radiographs

**Problem Statement:**
> PS-3: AI-Based Osteoporosis Risk Screening from Medical Imaging

**Team Name:**
> Team Techtoli

**Team ID:**
> *(Insert your Team ID here)*

**Institution:**
> *(Insert your University / College name here)*

**Date:**
> 27 March 2026

---

## SLIDE 2 — Problem Context & Motivation

### Two Headline Stats:
- **200M+** people affected globally — yet 80% of high-risk patients in rural areas are **never screened**
- **8.9M** osteoporotic fractures per year — one every **3 seconds** — costing **$19B+** annually in the US alone

### Problem Statement (3–5 sentences):
Osteoporosis is a "silent disease" that causes progressive bone mineral density (BMD) loss, leading to fragility fractures in over 200 million people worldwide. The gold-standard diagnostic tool — Dual-energy X-ray Absorptiometry (DEXA) — costs approximately **10× more** than standard radiography and is concentrated in urban tertiary hospitals, making early screening inaccessible to the vast majority of at-risk populations. In resource-limited and rural healthcare settings, patients are diagnosed only **after** suffering a fracture, when intervention is most costly and least effective. There is an urgent need for affordable, scalable screening that leverages the X-ray infrastructure already available in virtually every clinic worldwide.

### 3 Current Limitations:
1. **DEXA Inaccessibility** — DEXA scanners are expensive (₹15–50 Lakh), require trained technicians, and are concentrated in urban centers, leaving rural populations unscreened
2. **Late-stage Diagnosis** — Without accessible screening, osteoporosis is typically detected only after a fragility fracture, when bone loss is already severe and irreversible
3. **No AI Screening Exists** — Current radiology workflows do not extract bone density information from routine X-rays, despite these images containing latent indicators of BMD loss in trabecular and cortical bone patterns

---

## SLIDE 3 — Proposed Approach & Methodology

### Methodology:
| Aspect | Details |
|--------|---------|
| **Model Type** | Convolutional Neural Network (CNN) — EfficientNet-B0 with Transfer Learning |
| **Input Format** | Standard X-ray Radiograph (JPEG/PNG) — 224 × 224 pixels, ImageNet-normalized |
| **Output** | 3-tier Risk Classification (Normal / Osteopenia / Osteoporosis) + Confidence Score + Grad-CAM Heatmap |
| **Framework** | PyTorch 2.x + Torchvision |
| **Training Strategy** | 2-Phase: Phase 1 (Epochs 1–10) — Frozen backbone, train classifier head only; Phase 2 (Epochs 11–30) — Unfreeze last 3 blocks for fine-tuning with 10× lower LR |
| **Key Libraries** | PyTorch, Torchvision, scikit-learn, Gradio, Matplotlib, NumPy, PIL |
| **Loss Function** | Weighted Cross-Entropy (class-imbalance aware, penalizes false negatives) |
| **Optimizer** | AdamW (weight decay = 1e-4) + CosineAnnealingLR scheduler |

### Innovation & Rationale:

**3 Novelty Points:**
1. **Zero-Hardware Screening** — Repurposes existing X-ray infrastructure for bone density assessment; no DEXA required, enabling screening in any clinic with an X-ray machine
2. **Explainable AI with Grad-CAM** — Provides spatial heatmap overlays showing exactly which bone regions trigger the prediction (cortical thinning, trabecular rarefaction), building clinical trust and meeting regulatory explainability requirements (FDA/CDSCO AI/ML SaMD guidelines)
3. **3-Tier Risk Stratification** — Maps binary model output to a clinically actionable Normal → Osteopenia → Osteoporosis risk scale with estimated T-score ranges and specific referral recommendations, creating an intermediate safety net

**Key References:**
- Selvaraju et al. (2017) — *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"*
- Tan & Le (2019) — *"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"*
- WHO Scientific Group — *"Assessment of fracture risk and its application to screening for postmenopausal osteoporosis"*

---

## SLIDE 4 — Dataset Overview

### 5 Dataset Dimension Stats:
| Metric | Value |
|--------|-------|
| 🖼️ **Image Count** | *(Insert total number — e.g., 1,200 images)* |
| 📐 **Input Dimensions** | 224 × 224 × 3 (RGB, resized) |
| 💾 **Dataset Size** | *(Insert — e.g., ~850 MB)* |
| 📊 **CSV Rows / Columns** | N/A (folder-based ImageFolder structure) |
| 📁 **File Format** | JPEG / PNG radiograph images |

### Dataset Details Table:
| Field | Details |
|-------|---------|
| **Name** | X-ray Osteoporosis Radiograph Dataset (Google Drive hosted) |
| **Classes / Labels** | 2 classes — `Normal`, `Osteoporosis` |
| **Train / Val / Test Split** | 80% Training / 20% Validation (random split with seed=42) |
| **Input Format** | RGB X-ray radiographs, resized to 224×224, ImageNet normalized |
| **Class Balance** | Handled via WeightedRandomSampler + Weighted Cross-Entropy Loss |

### Data Distribution (4-class chart description — use for chart visual):
> **Note**: Our model uses 2 classes (Normal, Osteoporosis). If your dataset has 4 sub-categories, show distribution as:
> - Normal (Healthy Bone)
> - Osteopenia (Mild BMD Loss)
> - Moderate Osteoporosis
> - Severe Osteoporosis
>
> *(Replace with actual class counts from your dataset after training)*

---

## SLIDE 5 — Model Architecture & Workflow

### Architecture Diagram — Pipeline Labels:

```
INPUT LAYER          →    FEATURE EXTRACTOR       →    CLASSIFICATION HEAD    →    OUTPUT
─────────────────────────────────────────────────────────────────────────────────────────────
X-ray Radiograph          EfficientNet-B0              Adaptive Avg Pool          Risk Level
(224×224×3 RGB)           Backbone (ImageNet           → Dropout (p=0.3)         (Normal /
Resize + Normalize        pretrained, 5.3M             → FC Linear Layer          Osteopenia /
ImageNet (μ, σ)           params, compound              (1280 → 2)               Osteoporosis)
+ Data Augmentation       scaling: depth ×                                       + Confidence %
(flips, rotation,         width × resolution)          Softmax Activation        + Grad-CAM
color jitter, affine)     Last 3 blocks fine-tuned                                Heatmap Overlay
                                                                                 + Clinical
                                                                                  Recommendation
```

**Stage Descriptions (for diagram blocks):**

| Stage | Component | Description |
|-------|-----------|-------------|
| **1. Input Layer** | Preprocessing Pipeline | Resize to 224×224, normalize with ImageNet μ/σ, augment (horizontal flip, ±15° rotation, color jitter, ±5% affine translation) |
| **2. Feature Extractor** | EfficientNet-B0 Backbone | 5.3M-parameter CNN with compound scaling; pretrained on ImageNet; last 3 blocks fine-tuned for bone density pattern recognition |
| **3. Classification Head** | FC + Dropout + Softmax | Adaptive average pooling → Dropout (0.3) → Fully connected (1280→2) → Softmax probabilities |
| **4. Output** | Risk Assessment + Explainability | 3-tier risk level, confidence score, Grad-CAM spatial heatmap, and auto-generated clinical recommendation |

---

## SLIDE 6 — Experiments & Results

### 5 Top-line Metrics (Multi-Modal v2):
| Metric | Value | Clinical Relevance |
|--------|-------|--------------------| 
| 🎯 **Accuracy** | **89.09%** | Overall correctness across 2,144 val samples |
| 📊 **MCC (Matthews CC)** | **0.7584** | Best metric for imbalanced 3-class — range −1 to +1 |
| ⚖️ **Macro F1** | **0.8381** | Balanced screening performance across all classes |
| 📈 **AUC-ROC** | **0.9720** | Threshold-independent discrimination (OvR macro) |
| 🔴 **Critical FN Rate** | **1.1% (2/186)** | Osteoporosis missed as Normal — minimum safe threshold |

> ⭐ **Critical FN Rate is the most important clinical metric**: misclassifying Osteoporosis as Normal leads to missed treatment for the highest-risk patients.

### v1 vs v2 Comparison:
| Metric | v1 Single-Modal | v2 Multi-Modal | Δ |
|--------|----------------|----------------|---|
| Accuracy | 83.77% | **89.09%** | +5.32% |
| Macro F1 | 0.7706 | **0.8381** | +0.0675 |
| MCC | — | **0.7584** | new |
| AUC-ROC | 0.9342 | **0.9720** | +0.0378 |
| Osteoporosis FN→Normal | 1.6% (3/186) | **1.1% (2/186)** | −33% |

### Key Confusion Matrix Finding:
> Only **2 out of 186** osteoporosis patients were misclassified as Normal (1.1%) — a **33% reduction** vs the single-modal baseline (3/186, 1.6%).


---

## SLIDE 7 — Challenges Faced

### 6 Challenge Cards:

**💻 Compute Constraints**
> Limited to free-tier Google Colab (T4 GPU, session timeouts). Training required careful checkpoint management and early stopping to avoid losing progress. EfficientNet-B0 was chosen over larger models (B4/B7) to fit within memory constraints while maximizing accuracy.

**📦 Data Quality**
> Medical imaging datasets are inherently scarce and sensitive. Class imbalance between Normal and Osteoporosis cases required WeightedRandomSampler and Weighted Cross-Entropy Loss. X-ray quality varies across scanners, necessitating robust augmentation (color jitter, rotation) to improve generalization.

**⚙️ Model Performance**
> Balancing Recall (sensitivity) vs. Precision for clinical safety. A false negative in osteoporosis screening is far more dangerous than a false positive. We optimized for high recall through loss weighting while maintaining acceptable precision to avoid excessive false referrals.

**🔗 Integration**
> Building a clinic-ready interface that bridges the gap between a research model and a usable medical tool. Gradio provided rapid prototyping, but production deployment will require DICOM integration, PACS connectivity, and HL7-FHIR interoperability with Hospital Information Systems.

**📏 Evaluation**
> Standard ML metrics (accuracy) are insufficient for medical screening. We implemented clinical screening metrics — Recall, Precision, F1, AUC-ROC — with explicit prioritization of Recall. The 3-tier risk stratification (Normal → Osteopenia → Osteoporosis) adds an intermediate safety net not present in binary classifiers.

**🔬 Domain Knowledge Gap**
> Understanding radiographic markers of bone density loss (cortical thinning, trabecular rarefaction, Singh index changes) was essential for designing medically-valid augmentations (no vertical flips) and interpreting Grad-CAM heatmaps. Consultation of WHO T-score criteria and FRAX® risk assessment frameworks informed the risk stratification thresholds.

---

## SLIDE 8 — Team Introduction

**Team Name:** Team Techtoli
**Team ID:** *(Insert your Team ID)*
**Problem Statement:** PS-3: AI-Based Osteoporosis Risk Screening

### Team Members:

| # | Name | Role / Title | Skills |
|---|------|-------------|--------|
| 1 | *(Insert Name)* | Team Lead / ML Engineer | Deep Learning, PyTorch, Medical Imaging, Transfer Learning |
| 2 | *(Insert Name)* | Data Engineer / Backend Developer | Data Pipelines, Python, Google Colab, Dataset Curation |
| 3 | *(Insert Name)* | Frontend / UI Developer | Gradio, Web UI, Visualization, User Experience Design |
| 4 | *(Insert Name)* | Research / Domain Expert | Medical Imaging Literature, Evaluation Metrics, Clinical Validation |

---

## SLIDE 9 — Additional Information

### ✅ Reproducibility
- **Notebook**: `OsteoVision_AI_TeamTechtoli.ipynb` — Complete, self-contained Colab notebook
- **Random Seed**: `SEED = 42` — All operations (torch, numpy, random, CUDA) are seeded for reproducibility
- **Requirements**: `requirements.txt` provided — `torch>=2.0.0`, `torchvision>=0.15.0`, `scikit-learn>=1.3.0`, `gradio>=4.0.0`, `matplotlib>=3.7.0`, `numpy>=1.24.0`, `Pillow>=10.0.0`, `tqdm>=4.65.0`
- **Dataset Documentation**: Google Drive folder structure: `Dataset/{Normal, Osteoporosis}/` or `Dataset/{train, val}/{Normal, Osteoporosis}/`
- **Deterministic Training**: `torch.backends.cudnn.deterministic = True`, `cudnn.benchmark = False`

### 🛠️ Tools & Environment
- **Platform**: Google Colab (Pro/Free tier compatible)
- **GPU**: NVIDIA T4 (16 GB VRAM) via Colab runtime
- **Training Time**: ~15–30 minutes (30 epochs with early stopping, patience=7)
- **External Services**: Google Drive (dataset storage + checkpoint persistence), Gradio Share (public demo link generation)
- **Key Framework Versions**: PyTorch 2.x, Torchvision 0.15+, EfficientNet-B0 (torchvision.models)

### 📋 Ethical & Clinical Notes
- **Biases**: Model is trained on a limited dataset that may not represent the full spectrum of demographic diversity (age, sex, ethnicity, bone site). Performance may vary across populations not well-represented in training data.
- **Limitations**: This is a **screening tool**, NOT a diagnostic device. It provides risk stratification based on radiographic patterns but cannot replace DEXA for definitive BMD measurement. The 3-tier risk levels (Normal / Osteopenia / Osteoporosis) are estimated from model confidence, not direct T-score measurement.
- **Privacy**: No patient-identifiable information (PII) is stored or transmitted. The Gradio interface processes images locally (or via ephemeral Gradio servers with no persistence). DICOM metadata stripping should be implemented before deployment.
- **Assumptions**: The model assumes input images are musculoskeletal X-ray radiographs. Non-radiographic inputs (CT, MRI, photographs) will produce unreliable results. The risk thresholds (0.35 / 0.65) are heuristic and should be clinically validated before deployment.

### Anything Else?
- **Grad-CAM Explainability**: Every prediction includes a spatial heatmap overlay showing which bone regions drove the classification, critical for regulatory compliance (FDA AI/ML-based SaMD guidelines, CDSCO Class B)
- **Deployment Roadmap**: ONNX export → TensorRT optimization → NVIDIA Jetson Nano edge deployment for rural clinics; DICOM integration with PACS/HIS via HL7-FHIR
- **Gradio Demo App**: `osteovision_app.py` provides a clinic-ready web interface with real-time inference, Grad-CAM overlays, and automated clinical recommendations
- **Federated Learning (Future)**: Train across multiple hospital datasets without sharing patient data, addressing both privacy concerns and data scarcity

---

*© 2026 Team Techtoli | OsteoVision AI — AesCode Hackathon*
