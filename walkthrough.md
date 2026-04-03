# OsteoVision CBAM Experimental Notebook — Walkthrough

## What Was Created

**File:** [OsteoVision_CBAM_Experimental.ipynb](file:///C:/Users/divya/OneDrive/Documents/Aescode/OsteoVision_CBAM_Experimental.ipynb)  
**Total cells:** 39 | **Baseline:** untouched

---

## Task Sheet — All Items Checked

```
=== CBAM TASK SHEET — COMPLETED ===
[x] C1.  CBAMChannelAttention class implemented correctly
[x] C2.  CBAMSpatialAttention class implemented correctly
[x] C3.  CBAM class combines channel + spatial attention sequentially
[x] C4.  OsteoVisionCBAM model class created (separate from OsteoVisionMultiModal)
[x] C5.  CBAM inserted after model.features[-3], [-2], [-1] (indices 6,7,8)
[x] C6.  forward() passes image through backbone blocks with CBAM at 3 points
[x] C7.  forward() still accepts (image, tabular) — multimodal fusion preserved
[x] C8.  Phase 1: backbone frozen, CBAM+tabular+fusion train
[x] C9.  Phase 2: unfreeze features[6,7,8] + CBAM stays trainable
[x] C10. Training loop unchanged — same (imgs, tabs, lbls) unpacking
[x] C11. All 4 fixes: RGB, eval(), astype(int), hook finally
[x] C12. GradCAM targets cbam_8.spatial_attention.conv
[x] C13. BLOCK 1 dataset dimensions identical to baseline
[x] C14. Transforms identical (Resize + ToTensor only, no Normalize)
[x] C15. No augmentation, no WeightedRandomSampler, no weighted CE loss
[x] C16. Comparison table cell added as BLOCK 13
[x] VERIFY: model(imgs, tabs) in all call sites
[x] VERIFY: No forbidden patterns
[x] VERIFY: Shape check passes on CPU dummy tensors
=====================================
```

---

## Notebook Structure

| Block | Content |
|-------|---------|
| Title + Task Sheet | CBAM task sheet printed at start |
| Install | pip dependencies |
| Imports + CONFIG | All imports; `checkpoint = "osteovision_cbam_best.pth"` |
| **BLOCK 1** | Dataset dimensions (identical split to baseline) |
| **BLOCK 2** | BoneAgeDataset, transforms (Resize+ToTensor only), DataLoaders |
| **BLOCK 3** | `CBAMChannelAttention`, `CBAMSpatialAttention`, `CBAM`, `OsteoVisionCBAM` |
| **BLOCK 3b** | Phase 1 freezing (backbone frozen, CBAM+tabular+fusion train) |
| **BLOCK 4** | CrossEntropyLoss (no weights), AdamW optimizers, EarlyStopping |
| **BLOCK 5** | 2-phase training loop with Phase 2 unlock at epoch 11 |
| **BLOCK 6** | Full evaluation: Accuracy, Macro F1, AUC-ROC, MCC, confusion matrix, PR curves |
| **BLOCK 6b** | Per-class threshold tuning → `OPTIMAL_THRESHOLDS` |
| **BLOCK 7** | Monte Carlo Dropout (20 passes) |
| **BLOCK 8** | GradCAM targeting `cbam_8.spatial_attention.conv` |
| **BLOCK 9** | DICOM support (identical to baseline) |
| **BLOCK 10** | RISK_MAP (identical to baseline) |
| **BLOCK 11** | Gradio clinical interface |
| **BLOCK 12** | Technical Report Summary |
| **BLOCK 13** | **CBAM vs Baseline comparison table** |
| **BLOCK 14** | Verification checks (shape, CBAM channels, forbidden patterns) |
| Completed Sheet | Final task sheet printout |

---

## Verification Results

| Check | Result |
|-------|--------|
| 22 required patterns | ✅ All OK |
| 5 forbidden patterns in actual code | ✅ All ABSENT |
| Forbidden in print strings only | ✅ Safe (CHECK 3 message) |
| Notebook cell count | ✅ 39 cells |
| Baseline untouched | ✅ Not modified |

**Forbidden patterns verified absent from actual code:**
- `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `WeightedRandomSampler`, `transforms.Normalize`

---

## Key Architectural Changes vs Baseline

```
Baseline:                          CBAM Experimental:
OsteoVisionMultiModal              OsteoVisionCBAM
  backbone (EfficientNet-B0)         features[0..5] → unchanged
  → 1280-d features                  features[6] → cbam_6 (192ch)
                                     features[7] → cbam_7 (320ch)
                                     features[8] → cbam_8 (1280ch)
                                     → avgpool → 1280-d features

GradCAM: backbone.features[-1]     GradCAM: cbam_8.spatial_attention.conv
Checkpoint: osteovision_best.pth   Checkpoint: osteovision_cbam_best.pth
```

## Baseline to Beat
| Metric | Baseline |
|--------|----------|
| Macro F1 | 0.8381 |
| AUC-ROC | 0.9720 |
| MCC | 0.7584 |
| Accuracy | 0.8909 |
