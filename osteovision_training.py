"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     OsteoVision AI — Training Pipeline                      ║
║                          Team Techtoli | AesCode Hackathon                  ║
║                                                                              ║
║  Problem Statement 3: AI-Based Osteoporosis Risk Screening                  ║
║  Architecture: EfficientNet-B0 + Grad-CAM Explainability                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Google Colab Script — Copy-paste into a Colab notebook or run as .py

Dependencies:
    pip install torch torchvision efficientnet_pytorch scikit-learn
    matplotlib numpy pillow tqdm
"""

# %% [markdown]
# # 🦴 OsteoVision AI — Osteoporosis Risk Screening
# **Team Techtoli** | AesCode Hackathon
#
# This notebook trains an EfficientNet-B0 model on X-ray radiographs to detect
# osteoporosis-related bone mineral density (BMD) loss, with Grad-CAM
# explainability for clinical transparency.

# %% ─── 1. IMPORTS & CONFIGURATION ───────────────────────────────────────────

import os
import copy
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ──────── Reproducibility ────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ──────── Hyperparameters ────────
CONFIG = {
    "model_name": "efficientnet_b0",
    "num_classes": 2,                    # Normal, Osteoporosis
    "class_names": ["Normal", "Osteoporosis"],
    "image_size": 224,
    "batch_size": 32,
    "num_epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "patience": 7,                       # Early stopping patience
    "train_split": 0.8,
    "num_workers": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"🔧 Device: {CONFIG['device']}")
print(f"🔧 PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")

# %% ─── 2. MOUNT GOOGLE DRIVE (Colab) ────────────────────────────────────────

# Uncomment the following lines when running in Google Colab:
# from google.colab import drive
# drive.mount('/content/drive')

# ──────── Dataset Path Configuration ────────
# Update this path to match your Google Drive folder structure.
# Expected directory layout:
#   DATA_DIR/
#   ├── Normal/          ← X-ray radiographs of healthy bone structure
#   └── Osteoporosis/    ← X-ray radiographs showing BMD loss
#
# If your dataset has train/test splits already:
#   DATA_DIR/
#   ├── train/
#   │   ├── Normal/
#   │   └── Osteoporosis/
#   └── val/ (or test/)
#       ├── Normal/
#       └── Osteoporosis/

DATA_DIR = "/content/drive/MyDrive/AesCode/Dataset"  # <-- UPDATE THIS PATH

# Output directory for checkpoints and visualizations
OUTPUT_DIR = "/content/drive/MyDrive/AesCode/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% ─── 3. DATA AUGMENTATION & PREPROCESSING ─────────────────────────────────
# Medical imaging augmentation strategy:
# - Horizontal flips: Valid for bilateral anatomical structures
# - Random rotations (±15°): Simulates slight positioning variations
# - Color jitter: Accounts for exposure/contrast differences across scanners
# - NO vertical flips (anatomically unrealistic for X-rays)

train_transforms = transforms.Compose([
    transforms.Resize((CONFIG["image_size"] + 32, CONFIG["image_size"] + 32)),
    transforms.RandomCrop(CONFIG["image_size"]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    ),
])

val_transforms = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# %% ─── 4. DATASET LOADING & CLASS IMBALANCE HANDLING ─────────────────────────

def create_data_loaders(data_dir, train_transform, val_transform, config):
    """
    Creates train/val DataLoaders with automatic class imbalance handling.

    Handles two directory structures:
    1. Flat: DATA_DIR/{Normal, Osteoporosis}/ → auto-split 80/20
    2. Pre-split: DATA_DIR/{train, val}/{Normal, Osteoporosis}/

    Returns:
        train_loader, val_loader, class_weights (Tensor)
    """
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")

    if os.path.isdir(train_path) and os.path.isdir(val_path):
        # ── Pre-split dataset ──
        print("📂 Detected pre-split dataset (train/val folders)")
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    else:
        # ── Flat dataset → Random split ──
        print("📂 Flat dataset detected — splitting 80/20")
        full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
        train_size = int(config["train_split"] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        # Apply validation transforms to val split
        val_dataset.dataset = copy.deepcopy(full_dataset)
        val_dataset.dataset.transform = val_transform

    # ── Compute class weights for Weighted Cross-Entropy ──
    if hasattr(train_dataset, 'targets'):
        targets = train_dataset.targets
    elif hasattr(train_dataset, 'dataset'):
        targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
    else:
        targets = [s[1] for s in train_dataset.samples]

    class_counts = Counter(targets)
    total = sum(class_counts.values())
    class_weights = torch.tensor(
        [total / (len(class_counts) * class_counts[i]) for i in range(config["num_classes"])],
        dtype=torch.float32
    )

    print(f"\n📊 Dataset Summary:")
    print(f"   Training samples:   {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    for cls_idx, cls_name in enumerate(config["class_names"]):
        count = class_counts.get(cls_idx, 0)
        print(f"   {cls_name}: {count} images (weight: {class_weights[cls_idx]:.3f})")

    # ── Weighted Random Sampler for balanced mini-batches ──
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, class_weights


# %% ─── 5. MODEL ARCHITECTURE ────────────────────────────────────────────────
# EfficientNet-B0: Compound scaling (depth × width × resolution) provides
# superior feature extraction for medical imaging vs. standard ResNets.
# Transfer learning from ImageNet captures low-level features (edges, textures)
# relevant to trabecular bone microarchitecture analysis.

def build_model(config):
    """
    Builds EfficientNet-B0 with a custom classification head.

    Architecture modifications:
    - Pretrained backbone (frozen initially, fine-tuned later)
    - Dropout (0.3) for regularization
    - 2-class output: Normal vs. Osteoporosis
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, config["num_classes"])
    )

    # ── Freeze backbone initially for stable transfer learning ──
    for param in model.features.parameters():
        param.requires_grad = False

    model = model.to(config["device"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🏗️  Model: EfficientNet-B0")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen backbone:      {total_params - trainable_params:,}")

    return model


def unfreeze_backbone(model, num_layers_to_unfreeze=3):
    """
    Progressively unfreeze the last N feature blocks for fine-tuning.
    This strategy prevents catastrophic forgetting of pretrained features
    while allowing adaptation to bone density patterns in radiographs.
    """
    features = list(model.features.children())
    for layer in features[-num_layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔓 Unfroze last {num_layers_to_unfreeze} feature blocks")
    print(f"   Trainable parameters: {trainable:,}")


# %% ─── 6. TRAINING ENGINE ───────────────────────────────────────────────────

class EarlyStopping:
    """
    Monitors validation loss and stops training when no improvement
    is observed for `patience` consecutive epochs.
    Saves the best model checkpoint automatically.
    """
    def __init__(self, patience=7, min_delta=1e-4, checkpoint_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            return True  # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # No improvement


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs a single training epoch with progress tracking."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)

    # AUC-ROC for binary classification
    all_probs = np.array(all_probs)
    try:
        metrics["auc_roc"] = roc_auc_score(all_labels, all_probs[:, 1])
    except ValueError:
        metrics["auc_roc"] = 0.0

    return epoch_loss, metrics


def compute_metrics(labels, preds):
    """
    Computes clinical screening metrics.
    High Recall is critical: missing an osteoporosis case (false negative)
    is far more dangerous than a false positive in screening.
    """
    return {
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "accuracy": np.mean(np.array(labels) == np.array(preds)),
    }


# %% ─── 7. GRAD-CAM IMPLEMENTATION ──────────────────────────────────────────
# Gradient-weighted Class Activation Mapping (Grad-CAM):
#   Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations from Deep Networks"
#
# For a target class c, the importance weight α_k^c for feature map A^k is:
#   α_k^c = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)
#
# The Grad-CAM heatmap is:
#   L_Grad-CAM^c = ReLU(Σ_k α_k^c · A^k)
#
# This highlights regions where the network detects bone density loss patterns,
# providing transparency essential for clinical adoption.

class GradCAM:
    """
    Grad-CAM implementation for EfficientNet-B0.
    Targets the last convolutional layer in the feature extractor for
    maximum spatial resolution of explanations.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        # Default: last feature block of EfficientNet
        if target_layer is None:
            self.target_layer = model.features[-1]
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks for gradient and activation capture
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook: captures forward-pass activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook: captures backward-pass gradients."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generates a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed image tensor [1, 3, H, W]
            target_class: Class index to explain (None = predicted class)

        Returns:
            heatmap: Normalized heatmap [H, W] in range [0, 1]
        """
        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients and backward pass for target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Compute importance weights: global average pooling of gradients
        # α_k^c = GAP(∂y^c / ∂A^k)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        # L^c = ReLU(Σ_k α_k^c · A^k)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze()
        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()


def generate_gradcam_overlay(model, image_path, transform, device, config, save_path=None):
    """
    Generates and displays a Grad-CAM heatmap overlaid on the original radiograph.

    This visualization highlights anatomical regions where the model detects
    indicators of reduced bone mineral density (BMD), such as:
    - Cortical thinning
    - Trabecular rarefaction
    - Changes in Singh index patterns

    Args:
        model: Trained EfficientNet-B0 model
        image_path: Path to input X-ray radiograph
        transform: Validation preprocessing pipeline
        device: Computation device (cuda/cpu)
        config: Configuration dictionary
        save_path: Optional path to save the visualization

    Returns:
        fig: Matplotlib figure with side-by-side comparison
    """
    # Load and preprocess
    original_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    # Initialize Grad-CAM
    grad_cam = GradCAM(model)

    # Generate heatmap
    heatmap = grad_cam.generate(input_tensor)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = probabilities.argmax().item()
        confidence = probabilities[predicted_class].item()

    # Resize heatmap to match original image
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize(
            original_image.size, resample=Image.BILINEAR
        )
    ) / 255.0

    # Create colored heatmap overlay
    colored_heatmap = cm.jet(heatmap_resized)[:, :, :3]  # Drop alpha channel
    original_array = np.array(original_image) / 255.0
    overlay = 0.6 * original_array + 0.4 * colored_heatmap

    # ── Visualization ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original radiograph
    axes[0].imshow(original_image)
    axes[0].set_title("Original Radiograph", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Grad-CAM heatmap (raw)
    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Activation Map", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Overlay: heatmap on radiograph
    axes[2].imshow(overlay)
    prediction_text = f"{config['class_names'][predicted_class]} ({confidence:.1%})"
    color = "red" if predicted_class == 1 else "green"
    axes[2].set_title(f"Overlay — {prediction_text}", fontsize=14,
                      fontweight="bold", color=color)
    axes[2].axis("off")

    plt.suptitle("🦴 OsteoVision AI — Grad-CAM Explainability",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved Grad-CAM visualization: {save_path}")

    return fig


# %% ─── 8. TRAINING ORCHESTRATOR ─────────────────────────────────────────────

def train_model(config):
    """
    Full training pipeline orchestrator.

    Strategy:
    1. Phase 1 (Epochs 1–10): Train only the classification head (backbone frozen)
    2. Phase 2 (Epochs 11+): Unfreeze last 3 backbone blocks for fine-tuning
    3. CosineAnnealingLR scheduler for smooth learning rate decay
    4. Early stopping on validation loss
    """
    print("=" * 70)
    print("🦴  OsteoVision AI — Training Pipeline")
    print("=" * 70)

    # ── Data ──
    train_loader, val_loader, class_weights = create_data_loaders(
        DATA_DIR, train_transforms, val_transforms, config
    )

    # ── Model ──
    model = build_model(config)

    # ── Weighted Cross-Entropy Loss ──
    # Addresses class imbalance: penalizes misclassification of minority class more
    class_weights = class_weights.to(config["device"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"\n⚖️  Weighted CE Loss — weights: {class_weights.cpu().tolist()}")

    # ── Optimizer & Scheduler ──
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=1e-6
    )

    # ── Early Stopping ──
    checkpoint_path = os.path.join(OUTPUT_DIR, "osteovision_best.pth")
    early_stopping = EarlyStopping(
        patience=config["patience"],
        checkpoint_path=checkpoint_path
    )

    # ── Training History ──
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_precision": [], "val_recall": [], "val_f1": [], "val_auc": [],
        "lr": []
    }

    print(f"\n{'='*70}")
    print(f"{'Epoch':<8}{'Train Loss':<12}{'Val Loss':<12}{'Precision':<11}"
          f"{'Recall':<9}{'F1':<9}{'AUC':<9}{'LR':<12}")
    print(f"{'='*70}")

    for epoch in range(1, config["num_epochs"] + 1):
        # ── Phase 2: Unfreeze backbone at epoch 10 ──
        if epoch == 10:
            unfreeze_backbone(model, num_layers_to_unfreeze=3)
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config["learning_rate"] * 0.1,  # Lower LR for fine-tuning
                weight_decay=config["weight_decay"]
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["num_epochs"] - epoch, eta_min=1e-6
            )

        # ── Train & Validate ──
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, config["device"]
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, config["device"]
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # ── Log metrics ──
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics.get("auc_roc", 0))
        history["lr"].append(current_lr)

        # ── Print epoch summary ──
        improved = early_stopping(val_loss, model)
        marker = "✅" if improved else "  "
        print(f"{marker} {epoch:<5}{train_loss:<12.4f}{val_loss:<12.4f}"
              f"{val_metrics['precision']:<11.4f}{val_metrics['recall']:<9.4f}"
              f"{val_metrics['f1']:<9.4f}{val_metrics.get('auc_roc', 0):<9.4f}"
              f"{current_lr:<12.6f}")

        if early_stopping.early_stop:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
            break

    # ── Load best model ──
    model.load_state_dict(torch.load(checkpoint_path, map_location=config["device"]))
    print(f"\n✅ Best model loaded from: {checkpoint_path}")

    return model, history, val_loader


# %% ─── 9. VISUALIZATION UTILITIES ───────────────────────────────────────────

def plot_training_history(history, save_path=None):
    """Generates comprehensive training curves for performance analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Loss curves
    axes[0, 0].plot(history["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
    axes[0, 0].plot(history["val_loss"], label="Val Loss", color="#F44336", linewidth=2)
    axes[0, 0].set_title("Loss Curves", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(history["train_acc"], label="Train Acc", color="#2196F3", linewidth=2)
    axes[0, 1].plot(history["val_acc"], label="Val Acc", color="#F44336", linewidth=2)
    axes[0, 1].set_title("Accuracy Curves", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Clinical metrics (Precision, Recall, F1)
    axes[1, 0].plot(history["val_precision"], label="Precision", color="#4CAF50", linewidth=2)
    axes[1, 0].plot(history["val_recall"], label="Recall ⭐", color="#FF9800",
                    linewidth=2.5, linestyle="--")
    axes[1, 0].plot(history["val_f1"], label="F1-Score", color="#9C27B0", linewidth=2)
    axes[1, 0].set_title("Clinical Screening Metrics", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[1, 1].plot(history["lr"], color="#607D8B", linewidth=2)
    axes[1, 1].set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("🦴 OsteoVision AI — Training Dashboard",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved training curves: {save_path}")

    plt.show()
    return fig


def plot_confusion_matrix(model, loader, device, class_names, save_path=None):
    """Generates a confusion matrix for final model evaluation."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm_matrix = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_matrix, cmap="Blues")

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_matrix[i, j] > cm_matrix.max() / 2 else "black"
            ax.text(j, i, str(cm_matrix[i, j]), ha="center", va="center",
                    fontsize=20, fontweight="bold", color=color)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=13, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar(im)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    # Print classification report
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return fig


# %% ─── 10. MAIN EXECUTION ──────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Train the model ──
    model, history, val_loader = train_model(CONFIG)

    # ── Plot training history ──
    plot_training_history(
        history,
        save_path=os.path.join(OUTPUT_DIR, "training_curves.png")
    )

    # ── Confusion matrix ──
    plot_confusion_matrix(
        model, val_loader, CONFIG["device"], CONFIG["class_names"],
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )

    # ── Grad-CAM Demos ──
    # Uncomment and update paths to generate Grad-CAM visualizations:
    # sample_images = [
    #     "/path/to/sample_xray_1.png",
    #     "/path/to/sample_xray_2.png",
    # ]
    # for i, img_path in enumerate(sample_images):
    #     generate_gradcam_overlay(
    #         model, img_path, val_transforms, CONFIG["device"], CONFIG,
    #         save_path=os.path.join(OUTPUT_DIR, f"gradcam_demo_{i+1}.png")
    #     )

    # ── Alternatively: Grab samples from validation set ──
    val_dataset = val_loader.dataset
    if hasattr(val_dataset, 'samples'):
        sample_paths = [val_dataset.samples[i][0] for i in range(min(5, len(val_dataset)))]
    elif hasattr(val_dataset, 'dataset') and hasattr(val_dataset.dataset, 'samples'):
        sample_paths = [val_dataset.dataset.samples[val_dataset.indices[i]][0]
                        for i in range(min(5, len(val_dataset)))]
    else:
        sample_paths = []

    for i, img_path in enumerate(sample_paths):
        # Re-create model for clean Grad-CAM hooks each time
        demo_model = build_model(CONFIG)
        demo_model.load_state_dict(
            torch.load(os.path.join(OUTPUT_DIR, "osteovision_best.pth"),
                       map_location=CONFIG["device"])
        )
        generate_gradcam_overlay(
            demo_model, img_path, val_transforms, CONFIG["device"], CONFIG,
            save_path=os.path.join(OUTPUT_DIR, f"gradcam_sample_{i+1}.png")
        )

    print("\n" + "=" * 70)
    print("🎉 OsteoVision AI — Training Complete!")
    print(f"   Best model saved: {os.path.join(OUTPUT_DIR, 'osteovision_best.pth')}")
    print(f"   Visualizations:   {OUTPUT_DIR}")
    print("=" * 70)
