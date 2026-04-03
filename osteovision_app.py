"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  OsteoVision AI — Clinic-Ready Gradio Prototype             ║
║                          Team Techtoli | AesCode Hackathon                  ║
║                                                                              ║
║  AI-Based Osteoporosis Risk Screening with Explainable Grad-CAM             ║
║  3-class output: Normal (0) / Osteopenia (1) / Osteoporosis (2)             ║
║  Uncertainty: Monte Carlo Dropout (20 passes)                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Launch:
    pip install gradio torch torchvision pillow numpy matplotlib opencv-python
    python osteovision_app.py
"""

import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import cv2

import gradio as gr


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "model_path" : "osteovision_best.pth",
    "num_classes": 3,
    "class_names": ["Normal", "Osteopenia", "Osteoporosis"],
    "image_size" : 224,
    "device"     : "cuda" if torch.cuda.is_available() else "cpu",
}

# ── Preprocessing — matches training pipeline exactly ──
# NO normalization, NO augmentation per competition rules
inference_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
])

# ── Risk map: class index → clinical label + recommendation ──
RISK_MAP = {
    0: {
        "label"          : "Normal",
        "color"          : "#4CAF50",
        "recommendation" : (
            "Routine monitoring, standard preventive care. "
            "Maintain adequate calcium (1000–1200 mg/day) and vitamin D intake. "
            "Regular weight-bearing exercise recommended. "
            "Reassess in 2–3 years or if risk factors change."
        ),
    },
    1: {
        "label"          : "Osteopenia",
        "color"          : "#FF9800",
        "recommendation" : (
            "DEXA referral recommended, FRAX assessment advised. "
            "Review modifiable risk factors (smoking, alcohol, sedentary lifestyle). "
            "Consider calcium/vitamin D supplementation. Follow-up in 6–12 months."
        ),
    },
    2: {
        "label"          : "Osteoporosis",
        "color"          : "#F44336",
        "recommendation" : (
            "Urgent clinical action required. Immediate DEXA scan referral. "
            "Comprehensive metabolic bone panel. Pharmacological review "
            "(bisphosphonates, denosumab). Fall risk assessment. "
            "Specialist referral (Endocrinology / Rheumatology)."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(config):
    """Load EfficientNet-B0 with 3-class head. Falls back to ImageNet demo if no checkpoint."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, config["num_classes"]),   # 3-class output
    )

    if os.path.exists(config["model_path"]):
        state_dict = torch.load(config["model_path"], map_location=config["device"])
        model.load_state_dict(state_dict)
        print(f"✅ Loaded trained model: {config['model_path']}")
    else:
        # Demo mode: load pretrained backbone weights
        pretrained = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Carry over all feature layers; keep our new 3-class head (randomly initialised)
        model.features.load_state_dict(pretrained.features.state_dict())
        print(f"⚠️  Checkpoint not found: {config['model_path']}")
        print(f"   Running in DEMO MODE (ImageNet backbone, untrained classifier head).")

    model.to(config["device"])
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MONTE CARLO DROPOUT UNCERTAINTY
# ═══════════════════════════════════════════════════════════════════════════════

def mc_dropout_predict(model, image_tensor, n_passes=20):
    """
    Run N stochastic forward passes with dropout active.

    Returns:
        mean_probs  (np.ndarray, shape 3)  — mean softmax across passes
        std_probs   (np.ndarray, shape 3)  — std deviation per class
        flag        (str)                  — confidence flag
    """
    model.train()   # activate dropout
    probs_list = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(image_tensor)
            probs_list.append(torch.softmax(out, dim=1).cpu().numpy()[0])
    # FIX 2: restore eval() so downstream Grad-CAM/inference are deterministic
    model.eval()

    probs_arr  = np.array(probs_list)         # (n_passes, 3)
    mean_probs = probs_arr.mean(axis=0)       # (3,)
    std_probs  = probs_arr.std(axis=0)        # (3,)
    low_conf   = bool(std_probs.max() > 0.15)
    flag = "Low Confidence — Human Review Required" if low_conf else "High Confidence"
    return mean_probs, std_probs, flag


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GRAD-CAM ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Grad-CAM targeting model.features[-1] (last MBConv block).
    Hooks are registered and removed inside compute_cam() to prevent
    gradient accumulation across Gradio calls (FIX 4).
    """

    def __init__(self, model):
        self.model        = model
        self.target_layer = model.features[-1]
        self._grads       = None
        self._acts        = None

    def _save_activation(self, module, inp, out):
        self._acts = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def compute_cam(self, image_tensor, target_class=None):
        """Returns a 224×224 numpy heatmap in [0, 1]."""
        # FIX 4: register hooks fresh each call, always remove in finally
        fh = self.target_layer.register_forward_hook(self._save_activation)
        bh = self.target_layer.register_full_backward_hook(self._save_gradient)
        try:
            self.model.eval()
            output = self.model(image_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            self.model.zero_grad()
            output[0, target_class].backward()

            weights = self._grads.mean(dim=[2, 3], keepdim=True)
            cam     = torch.relu((weights * self._acts).sum(dim=1, keepdim=True)).squeeze()
            if cam.max() != cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = torch.zeros_like(cam)
            heatmap = cam.cpu().numpy()
            heatmap = cv2.resize(heatmap, (224, 224))
            return heatmap
        finally:
            fh.remove()
            bh.remove()


def overlay_heatmap(pil_image, heatmap, alpha=0.4):
    """Overlay JET colourmap heatmap on the original image."""
    orig    = np.array(pil_image.convert("RGB").resize((224, 224))) / 255.0
    colored = cm.jet(heatmap)[:, :, :3]
    overlay = np.clip((1 - alpha) * orig + alpha * colored, 0, 1)
    return (overlay * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. INFERENCE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

model       = load_model(CONFIG)
gradcam_eng = GradCAM(model)


def predict(input_image):
    """
    Full inference pipeline:
        preprocess → MC Dropout → Grad-CAM → risk card
    """
    if input_image is None:
        return None, "<p style='color:#aaa'>Please upload an X-ray radiograph.</p>", ""

    if isinstance(input_image, np.ndarray):
        pil_image = Image.fromarray(input_image).convert("RGB")
    else:
        pil_image = input_image.convert("RGB")

    tensor = inference_transform(pil_image).unsqueeze(0).to(CONFIG["device"])

    # ── 1. MC Dropout uncertainty ──
    mean_probs, std_probs, conf_flag = mc_dropout_predict(model, tensor, n_passes=20)
    pred_class = int(np.argmax(mean_probs))
    confidence = float(mean_probs[pred_class]) * 100

    # ── 2. Grad-CAM on predicted class ──
    heatmap     = gradcam_eng.compute_cam(tensor, pred_class)
    overlay_arr = overlay_heatmap(pil_image, heatmap, alpha=0.4)
    overlay_img = Image.fromarray(overlay_arr)

    # ── 3. Risk info ──
    risk  = RISK_MAP[pred_class]
    color = risk["color"]
    low_c = "Low" in conf_flag

    flag_html = (
        '<p style="color:#FF9800;font-weight:bold;margin:6px 0 0 0">'
        "&#9888; " + conf_flag + "</p>"
    ) if low_c else ""

    bar_w = min(int(confidence), 100)

    # Per-class probability rows
    rows = "".join(
        '<tr style="border-bottom:1px solid rgba(255,255,255,0.05)">'
        '<td style="padding:6px 10px">' + CONFIG["class_names"][i] + '</td>'
        '<td style="padding:6px 10px;font-weight:bold">' + f"{mean_probs[i] * 100:.1f}%" + '</td>'
        '<td style="padding:6px 10px;color:#888">&plusmn;' + f"{std_probs[i]:.3f}" + '</td>'
        "</tr>"
        for i in range(3)
    )

    risk_html = (
        '<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);'
        'border-radius:16px;padding:28px;color:white;'
        'font-family:\'Segoe UI\',system-ui,sans-serif;'
        'border:1px solid rgba(255,255,255,0.1);'
        'box-shadow:0 8px 32px rgba(0,0,0,0.3)">'

        '<h2 style="margin:0 0 4px 0;font-size:22px;color:' + color + '">'
        + risk["label"] + "</h2>"
        + flag_html

        + '<p style="color:#888;font-size:13px;margin:8px 0 16px 0">'
        'MC Dropout (20 passes) · std threshold: 0.15</p>'

        '<div style="margin-bottom:16px">'
        '<div style="display:flex;justify-content:space-between;margin-bottom:6px">'
        '<span style="font-size:13px;color:#888">Predicted confidence</span>'
        '<span style="font-size:13px;font-weight:600;color:' + color + '">'
        + f"{confidence:.1f}%" + "</span></div>"
        '<div style="background:rgba(255,255,255,0.1);border-radius:10px;height:10px;overflow:hidden">'
        '<div style="background:' + color + ';width:' + str(bar_w) + '%'
        ';height:100%;border-radius:10px"></div></div></div>'

        '<table style="width:100%;border-collapse:collapse;font-size:13px">'
        '<tr style="color:#666;font-size:11px;text-transform:uppercase">'
        '<th style="text-align:left;padding:4px 10px">Class</th>'
        '<th style="text-align:left;padding:4px 10px">Mean Prob</th>'
        '<th style="text-align:left;padding:4px 10px">Std Dev</th></tr>'
        + rows
        + "</table></div>"
    )

    recommendation = "**Clinical Recommendation**\n\n" + risk["recommendation"]
    return overlay_img, risk_html, recommendation


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GRADIO INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}
.app-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1b2a 100%);
    border-radius: 16px;
    margin-bottom: 20px;
    border: 1px solid rgba(100, 150, 255, 0.15);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.app-header h1 { color: #e0e0ff; font-size: 2em; margin: 0; letter-spacing: 1px; }
.app-header p  { color: #8888bb; margin: 8px 0 0 0; font-size: 0.95em; }
footer { display: none !important; }
"""

with gr.Blocks(css=custom_css, title="OsteoVision AI", theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div class="app-header">
        <h1>&#129460; OsteoVision AI</h1>
        <p>AI-Powered Osteoporosis Risk Screening with Explainable Grad-CAM</p>
        <p style="font-size:0.8em;color:#555;margin-top:12px">
            Team Techtoli | AesCode Hackathon — Problem Statement 3
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📤 Upload X-ray Radiograph (JPEG / PNG)",
                type="pil",
                height=300,
            )
            analyze_btn = gr.Button("🔍 Analyze Radiograph", variant="primary", size="lg")
            gr.Markdown("""\
> **Instructions**: Upload a musculoskeletal hand/wrist X-ray radiograph.
> The AI runs Monte Carlo Dropout for uncertainty quantification and
> Grad-CAM for explainability.

> ⚠️ **Disclaimer**: Screening aid only — does not replace clinical
> diagnosis or DEXA scan evaluation.
""")

        with gr.Column(scale=2):
            gradcam_output = gr.Image(
                label="🔥 Grad-CAM Overlay (JET · α=0.4)",
                height=280,
            )
            risk_output       = gr.HTML(label="Risk Classification")
            recommendation_out = gr.Markdown(label="📋 Clinical Recommendation")

    analyze_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[gradcam_output, risk_output, recommendation_out],
    )

    gr.HTML("""
    <div style="text-align:center;padding:16px;margin-top:20px;color:#555;
                font-size:0.85em;border-top:1px solid rgba(255,255,255,0.1)">
        <strong>OsteoVision AI v2.0</strong> — EfficientNet-B0 + Grad-CAM + MC Dropout<br/>
        3-class risk output: Normal / Osteopenia / Osteoporosis<br/>
        Built for rural healthcare screening · © 2026 Team Techtoli | AesCode Hackathon
    </div>
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LAUNCH
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
