"""
 GradCAM Explainability for Plant Disease Classifier

"""

import argparse
import os
import random
import sys

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from finetune import PlantDiseaseClassifier
from model import Encoder

sys.stdout.reconfigure(encoding="utf-8")


# ============================================================================
# 1.  GradCAM Engine
# ============================================================================

class GradCAM:
    """
    GradCAM implementation using forward/backward hooks.

    Hooks are registered on the target layer during __init__ and
    automatically removed after each call to avoid memory leaks.

    Args:
        model        : the full PlantDiseaseClassifier
        target_layer : the nn.Module to hook  (ResNet-18 layer4)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None   # forward  feature maps
        self._gradients   = None   # backward gradients

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    # ------------------------------------------------------------------
    def _save_activations(self, module, input, output):
        """Forward hook: store the layer output (feature maps)."""
        self._activations = output.detach()   # (1, C, H, W)

    def _save_gradients(self, module, grad_input, grad_output):
        """Backward hook: store the gradient w.r.t. layer output."""
        self._gradients = grad_output[0].detach()   # (1, C, H, W)

    # ------------------------------------------------------------------
    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Generate a GradCAM heatmap for the given input.

        Args:
            input_tensor : (1, 3, H, W) normalised image tensor
            class_idx    : target class index.
                           If None, uses the predicted (argmax) class.

        Returns:
            cam          : numpy array (H, W), values in [0, 1]
            pred_idx     : predicted class index
            pred_prob    : predicted class probability (softmax)
            all_probs    : full softmax probability vector
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # --- Forward pass ---
        logits    = self.model(input_tensor)             # (1, num_classes)
        all_probs = F.softmax(logits, dim=1)
        pred_idx  = logits.argmax(dim=1).item()
        pred_prob = all_probs[0, pred_idx].item()

        target_idx = class_idx if class_idx is not None else pred_idx

        # --- Backward pass for target class ---
        self.model.zero_grad()
        score = logits[0, target_idx]
        score.backward()

        # --- GradCAM computation ---
        # Gradients: (1, C, H, W) -> global average pool -> (C,)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # Weight the activation maps by the importance weights
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)                                              # keep positive influence only

        # Upsample to input resolution
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],   # (H_input, W_input)
            mode="bilinear",
            align_corners=False,
        )

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_idx, pred_prob, all_probs.detach().cpu().numpy()[0]

    # ------------------------------------------------------------------
    def remove_hooks(self):
        """Must be called after use to release hook memory."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ============================================================================
# 2.  Visualisation Helpers
# ============================================================================

def cam_to_heatmap(cam: np.ndarray) -> np.ndarray:
    """
    Convert a [0,1] float CAM to a JET colourmap BGR uint8 image.
    Uses cv2 for the colourmap application.
    """
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap   = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)   # BGR uint8
    return heatmap   # (H, W, 3)  BGR


def overlay_cam(original_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Blend GradCAM heatmap onto the original RGB image.

    Args:
        original_rgb : (H, W, 3)  uint8 RGB
        cam          : (H, W)     float [0, 1]
        alpha        : heatmap opacity (0=invisible, 1=fully opaque)

    Returns:
        blended RGB uint8 image
    """
    heatmap_bgr = cam_to_heatmap(cam)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended     = (original_rgb.astype(float) * (1 - alpha)
                   + heatmap_rgb.astype(float) * alpha).clip(0, 255).astype(np.uint8)
    return blended


def save_panel(original_rgb, cam, overlay, class_name, pred_name, prob, correct, save_path):
    """
    Save a 3-panel figure: Original | GradCAM Heatmap | Overlay.
    Title shows true class, predicted class, confidence, and correct/wrong marker.
    """
    heatmap_rgb = cv2.cvtColor(cam_to_heatmap(cam), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")

    titles = ["Original Image", "GradCAM Heatmap", "Overlay"]
    images = [original_rgb, heatmap_rgb, overlay]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    verdict  = "CORRECT" if correct else "WRONG"
    color    = "#00e676" if correct else "#ff5252"
    sup      = (f"True: {class_name}   |   "
                f"Predicted: {pred_name}   |   "
                f"Confidence: {prob*100:.1f}%   |   [{verdict}]")
    fig.suptitle(sup, color=color, fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def save_summary_grid(results: list, xai_dir: str, n_cols: int = 6):
    """
    Save one large grid image showing the GradCAM overlay for every class.
    Each cell shows: overlay image + true/pred label + confidence.

    Args:
        results  : list of dicts from run_inference()
        xai_dir  : output folder
        n_cols   : number of columns in the grid
    """
    n       = len(results)
    n_rows  = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.2, n_rows * 3.8))
    fig.patch.set_facecolor("#1a1a2e")
    axes_flat = axes.flatten() if n > 1 else [axes]

    for idx, (ax, res) in enumerate(zip(axes_flat, results)):
        ax.imshow(res["overlay"])
        ax.axis("off")

        correct = res["true_class"] == res["pred_class"]
        color   = "#00e676" if correct else "#ff5252"
        label   = (f"T: {res['true_class'].split('___')[-1]}\n"
                   f"P: {res['pred_class'].split('___')[-1]}\n"
                   f"{res['confidence']*100:.1f}%")
        ax.set_title(label, color=color, fontsize=7, pad=3)

    # Hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("GradCAM Summary -- One image per class",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(xai_dir, "summary_grid.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[GradCAM] Summary grid saved >> {out_path}")


# ============================================================================
# 3.  Image Preprocessing
# ============================================================================

def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """Same deterministic transform used during validation."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation and convert to uint8 RGB numpy array.
    Used to display the original image alongside GradCAM.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = (tensor.squeeze().cpu() * std + mean).clamp(0, 1)
    img  = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


# ============================================================================
# 4.  Sample one image per class from val/
# ============================================================================

def sample_one_per_class(val_dir: str, seed: int = 42) -> dict:
    """
    For each class subfolder in val_dir, randomly pick one image path.

    Returns:
        { class_name: image_path, ... }
    """
    random.seed(seed)
    VALID = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples = {}

    class_dirs = sorted([
        d for d in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, d))
    ])

    if not class_dirs:
        raise ValueError(f"No class subfolders found in '{val_dir}'")

    for cls in class_dirs:
        cls_path = os.path.join(val_dir, cls)
        images   = [
            os.path.join(cls_path, f)
            for f in os.listdir(cls_path)
            if os.path.splitext(f)[1].lower() in VALID
        ]
        if images:
            samples[cls] = random.choice(images)
        else:
            print(f"[Warning] No images found for class '{cls}', skipping.")

    print(f"[Inference] Sampled {len(samples)} images ({len(class_dirs)} classes)")
    return samples


# ============================================================================
# 5.  Load Model
# ============================================================================

def load_model(model_path: str, device: torch.device) -> tuple:
    """
    Load the fine-tuned PlantDiseaseClassifier from a checkpoint.

    Returns:
        model       : PlantDiseaseClassifier in eval mode
        class_names : sorted list of class name strings
        embed_dim   : encoder embedding dimension
    """
    print(f"[Model] Loading from: {model_path}")
    ckpt = torch.load(model_path, map_location=device)

    embed_dim   = ckpt.get("embed_dim",   256)
    num_classes = ckpt.get("num_classes", 38)
    strategy    = ckpt.get("strategy",    "partial")
    class_names = ckpt["class_names"]

    encoder = Encoder(embed_dim=embed_dim)
    model   = PlantDiseaseClassifier(
        encoder=encoder,
        num_classes=num_classes,
        strategy=strategy,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"[Model] Loaded  --  {num_classes} classes  |  strategy='{strategy}'  |  embed_dim={embed_dim}")
    return model, class_names, embed_dim


# ============================================================================
# 6.  Run Inference + GradCAM for all sampled images
# ============================================================================

def run_inference(
    model:       nn.Module,
    class_names: list,
    samples:     dict,
    xai_dir:     str,
    device:      torch.device,
    image_size:  int = 224,
) -> list:
    """
    For each sampled image:
      1. Preprocess
      2. Run inference (get predicted class + confidence)
      3. Compute GradCAM heatmap
      4. Save panel + individual artefacts
      5. Collect results for summary grid

    GradCAM target layer: encoder.backbone[-2]  (ResNet-18 layer4)
    This is the last spatial conv block (output: 7x7 feature maps at 224x224).
    It captures the highest-level spatial semantics — ideal for disease localisation.

    Returns:
        list of result dicts for summary grid generation
    """
    transform = get_val_transform(image_size)

    # Register GradCAM on ResNet layer4 (index -2 of backbone Sequential,
    # since -1 is the AdaptiveAvgPool that collapses spatial dims)
    target_layer = model.encoder.backbone[-2]
    gradcam      = GradCAM(model, target_layer)

    results      = []
    correct_count = 0

    print(f"\n[Inference] Processing {len(samples)} images...\n")

    for true_class, img_path in samples.items():
        # --- Load + preprocess ---
        pil_img    = Image.open(img_path).convert("RGB")
        tensor     = transform(pil_img).unsqueeze(0).to(device)  # (1,3,H,W)
        original   = denormalise(tensor)                          # uint8 RGB for display

        # --- GradCAM forward+backward ---
        cam, pred_idx, pred_prob, all_probs = gradcam.generate(tensor)

        pred_class = class_names[pred_idx]
        correct    = (true_class == pred_class)
        if correct:
            correct_count += 1

        # --- Build visualisations ---
        heatmap_rgb = cv2.cvtColor(cam_to_heatmap(cam), cv2.COLOR_BGR2RGB)
        overlay     = overlay_cam(original, cam, alpha=0.45)

        # --- Save to xai/<ClassName>/ ---
        out_dir = os.path.join(xai_dir, true_class)
        os.makedirs(out_dir, exist_ok=True)

        # 1. Original
        Image.fromarray(original).save(os.path.join(out_dir, "original.jpg"))

        # 2. Standalone heatmap
        Image.fromarray(heatmap_rgb).save(os.path.join(out_dir, "gradcam_heatmap.png"))

        # 3. Overlay
        Image.fromarray(overlay).save(os.path.join(out_dir, "gradcam_overlay.png"))

        # 4. Side-by-side panel
        save_panel(
            original_rgb=original,
            cam=cam,
            overlay=overlay,
            class_name=true_class,
            pred_name=pred_class,
            prob=pred_prob,
            correct=correct,
            save_path=os.path.join(out_dir, "gradcam_panel.png"),
        )

        # --- Top-5 predictions for this image ---
        top5_idx  = all_probs.argsort()[::-1][:5]
        top5_info = [(class_names[i], all_probs[i]) for i in top5_idx]

        verdict = "CORRECT" if correct else "WRONG"
        print(f"  [{verdict:<7}] {true_class:<45}  pred={pred_class:<45}  conf={pred_prob*100:.1f}%")

        results.append({
            "true_class":  true_class,
            "pred_class":  pred_class,
            "confidence":  pred_prob,
            "correct":     correct,
            "overlay":     overlay,
            "top5":        top5_info,
        })

    gradcam.remove_hooks()

    # --- Print summary ---
    total    = len(results)
    accuracy = correct_count / total if total > 0 else 0.0
    print(f"\n[Inference] Accuracy on sampled images: {correct_count}/{total}  ({accuracy*100:.1f}%)")

    return results


# ============================================================================
# 7.  Save Per-class Confidence Bar Chart
# ============================================================================

def save_confidence_report(results: list, xai_dir: str):
    """
    Horizontal bar chart showing prediction confidence per class.
    Green = correct, Red = wrong.
    """
    classes     = [r["true_class"].split("___")[-1] for r in results]
    confidences = [r["confidence"] * 100 for r in results]
    colours     = ["#00e676" if r["correct"] else "#ff5252" for r in results]

    n = len(results)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.35)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    bars = ax.barh(range(n), confidences, color=colours, edgecolor="none", height=0.7)

    ax.set_yticks(range(n))
    ax.set_yticklabels(classes, fontsize=8, color="white")
    ax.set_xlabel("Confidence (%)", color="white", fontsize=10)
    ax.set_title("Prediction Confidence per Class", color="white", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.tick_params(colors="white")
    ax.axvline(x=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Value labels on bars
    for bar, val in zip(bars, confidences):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=7, color="white")

    legend_patches = [
        mpatches.Patch(color="#00e676", label="Correct"),
        mpatches.Patch(color="#ff5252", label="Wrong"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              framealpha=0.3, labelcolor="white", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(xai_dir, "confidence_report.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[GradCAM] Confidence report saved >> {out_path}")


# ============================================================================
# 8.  Argument Parsing
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(description="GradCAM Inference -- Plant Disease")

    parser.add_argument("--model_path",  type=str, default="finetune_checkpoints/best_model.pt",
                        help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--val_dir",     type=str, default="Plant Disease/Images/Data/valid",
                        help="Validation directory with class subfolders")
    parser.add_argument("--xai_dir",     type=str, default="xai",
                        help="Output directory for GradCAM visualisations")
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--seed",        type=int, default=3,
                        help="Random seed for reproducible image sampling")

    return parser.parse_args()


# ============================================================================
# 9.  Main
# ============================================================================

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.xai_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  GradCAM Inference -- Plant Disease Classifier")
    print(f"{'='*55}")
    print(f"  Device     : {device}")
    print(f"  Model      : {args.model_path}")
    print(f"  Val dir    : {args.val_dir}")
    print(f"  Output dir : {args.xai_dir}")
    print(f"{'='*55}\n")

    # 1. Load model
    model, class_names, _ = load_model(args.model_path, device)

    # 2. Sample one image per class
    samples = sample_one_per_class(args.val_dir, seed=args.seed)

    # 3. Run inference + generate GradCAM for all samples
    results = run_inference(
        model=model,
        class_names=class_names,
        samples=samples,
        xai_dir=args.xai_dir,
        device=device,
        image_size=args.image_size,
    )

    # 4. Summary grid (all classes in one image)
    save_summary_grid(results, args.xai_dir, n_cols=6)

    # 5. Confidence bar chart
    save_confidence_report(results, args.xai_dir)

    # 6. Final summary
    correct = sum(1 for r in results if r["correct"])
    total   = len(results)
    print(f"\n{'='*55}")
    print(f"  Done!  {correct}/{total} correct  ({correct/total*100:.1f}% accuracy)")
    print(f"\n  Output structure:")
    print(f"  xai/")
    print(f"    <ClassName>/")
    print(f"      original.jpg          -- raw input image")
    print(f"      gradcam_heatmap.png   -- standalone heatmap")
    print(f"      gradcam_overlay.png   -- heatmap blended on image")
    print(f"      gradcam_panel.png     -- 3-panel side-by-side view")
    print(f"    summary_grid.png        -- all {total} classes in one grid")
    print(f"    confidence_report.png   -- per-class confidence bar chart")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
