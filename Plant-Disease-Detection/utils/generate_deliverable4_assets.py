"""Generate Grad-CAM, LIME, and SHAP visualizations for Deliverable 4."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless image export
import matplotlib.pyplot as plt
import numpy as np
import shap
import timm
import torch
import torch.nn.functional as F
from lime import lime_image
from PIL import Image
from skimage.segmentation import mark_boundaries
from torchvision import transforms
j
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parents[0].parent
IMG_PATH = ROOT / "demo_images" / "4.JPG"
OUTPUT_DIR = ROOT / "results" / "interpretability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 34  # Updated for Pakistan dataset (Rice, Cotton, Wheat, Mango + PlantVillage crops)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "Blueberry___healthy",
    "Apple___Bacterial_spot",
]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class GradCAM:
    """Minimal Grad-CAM helper."""

    def __init__(self, model: torch.nn.Module, target_layer: str) -> None:
        self.model = model
        self.gradients = None
        self.activations = None

        layers = dict(model.named_modules())
        if target_layer not in layers:
            raise KeyError(f"Layer {target_layer} not found in model")
        layer = layers[target_layer]
        layer.register_forward_hook(self._forward_hook)
        layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, _, __, output):
        self.activations = output.detach()

    def _backward_hook(self, _, grad_input, grad_output):
        del grad_input  # unused
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int | None = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1))
        logits[:, class_idx].backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam, class_idx


def load_model() -> torch.nn.Module:
    model = timm.create_model("tf_efficientnet_b4_ns", pretrained=True, num_classes=NUM_CLASSES)
    model.eval().to(DEVICE)
    return model


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    heatmap_rgb = plt.cm.inferno(heatmap)[..., :3]
    heatmap_img = Image.fromarray((heatmap_rgb * 255).astype(np.uint8)).resize(image.size)
    return Image.blend(image.convert("RGB"), heatmap_img, alpha)


def run_gradcam(model: torch.nn.Module, pil_img: Image.Image, tensor: torch.Tensor) -> Path:
    cam = GradCAM(model, "conv_head")
    heatmap, idx = cam.generate(tensor)
    blended = overlay_heatmap(pil_img, heatmap)
    out_path = OUTPUT_DIR / "gradcam_effb4.png"
    blended.save(out_path)
    print(f"Grad-CAM saved to {out_path} (class idx: {idx})")
    return out_path


def run_lime(model: torch.nn.Module, pil_img: Image.Image) -> Path:
    explainer = lime_image.LimeImageExplainer()

    def lime_predict(batch_images):
        tensors = []
        for arr in batch_images:
            candidate = Image.fromarray(arr.astype(np.uint8))
            tensors.append(TRANSFORM(candidate))
        stacked = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            logits = model(stacked)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    explanation = explainer.explain_instance(
        np.array(pil_img.resize((224, 224))),
        lime_predict,
        top_labels=1,
        hide_color=0,
        num_samples=400,
    )

    lime_img, lime_mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        hide_rest=False,
        num_features=6,
        min_weight=0.05,
    )

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(mark_boundaries(lime_img / 255.0, lime_mask))
    plt.axis("off")
    plt.title("LIME · Local Regions", fontsize=12)
    out_path = OUTPUT_DIR / "lime_effb4.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"LIME visualization saved to {out_path}")
    return out_path


def run_shap(model: torch.nn.Module, pil_img: Image.Image, tensor: torch.Tensor) -> Path | None:
    try:
        background = torch.stack(
            [
                TRANSFORM(pil_img),
                TRANSFORM(pil_img.rotate(12)),
                TRANSFORM(pil_img.rotate(-12)),
                TRANSFORM(pil_img.transpose(Image.FLIP_LEFT_RIGHT)),
            ]
        ).to(DEVICE)

        explainer = shap.DeepExplainer(model, background)
        shap_values, indexes = explainer.shap_values(tensor, ranked_outputs=1)
        shap_values = shap_values[0]
        class_idx = int(indexes[0][0])

        input_np = tensor.detach().cpu().numpy()
        input_np = np.transpose(input_np, (0, 2, 3, 1))
        shap_np = np.transpose(shap_values, (0, 2, 3, 1))

        shap.image_plot(
            shap_values=shap_np,
            pixel_values=input_np,
            show=False,
            labels=[CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Class {class_idx}"],
        )
        out_path = OUTPUT_DIR / "shap_effb4.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"SHAP visualization saved to {out_path} (class idx: {class_idx})")
        return out_path
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] SHAP visualization failed: {exc}")
        return None


def main() -> None:
    if not IMG_PATH.exists():
        raise FileNotFoundError(f"Sample image not found at {IMG_PATH}")

    pil_img = Image.open(IMG_PATH).convert("RGB")
    tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    model = load_model()

    gradcam_path = run_gradcam(model, pil_img, tensor)
    lime_path = run_lime(model, pil_img)
    shap_path = run_shap(model, pil_img, tensor)

    print("\nArtifacts ready:")
    for path in [gradcam_path, lime_path, shap_path]:
        if path is None:
            continue
        print(f" - {path}")


if __name__ == "__main__":
    main()
