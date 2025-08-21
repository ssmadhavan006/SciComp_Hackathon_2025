import argparse
import os
import urllib.request
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_models(device: torch.device):
    """
    Load pretrained ResNet50 and VGG16 in eval mode.
    Tries new torchvision API first (weights=...), falls back to pretrained=True.
    Returns (resnet, vgg, categories) where categories are ImageNet class names if available.
    """
    resnet = None
    vgg = None
    categories = None

    try:
        # Newer torchvision API (0.13+)
        from torchvision.models import ResNet50_Weights, VGG16_Weights
        resnet_w = ResNet50_Weights.IMAGENET1K_V2
        vgg_w = VGG16_Weights.IMAGENET1K_V1

        resnet = models.resnet50(weights=resnet_w).to(device).eval()
        vgg = models.vgg16(weights=vgg_w).to(device).eval()
        # Prefer categories from weights metadata
        categories = resnet_w.meta.get("categories", None)
    except Exception:
        # Older API compatibility
        resnet = models.resnet50(pretrained=True).to(device).eval()
        vgg = models.vgg16(pretrained=True).to(device).eval()
        categories = None

    return resnet, vgg, categories


def build_preprocess():
    """
    Standard ImageNet preprocessing per PRD.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image(image_path: str) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    return img


def ensure_imagenet_classes() -> Optional[List[str]]:
    """
    Try to ensure we have human-readable ImageNet class names.
    Strategy:
    1) If a local imagenet_classes.txt exists, use it.
    2) Try to download the canonical file (no extra pip deps).
    3) Return None if unavailable; downstream will fallback to indices.
    """
    local_file = "imagenet_classes.txt"
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

    if os.path.exists(local_file):
        try:
            with open(local_file, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            if len(classes) == 1000:
                return classes
        except Exception:
            pass

    # Try download if not available
    try:
        urllib.request.urlretrieve(url, local_file)
        with open(local_file, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        if len(classes) == 1000:
            return classes
    except Exception:
        return None

    return None


def get_categories_fallback(weights_categories: Optional[List[str]]) -> Optional[List[str]]:
    """
    Prefer weights' categories if available; else try to fetch txt file.
    """
    if weights_categories and len(weights_categories) == 1000:
        return list(weights_categories)
    return ensure_imagenet_classes()


def topk_predictions(probs: torch.Tensor, k: int, categories: Optional[List[str]]) -> List[Tuple[int, str, float]]:
    """
    Given probabilities tensor of shape [1000], return top-k tuples (idx, name, prob).
    """
    top_probs, top_idxs = torch.topk(probs, k)
    results = []
    for p, i in zip(top_probs.tolist(), top_idxs.tolist()):
        name = categories[i] if categories and i < len(categories) else f"class_{i}"
        results.append((i, name, float(p)))
    return results


def print_topk(label: str, probs: torch.Tensor, k: int, categories: Optional[List[str]]):
    topk = topk_predictions(probs, k, categories)
    print(f"Top-{k} predictions for {label}:")
    for idx, name, prob in topk:
        print(f"  {idx:4d}: {name:25s} | {prob:.4f}")
    print("")


def clamp_normalized(img: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Clamp a normalized image tensor (NCHW) so that its de-normalized values are in [0,1].
    Equivalent to per-channel clamping in normalized space:
        min_c = (0 - mean_c) / std_c
        max_c = (1 - mean_c) / std_c
    """
    device = img.device
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    min_t = (0.0 - mean_t) / std_t
    max_t = (1.0 - mean_t) / std_t
    return torch.max(torch.min(img, max_t), min_t)


def denormalize(img: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Convert normalized image tensor (NCHW) back to [0,1] range for visualization.
    """
    device = img.device
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    return img * std_t + mean_t


def fgsm_attack(model: torch.nn.Module, image: torch.Tensor, label: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Untargeted FGSM: maximize loss for the true (or predicted) label.
    image: normalized input, shape [1,3,224,224], requires_grad will be set in this function.
    """
    image_adv = image.clone().detach().requires_grad_(True)

    output = model(image_adv)
    loss = torch.nn.CrossEntropyLoss()(output, label)

    model.zero_grad()
    if image_adv.grad is not None:
        image_adv.grad.zero_()
    loss.backward()

    sign_grad = image_adv.grad.sign()
    adv = image_adv + epsilon * sign_grad
    adv = clamp_normalized(adv, IMAGENET_MEAN, IMAGENET_STD)
    return adv.detach()


def run_inference(model: torch.nn.Module, input_tensor: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
    """
    Returns: (pred_class_idx, pred_conf, probs[1000])
    """
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output[0], dim=0)
        pred_idx = int(torch.argmax(probs).item())
        pred_conf = float(probs[pred_idx].item())
    return pred_idx, pred_conf, probs


def visualize(original_img: Image.Image,
              adv_tensor: torch.Tensor,
              orig_label: str,
              adv_label: str,
              save_path: str = "fgsm_result.png"):
    """
    Show original PIL image and adversarial tensor (normalized) side-by-side,
    and save the figure.
    """
    adv_vis = denormalize(adv_tensor.detach().cpu(), IMAGENET_MEAN, IMAGENET_STD)
    adv_vis = torch.clamp(adv_vis, 0.0, 1.0).squeeze(0)
    adv_pil = transforms.ToPILImage()(adv_vis)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original_img)
    ax1.set_title(f"Original: {orig_label}")
    ax1.axis("off")

    ax2.imshow(adv_pil)
    ax2.set_title(f"Adversarial: {adv_label}")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison figure to: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Phase 1: ResNet50/VGG16 + FGSM Attack on single image")
    parser.add_argument("--image", type=str, default="stop_sign.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vgg16"], help="Model to run FGSM on")
    parser.add_argument("--epsilon", type=float, default=0.02, help="FGSM epsilon (perturbation magnitude)")
    parser.add_argument("--topk", type=int, default=5, help="Top-k predictions to display")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}. Please place an image in this folder or pass --image PATH")

    device = get_device()
    print(f"Using device: {device}")

    resnet, vgg, weights_categories = build_models(device)
    preprocess = build_preprocess()

    # Use weights categories if available; else try to fetch local/remote txt file
    categories = get_categories_fallback(weights_categories)

    # Load and preprocess image
    img = load_image(args.image)
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # Basic inference on both models for completeness
    print("Running inference on ResNet50...")
    r_idx, r_conf, r_probs = run_inference(resnet, input_tensor)
    r_name = categories[r_idx] if categories and r_idx < len(categories) else f"class_{r_idx}"
    print(f"ResNet50 prediction: {r_idx} ({r_name}) | conf={r_conf:.4f}")
    print_topk("ResNet50", r_probs, args.topk, categories)

    print("Running inference on VGG16...")
    v_idx, v_conf, v_probs = run_inference(vgg, input_tensor)
    v_name = categories[v_idx] if categories and v_idx < len(categories) else f"class_{v_idx}"
    print(f"VGG16 prediction: {v_idx} ({v_name}) | conf={v_conf:.4f}")
    print_topk("VGG16", v_probs, args.topk, categories)

    # Pick model for FGSM attack
    if args.model.lower() == "resnet50":
        atk_model = resnet
        orig_idx = r_idx
        orig_name = r_name
    else:
        atk_model = vgg
        orig_idx = v_idx
        orig_name = v_name

    # Prepare label for untargeted FGSM (use model's original prediction)
    label = torch.tensor([orig_idx], device=device, dtype=torch.long)

    # Run FGSM
    adv_img = fgsm_attack(atk_model, input_tensor, label, epsilon=args.epsilon)

    # Re-classify adversarial image (on the same attack model)
    adv_idx, adv_conf, adv_probs = run_inference(atk_model, adv_img)
    adv_name = categories[adv_idx] if categories and adv_idx < len(categories) else f"class_{adv_idx}"

    print(f"Adversarial prediction ({args.model}): {adv_idx} ({adv_name}) | conf={adv_conf:.4f}")
    print_topk(f"{args.model} (Adversarial)", adv_probs, args.topk, categories)

    changed = adv_idx != orig_idx
    print(f"Prediction changed: {changed}")
    visualize(
        original_img=img,
        adv_tensor=adv_img,
        orig_label=f"{orig_idx} ({orig_name})",
        adv_label=f"{adv_idx} ({adv_name})",
        save_path="fgsm_result.png",
    )


if __name__ == "__main__":
    main()