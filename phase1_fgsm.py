import argparse
import os
import urllib.request
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.utils import save_image

# 🔍 --- GPU/CUDA DEBUG INFO ---
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ CUDA is NOT available. Using CPU only.")
print("-" * 60)
# --- END DEBUG BLOCK ---

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path: str) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    return img


def ensure_imagenet_classes() -> Optional[List[str]]:
    """
    Try to ensure we have human-readable ImageNet class names.
    """
    local_file = "imagenet_classes.txt"
    # ✅ Fixed: Removed trailing spaces
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

    if os.path.exists(local_file):
        try:
            with open(local_file, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            if len(classes) == 1000:
                return classes
        except Exception as e:
            print(f"⚠️ Failed to read {local_file}: {e}")

    try:
        print("📥 Downloading imagenet_classes.txt...")
        urllib.request.urlretrieve(url, local_file)
        with open(local_file, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        if len(classes) == 1000:
            return classes
        else:
            print(f"⚠️ Downloaded file has {len(classes)} entries, expected 1000.")
    except Exception as e:
        print(f"⚠️ Failed to download class names: {e}")

    return None


def get_categories_fallback(weights_categories: Optional[List[str]]) -> Optional[List[str]]:
    if weights_categories and len(weights_categories) == 1000:
        return list(weights_categories)
    return ensure_imagenet_classes()


def build_preprocess():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def topk_predictions(probs: torch.Tensor, k: int, categories: Optional[List[str]]) -> List[Tuple[int, str, float]]:
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
    device = img.device
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    min_t = (0.0 - mean_t) / std_t
    max_t = (1.0 - mean_t) / std_t
    return torch.max(torch.min(img, max_t), min_t)


def denormalize(img: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    device = img.device
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    return img * std_t + mean_t


# -------------------------------
# ✅ DEFENSE FUNCTIONS
# -------------------------------

def apply_gaussian_blur(pil_img: Image.Image, radius: float = 1.0) -> Image.Image:
    """Reduce high-frequency noise (adversarial perturbations)."""
    return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))


def simulate_jpeg_compression(pil_img: Image.Image, quality: int = 75) -> Image.Image:
    """Simulate JPEG compression to remove small perturbations."""
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf)


def reduce_color_depth(pil_img: Image.Image, levels: int = 64) -> Image.Image:
    """Feature squeezing: reduce color resolution."""
    np_img = np.array(pil_img) // (256 // levels) * (256 // levels)
    return Image.fromarray(np_img)


def apply_defense(pil_img: Image.Image, defense: str = "none") -> Image.Image:
    """
    Apply defense before preprocessing.
    Options: "blur", "jpeg", "color", "all", "none"
    """
    if defense == "none":
        return pil_img
    if defense == "blur":
        return apply_gaussian_blur(pil_img, radius=1.0)
    if defense == "jpeg":
        return simulate_jpeg_compression(pil_img, quality=75)
    if defense == "color":
        return reduce_color_depth(pil_img, levels=64)
    if defense == "all":
        img = apply_gaussian_blur(pil_img, radius=0.8)
        img = simulate_jpeg_compression(img, quality=80)
        img = reduce_color_depth(img, levels=64)
        return img
    return pil_img


# -------------------------------
# ✅ FAST INFERENCE (Mixed Precision)
# -------------------------------

@torch.no_grad()
def run_inference_fast(model: torch.nn.Module, input_tensor: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
    """
    Faster inference using autocast (half precision on CUDA).
    """
    device = input_tensor.device
    with torch.autocast(device_type=str(device).split(":")[0], enabled=device.type == "cuda"):
        output = model(input_tensor)
        probs = F.softmax(output[0], dim=0)
        pred_idx = int(torch.argmax(probs).item())
        pred_conf = float(probs[pred_idx].item())
    return pred_idx, pred_conf, probs


# -------------------------------
# ✅ ATTACKS
# -------------------------------

def pgd_attack(model: torch.nn.Module,
               image: torch.Tensor,
               label: torch.Tensor,
               epsilon: float = 0.03,
               alpha: float = 0.01,
               iters: int = 40,
               targeted: bool = False,
               target_label: Optional[torch.Tensor] = None) -> torch.Tensor:
    device = image.device
    original = image.clone().detach()
    adv = image.clone().detach()

    for _ in range(iters):
        adv.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        output = model(adv)
        if targeted and target_label is not None:
            loss = torch.nn.CrossEntropyLoss()(output, target_label)
            direction = -1.0
        else:
            loss = torch.nn.CrossEntropyLoss()(output, label)
            direction = 1.0

        loss.backward()

        with torch.no_grad():
            grad_sign = adv.grad.sign()
            adv = adv + direction * alpha * grad_sign
            adv = torch.max(torch.min(adv, original + epsilon), original - epsilon)
            adv = clamp_normalized(adv, IMAGENET_MEAN, IMAGENET_STD)

        adv = adv.detach()

    return adv


def fgsm_attack(model: torch.nn.Module,
                image: torch.Tensor,
                label: torch.Tensor,
                epsilon: float = 0.03,
                targeted: bool = False,
                target_label: Optional[torch.Tensor] = None) -> torch.Tensor:
    device = image.device
    original = image.clone().detach()
    adv = image.clone().detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        output = model(adv)
        if targeted and target_label is not None:
            loss = torch.nn.CrossEntropyLoss()(output, target_label)
        else:
            loss = torch.nn.CrossEntropyLoss()(output, label)
        loss.backward()

    grad_sign = adv.grad.sign()
    direction = -1.0 if targeted else 1.0
    adv = adv + direction * epsilon * grad_sign
    adv = torch.max(torch.min(adv, original + epsilon), original - epsilon)
    adv = clamp_normalized(adv, IMAGENET_MEAN, IMAGENET_STD)

    return adv.detach()


def compute_metrics(model: torch.nn.Module,
                    orig_image: torch.Tensor,
                    adv_image: torch.Tensor,
                    orig_label_idx: int,
                    norm: str = "linf") -> dict:
    with torch.no_grad():
        orig_output = model(orig_image)
        adv_output = model(adv_image)

    orig_probs = F.softmax(orig_output, dim=1)
    adv_probs = F.softmax(adv_output, dim=1)

    orig_pred = int(orig_probs.argmax(dim=1).item())
    adv_pred = int(adv_probs.argmax(dim=1).item())
    success = int(orig_pred != adv_pred)

    conf_shift = float(orig_probs[0, orig_label_idx].item() - adv_probs[0, orig_label_idx].item())

    diff = (adv_image - orig_image).detach().view(-1)
    if norm.lower() == "l2":
        perturbation = float(torch.norm(diff, p=2).item())
    else:
        perturbation = float(torch.norm(diff, p=float("inf")).item())

    return {
        "original_class": orig_pred,
        "adv_class": adv_pred,
        "success": success,
        "confidence_shift": conf_shift,
        "perturbation_norm": perturbation,
        "norm": norm.lower(),
    }


def run_batch_attack(model: torch.nn.Module,
                     image_paths: List[str],
                     labels: Optional[List[int]],
                     attack: str,
                     epsilon: float,
                     alpha: float,
                     iters: int,
                     targeted: bool,
                     target_idx: Optional[int],
                     export_dir: str,
                     categories: Optional[List[str]],
                     norm: str,
                     device: torch.device,
                     defense: str = "none"):
    os.makedirs(export_dir, exist_ok=True)
    results = []

    preprocess = build_preprocess()

    if labels is not None and len(labels) != len(image_paths):
        raise ValueError("--labels length must match number of --images")

    for i, img_path in enumerate(image_paths):
        try:
            pil_img = load_image(img_path)
        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")
            continue

        defended_pil = apply_defense(pil_img, defense=defense)
        orig = preprocess(defended_pil).unsqueeze(0).to(device)

        if labels is None:
            with torch.no_grad():
                out = model(orig)
                orig_idx = int(out.softmax(dim=1).argmax(dim=1).item())
        else:
            orig_idx = int(labels[i])

        label = torch.tensor([orig_idx], device=device, dtype=torch.long)
        target_label = torch.tensor([target_idx], device=device, dtype=torch.long) if targeted and target_idx is not None else None

        if attack == "pgd":
            adv = pgd_attack(model, orig, label, epsilon, alpha, iters, targeted, target_label)
        else:
            adv = fgsm_attack(model, orig, label, epsilon, targeted, target_label)

        adv_vis = denormalize(adv.detach().cpu(), IMAGENET_MEAN, IMAGENET_STD)
        adv_vis = torch.clamp(adv_vis, 0.0, 1.0)
        save_path = os.path.join(export_dir, f"adv_{i}.png")
        save_image(adv_vis, save_path)

        m = compute_metrics(model, orig, adv, orig_idx, norm)
        m["image"] = img_path
        m["adv_image"] = save_path
        m["original_label_name"] = categories[orig_idx] if categories and orig_idx < len(categories) else f"class_{orig_idx}"
        m["adv_label_name"] = categories[m['adv_class']] if categories and m['adv_class'] < len(categories) else f"class_{m['adv_class']}"
        results.append(m)

    df = pd.DataFrame(results)
    csv_path = os.path.join(export_dir, "attack_results.csv")
    try:
        df.to_csv(csv_path, index=False)
        print(f"Saved batch results to: {csv_path}")
    except PermissionError:
        print(f"❌ Cannot save: {csv_path} is locked. Close any program using it or change --export_dir.")
        raise

    return df


def visualize(original_img: Image.Image,
              adv_tensor: torch.Tensor,
              orig_label: str,
              adv_label: str,
              save_path: str = "fgsm_result.png"):
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


def load_model(name: str, device: torch.device):
    model = None
    categories = None
    name = name.lower()

    try:
        from torchvision.models import ResNet50_Weights, VGG16_Weights, DenseNet121_Weights
        if name == "resnet50":
            w = ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=w).to(device).eval()
            categories = w.meta.get("categories", None)
        elif name == "vgg16":
            w = VGG16_Weights.IMAGENET1K_V1
            model = models.vgg16(weights=w).to(device).eval()
            categories = w.meta.get("categories", None)
        elif name == "densenet121":
            w = DenseNet121_Weights.IMAGENET1K_V1
            model = models.densenet121(weights=w).to(device).eval()
            categories = w.meta.get("categories", None)
        else:
            raise ValueError(f"Unsupported model: {name}")
    except Exception as e:
        print(f"⚠️ Fallback: {e}")
        if name == "resnet50":
            model = models.resnet50(pretrained=True).to(device).eval()
        elif name == "vgg16":
            model = models.vgg16(pretrained=True).to(device).eval()
        elif name == "densenet121":
            model = models.densenet121(pretrained=True).to(device).eval()
        else:
            raise ValueError(f"Unsupported model: {name}")

    if categories is None or len(categories) != 1000:
        categories = get_categories_fallback(categories)

    return model, categories


def main():
    parser = argparse.ArgumentParser(description="Adversarial Attack Engine with Defenses & Speed")
    parser.add_argument("--image", type=str, default="stop_sign.jpg", help="Path to input image (single-image mode)")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vgg16", "densenet121"], help="Model to attack")
    parser.add_argument("--epsilon", type=float, default=0.03, help="Perturbation size (in normalized space)")
    parser.add_argument("--topk", type=int, default=5, help="Top-k predictions to show")
    parser.add_argument("--attack", type=str, default="fgsm", choices=["fgsm", "pgd"], help="Attack method")
    parser.add_argument("--alpha", type=float, default=0.01, help="PGD step size")
    parser.add_argument("--iters", type=int, default=20, help="PGD iterations")
    parser.add_argument("--targeted", action="store_true", help="Run targeted attack")
    parser.add_argument("--target_idx", type=int, default=954, help="Target class index (e.g., 954=banana)")
    parser.add_argument("--norm", type=str, default="linf", choices=["linf", "l2"], help="Perturbation norm")
    parser.add_argument("--images", type=str, nargs="+", help="List of image paths for batch attack")
    parser.add_argument("--labels", type=int, nargs="+", help="True labels for batch images (optional)")
    parser.add_argument("--batch_dir", type=str, help="Directory of images for batch processing")
    parser.add_argument("--export_dir", type=str, default="adv_outputs", help="Output directory for adversarial images and CSV")
    # ✅ New: Defense
    parser.add_argument("--defense", type=str, default="none", choices=["none", "blur", "jpeg", "color", "all"], help="Apply defense before inference")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load model
    atk_model, categories = load_model(args.model, device)
    preprocess = build_preprocess()

    # ==================== BATCH MODE ====================
    if args.images or args.batch_dir:
        image_paths = []
        if args.images:
            image_paths.extend(args.images)
        if args.batch_dir:
            if not os.path.exists(args.batch_dir):
                raise FileNotFoundError(f"Batch directory not found: {args.batch_dir}")
            for fname in os.listdir(args.batch_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(args.batch_dir, fname))
        if not image_paths:
            raise ValueError("No images found for batch processing")

        print(f"Running batch attack on {len(image_paths)} images with {args.model} using {args.attack.upper()}...")
        run_batch_attack(
            model=atk_model,
            image_paths=image_paths,
            labels=args.labels,
            attack=args.attack,
            epsilon=args.epsilon,
            alpha=args.alpha,
            iters=args.iters,
            targeted=args.targeted,
            target_idx=args.target_idx if args.targeted else None,
            export_dir=args.export_dir,
            categories=categories,
            norm=args.norm,
            device=device,
            defense=args.defense
        )
        return

    # ================= SINGLE-IMAGE MODE ================
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    img = load_image(args.image)

    # ✅ Apply defense before preprocessing
    defended_img = apply_defense(img, defense=args.defense)
    input_tensor = preprocess(defended_img).unsqueeze(0).to(device)

    print(f"Running inference on {args.model}...")
    orig_idx, orig_conf, orig_probs = run_inference_fast(atk_model, input_tensor)
    orig_name = categories[orig_idx] if categories and orig_idx < len(categories) else f"class_{orig_idx}"
    print(f"{args.model} prediction: {orig_idx} ({orig_name}) | conf={orig_conf:.4f}")
    print_topk(args.model, orig_probs, args.topk, categories)

    label = torch.tensor([orig_idx], device=device, dtype=torch.long)
    target_label = torch.tensor([args.target_idx], device=device, dtype=torch.long) if args.targeted else None

    if args.attack == "pgd":
        adv_input = pgd_attack(atk_model, input_tensor, label, args.epsilon, args.alpha, args.iters, args.targeted, target_label)
    else:
        adv_input = fgsm_attack(atk_model, input_tensor, label, args.epsilon, args.targeted, target_label)

    adv_idx, adv_conf, adv_probs = run_inference_fast(atk_model, adv_input)
    adv_name = categories[adv_idx] if categories and adv_idx < len(categories) else f"class_{adv_idx}"

    print(f"🎯 Original: {orig_name} ({orig_conf:.4f})")
    print(f"💥 Adversarial: {adv_name} ({adv_conf:.4f})")
    print(f"✅ Success: {orig_idx != adv_idx}")
    if orig_idx == adv_idx:
        print(f"⚠️  Warning: Attack failed. Defense may be working!")
    else:
        print(f"🔥 Success! Model was fooled.")

    metrics = compute_metrics(atk_model, input_tensor, adv_input, orig_idx, norm=args.norm)
    print(f"[Metrics] success={metrics['success']} | norm={metrics['norm']} | perturbation={metrics['perturbation_norm']:.4f} | Δconf(true)={metrics['confidence_shift']:+.4f}")
    print_topk(f"{args.model} (adversarial, {args.attack})", adv_probs, args.topk, categories)

    visualize(
        original_img=img,
        adv_tensor=adv_input,
        orig_label=f"{orig_idx} ({orig_name})",
        adv_label=f"{adv_idx} ({adv_name})",
        save_path="fgsm_result.png",
    )


if __name__ == "__main__":
    main()