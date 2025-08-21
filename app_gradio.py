import os
import re
import sys
import tempfile
import subprocess
from typing import Tuple, Dict, Any, Optional, List

from PIL import Image
import gradio as gr
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(SCRIPT_DIR, "phase1_fgsm.py")
CWD = SCRIPT_DIR  # Run subprocess here so imagenet_classes.txt and outputs land in project root
OUTPUT_IMAGE_NAME = "fgsm_result.png"  # produced by phase1_fgsm.py


def _label_to_arg_defense(label: str) -> str:
    mapping = {
        "None": "none",
        "Gaussian Blur": "blur",
        "JPEG Compression": "jpeg",
        "Color Depth Reduction": "color",
        "All": "all",
    }
    return mapping.get(label, "none")


def _model_to_arg(name: str) -> str:
    return name.replace(" ", "").lower()  # ResNet50 -> resnet50


def _attack_to_arg(name: str) -> str:
    return name.lower()  # FGSM/PGD


def _concat_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    # Make same height, keep aspect ratio
    h = max(left.height, right.height)
    def resize_to_h(img: Image.Image, target_h: int) -> Image.Image:
        w = int(img.width * (target_h / img.height))
        return img.resize((w, target_h))
    L = resize_to_h(left, h)
    R = resize_to_h(right, h)
    out = Image.new("RGB", (L.width + R.width, h), (255, 255, 255))
    out.paste(L, (0, 0))
    out.paste(R, (L.width, 0))
    return out


def _parse_stdout(stdout: str) -> Dict[str, Any]:
    # Defaults
    info = {
        "orig_name": None,
        "orig_conf": None,
        "adv_name": None,
        "adv_conf": None,
        "success": None,
        "perturbation": None,
        "norm": None,
        "delta_conf_true": None,
    }

    # Examples printed by your script:
    # 🎯 Original: {orig_name} ({orig_conf:.4f})
    # 💥 Adversarial: {adv_name} ({adv_conf:.4f})
    # [Metrics] success=1 | norm=linf | perturbation=0.0123 | Δconf(true)=-0.4567
    orig_re = re.compile(r"Original:\s*(.+?)\s*\(([\d.]+)\)")
    adv_re = re.compile(r"Adversarial:\s*(.+?)\s*\(([\d.]+)\)")
    metrics_re = re.compile(
        r"\[Metrics\]\s*success=(\d+)\s*\|\s*norm=([^\|]+)\s*\|\s*perturbation=([\d.]+)\s*\|\s*Δconf\(true\)=([+\-]?[\d.]+)"
    )

    mo = orig_re.search(stdout)
    if mo:
        info["orig_name"] = mo.group(1).strip()
        info["orig_conf"] = float(mo.group(2))
    mo = adv_re.search(stdout)
    if mo:
        info["adv_name"] = mo.group(1).strip()
        info["adv_conf"] = float(mo.group(2))
    mo = metrics_re.search(stdout)
    if mo:
        info["success"] = (mo.group(1).strip() == "1")
        info["norm"] = mo.group(2).strip()
        info["perturbation"] = float(mo.group(3))
        info["delta_conf_true"] = float(mo.group(4))

    return info


def _run_phase1(
    image_path: str,
    model: str,
    attack: str,
    epsilon: float,
    defense_label: str,
    alpha: float,
    iters: int,
) -> Tuple[Optional[str], str]:
    """
    Runs phase1_fgsm.py and returns (adv_image_path, logs).
    adv_image_path is None if run failed.
    """
    defense_arg = _label_to_arg_defense(defense_label)
    model_arg = _model_to_arg(model)
    attack_arg = _attack_to_arg(attack)

    cmd = [
        sys.executable,
        SCRIPT_PATH,
        "--image", image_path,
        "--model", model_arg,
        "--attack", attack_arg,
        "--epsilon", str(epsilon),
        "--defense", defense_arg,
        "--topk", "5",
        "--norm", "linf",
    ]
    if attack_arg == "pgd":
        cmd += ["--alpha", str(alpha), "--iters", str(iters)]

    try:
        proc = subprocess.run(
            cmd,
            cwd=CWD,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired:
        return None, "Error: phase1_fgsm.py timed out."
    except Exception as e:
        return None, f"Error launching phase1_fgsm.py: {e}"

    logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        return None, f"phase1_fgsm.py exited with code {proc.returncode}\n{logs}"

    # phase1_fgsm.py saves 'fgsm_result.png' in CWD
    out_path = os.path.join(CWD, OUTPUT_IMAGE_NAME)
    if not os.path.exists(out_path):
        # In case file name changes or not produced
        return None, f"Expected output image not found: {out_path}\n{logs}"

    # Copy to a temp path to avoid overwriting when multiple requests come in
    with Image.open(out_path) as adv_img:
        tmp_out = os.path.join(tempfile.gettempdir(), f"adv_{next(tempfile._get_candidate_names())}.png")
        adv_img.save(tmp_out)

    return tmp_out, logs


def predict_single(
    pil_image: Image.Image,
    model: str,
    attack: str,
    epsilon: float,
    defense: str,
    alpha: float,
    iters: int,
) -> Tuple[Optional[Image.Image], str, str]:
    """
    Returns: (composite_image, summary_text, logs_text)
    """
    if pil_image is None:
        return None, "Please upload an image.", ""

    # Save input to temp file
    tmp_in = os.path.join(tempfile.gettempdir(), f"in_{next(tempfile._get_candidate_names())}.png")
    pil_image.save(tmp_in)

    adv_path, logs = _run_phase1(tmp_in, model, attack, epsilon, defense, alpha, iters)
    if adv_path is None:
        return None, "Attack failed to run. See logs for details.", logs

    # Load the script-produced side-by-side figure directly
    try:
        adv_img = Image.open(adv_path).convert("RGB")
    except Exception as e:
        return None, f"Failed to open adversarial image: {e}", logs

    info = _parse_stdout(logs)
    success_emoji = "✅" if info.get("success") else "❌" if info.get("success") is not None else "❓"
    summary = (
        f"Model: {model} | Attack: {attack} | ε={epsilon:.3f} | Defense: {defense}\n"
        f"Original: {info.get('orig_name', '?')} ({info.get('orig_conf', '?')})\n"
        f"Adversarial: {info.get('adv_name', '?')} ({info.get('adv_conf', '?')})\n"
        f"Success: {success_emoji} | Perturbation: {info.get('perturbation', '?')} ({info.get('norm', '?')}) | "
        f"Δconf(true)={info.get('delta_conf_true', '?')}"
    )

    return adv_img, summary, logs


def benchmark_models(
    pil_image: Image.Image,
    attack: str,
    epsilon: float,
    defense: str,
    alpha: float,
    iters: int,
) -> Tuple[pd.DataFrame, str]:
    """
    Runs the same attack across ResNet50, VGG16, DenseNet121, returns a dataframe and combined logs.
    """
    if pil_image is None:
        return pd.DataFrame(columns=["Model", "Success?", "Confidence Drop", "Perturbation", "Norm"]), "Please upload an image."

    models = ["ResNet50", "VGG16", "DenseNet121"]
    rows: List[Dict[str, Any]] = []
    all_logs = []

    tmp_in = os.path.join(tempfile.gettempdir(), f"in_{next(tempfile._get_candidate_names())}.png")
    pil_image.save(tmp_in)

    for m in models:
        adv_path, logs = _run_phase1(tmp_in, m, attack, epsilon, defense, alpha, iters)
        all_logs.append(f"--- {m} ---\n{logs}")

        info = _parse_stdout(logs)
        rows.append({
            "Model": m,
            "Success?": "✅" if info.get("success") else "❌",
            "Confidence Drop": info.get("delta_conf_true"),
            "Perturbation": info.get("perturbation"),
            "Norm": info.get("norm"),
        })

    df = pd.DataFrame(rows, columns=["Model", "Success?", "Confidence Drop", "Perturbation", "Norm"])
    return df, "\n\n".join(all_logs)


with gr.Blocks(title="Adversarial Attack Lab") as demo:
    gr.Markdown("## 🛡️ Adversarial Attack Lab")
    gr.Markdown(
        "_Test how easily AI vision models can be fooled by tiny changes. Upload an image, choose a model and attack, and see if defenses help._"
    )

    with gr.Tab("Run Attack"):
        with gr.Row():
            with gr.Column(scale=1):
                inp_img = gr.Image(label="Upload Image", type="pil")
                model_dd = gr.Dropdown(
                    label="Model",
                    choices=["ResNet50", "VGG16", "DenseNet121"],
                    value="ResNet50",
                )
                attack_rd = gr.Radio(
                    label="Attack",
                    choices=["FGSM", "PGD"],
                    value="FGSM",
                )
                eps_sl = gr.Slider(
                    label="Epsilon (L∞)",
                    minimum=0.01,
                    maximum=0.10,
                    step=0.005,
                    value=0.03,
                )
                defense_rd = gr.Radio(
                    label="Defense",
                    choices=["None", "Gaussian Blur", "JPEG Compression", "Color Depth Reduction", "All"],
                    value="None",
                )
                with gr.Accordion("Advanced (PGD)", open=False):
                    alpha_sl = gr.Slider(label="PGD step size (alpha)", minimum=0.001, maximum=0.05, step=0.001, value=0.01)
                    iters_sl = gr.Slider(label="PGD iterations", minimum=1, maximum=100, step=1, value=20)

                run_btn = gr.Button("Launch Attack", variant="primary")

            with gr.Column(scale=1):
                out_img = gr.Image(label="Original (Left) vs Adversarial (Right)", type="pil")
                out_summary = gr.Textbox(label="Summary", value="Results will appear here...", lines=6)
                out_logs = gr.Textbox(label="Logs", value="", lines=10)

        run_btn.click(
            predict_single,
            inputs=[inp_img, model_dd, attack_rd, eps_sl, defense_rd, alpha_sl, iters_sl],
            outputs=[out_img, out_summary, out_logs],
        )

    with gr.Tab("Model Robustness Benchmark"):
        with gr.Row():
            with gr.Column(scale=1):
                bench_img = gr.Image(label="Upload Image", type="pil")
                bench_attack = gr.Radio(label="Attack", choices=["FGSM", "PGD"], value="FGSM")
                bench_eps = gr.Slider(label="Epsilon (L∞)", minimum=0.01, maximum=0.10, step=0.005, value=0.03)
                bench_defense = gr.Radio(
                    label="Defense",
                    choices=["None", "Gaussian Blur", "JPEG Compression", "Color Depth Reduction", "All"],
                    value="None",
                )
                with gr.Accordion("Advanced (PGD)", open=False):
                    bench_alpha = gr.Slider(label="PGD step size (alpha)", minimum=0.001, maximum=0.05, step=0.001, value=0.01)
                    bench_iters = gr.Slider(label="PGD iterations", minimum=1, maximum=100, step=1, value=20)
                bench_btn = gr.Button("Benchmark All Models", variant="primary")

            with gr.Column(scale=1):
                bench_table = gr.Dataframe(headers=["Model", "Success?", "Confidence Drop", "Perturbation", "Norm"])
                bench_logs = gr.Textbox(label="Logs", value="", lines=12)

        bench_btn.click(
            benchmark_models,
            inputs=[bench_img, bench_attack, bench_eps, bench_defense, bench_alpha, bench_iters],
            outputs=[bench_table, bench_logs],
        )

if __name__ == "__main__":
    # Launch Gradio
    demo.launch()