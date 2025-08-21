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

    # Accept lines with/without emojis, only matching "Original:" / "Adversarial:"
    orig_re = re.compile(r"Original:\s*(.+?)\s*\(([\d.]+)\)")
    adv_re = re.compile(r"Adversarial:\s*(.+?)\s*\(([\d.]+)\)")

    # Accept both "Δconf(true)" and "dconf_true"; allow scientific notation
    metrics_re = re.compile(
        r"\[Metrics\]\s*success=(\d+)\s*\|\s*norm=([^\|]+)\s*\|\s*perturbation=([0-9.eE+\-]+)\s*\|\s*(?:Δconf\(true\)|dconf_true)=([+\-]?[0-9.eE]+)"
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
            # Force headless plotting and UTF-8 stdio to avoid Unicode errors on Windows
            env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        return None, "Error: phase1_fgsm.py timed out."
    except Exception as e:
        return None, f"Error launching phase1_fgsm.py: {e}"

    logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        return None, f"phase1_fgsm.py exited with code {proc.returncode}\n{logs}"

    # Use the figure created by your script; then save a copy as adv_output.png
    combined_path = os.path.join(CWD, "fgsm_result.png")
    if not os.path.exists(combined_path):
        return None, f"Expected output image not found: {combined_path}\n{logs}"

    adv_output = os.path.join(CWD, "adv_output.png")
    try:
        with Image.open(combined_path) as im:
            im.convert("RGB").save(adv_output)
    except Exception as e:
        return None, f"Failed to save adv_output.png: {e}\n{logs}"

    return adv_output, logs


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
        # Return no image, a user-friendly summary, and the raw logs
        return None, "<div class='summary-card'><h3>Attack Summary</h3><p class='bad'>Attack failed to run. See logs below.</p></div>", logs

    try:
        adv_img = Image.open(adv_path).convert("RGB")
    except Exception as e:
        return None, f"<div class='summary-card'><h3>Attack Summary</h3><p class='bad'>Failed to open adversarial image: {e}</p></div>", logs

    info = _parse_stdout(logs)
    success = info.get("success")
    success_emoji = "✅" if success else "❌" if success is not None else "❓"
    success_class = "ok" if success else "bad" if success is not None else ""
    # Build dark styled summary card (HTML)
    summary_html = f"""
    <div class="summary-card">
      <h3>Attack Summary</h3>
      <ul>
        <li>Model: <strong>{model}</strong></li>
        <li>Attack: <strong>{attack}</strong> | ε=<strong>{epsilon:.3f}</strong> | Defense: <strong>{defense}</strong></li>
        <li>Original: <strong>{info.get('orig_name', '?')}</strong> ({info.get('orig_conf', '?')})</li>
        <li>Adversarial: <strong>{info.get('adv_name', '?')}</strong> ({info.get('adv_conf', '?')})</li>
        <li class="{success_class}">Success: <strong>{success_emoji}</strong> | Perturbation: <strong>{info.get('perturbation', '?')}</strong> ({info.get('norm', '?')}) | Δconf(true)=<strong>{info.get('delta_conf_true', '?')}</strong></li>
      </ul>
    </div>
    """

    return adv_img, summary_html, logs


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
    rows = []
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
        })

    df = pd.DataFrame(rows, columns=["Model", "Success?", "Confidence Drop", "Perturbation"])
    return df, "\n\n".join(all_logs)


custom_css = """
:root, .gradio-container { --radius-xl: 14px; }
.gradio-container, body { background: #0b0f17; color: #e5e7eb; }
.section { background: #0e1525; border: 1px solid #1f2937; border-radius: 12px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
.summary-card { background: #111827; border: 1px solid #374151; border-radius: 12px; padding: 14px 16px; }
.summary-card h3 { margin: 4px 0 10px 0; }
.summary-card ul { margin: 0; padding-left: 16px; }
.summary-card li { margin: 6px 0; }
.ok { color: #22c55e; } /* green */
.bad { color: #ef4444; } /* red */
"""

with gr.Blocks(title="Adversarial Attack Lab", css=custom_css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="gray")) as demo:
    gr.Markdown("### 🛡️ Adversarial Attack Lab")
    gr.Markdown("_Test how easily AI vision models can be fooled by tiny changes._")

    with gr.Tab("Run Attack"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["section"]):
                    inp_img = gr.Image(label="Upload Image", type="pil")
                    clear_btn = gr.Button("Clear")
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["section"]):
                    model_dd = gr.Dropdown(label="Model", choices=["ResNet50", "VGG16", "DenseNet121"], value="ResNet50")
                    attack_rd = gr.Radio(label="Attack", choices=["FGSM", "PGD"], value="FGSM")
                    eps_sl = gr.Slider(label="Epsilon (L∞)", minimum=0.01, maximum=0.10, step=0.01, value=0.03)
                    defense_rd = gr.Radio(label="Defense", choices=["None", "Gaussian Blur", "JPEG Compression", "Color Depth Reduction", "All"], value="None")
                    with gr.Accordion("Advanced (PGD)", open=False):
                        alpha_sl = gr.Slider(label="PGD step size (alpha)", minimum=0.001, maximum=0.05, step=0.001, value=0.01)
                        iters_sl = gr.Slider(label="PGD iterations", minimum=1, maximum=100, step=1, value=20)
                    run_btn = gr.Button("Launch Attack", variant="primary")

                with gr.Group(elem_classes=["section"]):
                    out_img = gr.Image(label="Original (Left) vs Adversarial (Right)", type="pil")
                    out_summary = gr.HTML(value="<div class='summary-card'><h3>Attack Summary</h3><p>Results will appear here...</p></div>")
                    out_logs = gr.Textbox(label="Logs", value="", lines=10)

        run_btn.click(
            predict_single,
            inputs=[inp_img, model_dd, attack_rd, eps_sl, defense_rd, alpha_sl, iters_sl],
            outputs=[out_img, out_summary, out_logs],
        )

        def _clear():
            return None, "<div class='summary-card'><h3>Attack Summary</h3><p>Cleared.</p></div>", ""
        clear_btn.click(_clear, outputs=[inp_img, out_summary, out_logs])

    with gr.Tab("Model Robustness Benchmark"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["section"]):
                    bench_img = gr.Image(label="Upload Image", type="pil")
                    bench_attack = gr.Radio(label="Attack", choices=["FGSM", "PGD"], value="FGSM")
                    bench_eps = gr.Slider(label="Epsilon (L∞)", minimum=0.01, maximum=0.10, step=0.01, value=0.03)
                    bench_defense = gr.Radio(label="Defense", choices=["None", "Gaussian Blur", "JPEG Compression", "Color Depth Reduction", "All"], value="None")
                    with gr.Accordion("Advanced (PGD)", open=False):
                        bench_alpha = gr.Slider(label="PGD step size (alpha)", minimum=0.001, maximum=0.05, step=0.001, value=0.01)
                        bench_iters = gr.Slider(label="PGD iterations", minimum=1, maximum=100, step=1, value=20)
                    bench_btn = gr.Button("Benchmark All Models", variant="primary")
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["section"]):
                    bench_table = gr.Dataframe(headers=["Model", "Success?", "Confidence Drop", "Perturbation"])
                    bench_logs = gr.Textbox(label="Logs", value="", lines=12)

        bench_btn.click(
            benchmark_models,
            inputs=[bench_img, bench_attack, bench_eps, bench_defense, bench_alpha, bench_iters],
            outputs=[bench_table, bench_logs],
        )

if __name__ == "__main__":
    demo.launch()