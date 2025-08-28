import os
import re
import sys
import tempfile
import subprocess
from typing import Tuple, Dict, Any, Optional

from PIL import Image
import gradio as gr


# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(SCRIPT_DIR, "phase1_fgsm.py")
CWD = SCRIPT_DIR  # Run subprocess here so imagenet_classes.txt and outputs land in project root
OUTPUT_IMAGE_NAME = "fgsm_result.png"  # produced by phase1_fgsm.py


# -------------------------
# Helpers: CLI mappings
# -------------------------
def _label_to_arg_defense(label: str) -> str:
    mapping = {
        "None": "none",
        "Gaussian Blur": "blur",
        "JPEG Compression": "jpeg",
        "All": "all",
    }
    return mapping.get(label, "none")


def _model_to_arg(name: str) -> str:
    return name.replace(" ", "").lower()  # ResNet50 -> resnet50


def _attack_to_arg(name: str) -> str:
    return name.lower()  # FGSM/PGD


# -------------------------
# Helpers: imaging
# -------------------------
def _concat_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    # Make same height, keep aspect ratio
    h = max(left.height, right.height)
    def resize_to_h(img: Image.Image, target_h: int) -> Image.Image:
        w = int(img.width * (target_h / img.height))
        return img.resize((w, target_h))
    L = resize_to_h(left, h)
    R = resize_to_h(right, h)
    out = Image.new("RGB", (L.width + R.width, h), (0, 0, 0))
    out.paste(L, (0, 0))
    out.paste(R, (L.width, 0))
    return out


def _crop_adv_from_figure(fig_path: str) -> Optional[Image.Image]:
    """
    Best-effort: phase1_fgsm.py saves a composite figure (typically 1x2).
    Crop the right half for the adversarial image. If cropping fails, return None.
    """
    try:
        with Image.open(fig_path) as fig:
            w, h = fig.size
            # right half, full height
            crop_box = (w // 2, 0, w, h)
            adv_img = fig.crop(crop_box).convert("RGB")
            return adv_img
    except Exception:
        return None


# -------------------------
# Parsing phase1_fgsm.py stdout
# -------------------------
def _parse_stdout(stdout: str) -> Dict[str, Any]:
    """
    Parse lines like:
      Original: cat (0.9321)
      Adversarial: dog (0.8743)
      [Metrics] success=1 | norm=linf | perturbation=0.0300 | dconf_true=-0.4512
    """
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

    orig_re = re.compile(r"Original:\s*(.+?)\s*\(([\d.]+)\)")
    adv_re = re.compile(r"Adversarial:\s*(.+?)\s*\(([\d.]+)\)")
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


# -------------------------
# Subprocess runner
# -------------------------
def _run_phase1(
    image_path: str,
    model: str,
    attack: str,
    epsilon: float,
    defense_label: str,
) -> Tuple[Optional[str], str]:
    """
    Runs phase1_fgsm.py and returns (output_image_path, logs).
    output_image_path is None if run failed.
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
    # Minimal PGD defaults to keep UI simple
    if attack_arg == "pgd":
        cmd += ["--alpha", "0.01", "--iters", "20"]

    try:
        proc = subprocess.run(
            cmd,
            cwd=CWD,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return None, "Error: phase1_fgsm.py timed out."
    except Exception as e:
        return None, f"Error launching phase1_fgsm.py: {e}"

    logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        return None, f"phase1_fgsm.py exited with code {proc.returncode}\n{logs}"

    out_path = os.path.join(CWD, OUTPUT_IMAGE_NAME)
    if not os.path.exists(out_path):
        return None, f"Expected output image not found: {out_path}\n{logs}"

    return out_path, logs


# -------------------------
# Gradio callbacks
# -------------------------
def predict_single(
    pil_image: Image.Image,
    model: str,
    attack: str,
    epsilon: float,
    defense: str,
):
    """
    Returns: (composite_side_by_side, summary_html, logs_text, status_text, download_path)
    Uses yield for smooth loading feedback.
    """
    if pil_image is None:
        yield None, "<div class='summary-card'><h3>Attack Summary</h3><p class='bad'>Please upload an image.</p></div>", "", "❌ Upload an image first.", None

    # Safer temp handling for input
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_in = f.name
        pil_image.save(tmp_in)

    # Step 1: Running attack
    yield None, "<div class='summary-card'><h3>Attack Summary</h3><p>🔄 Running attack...</p></div>", "", "🔄 Running attack...", None

    try:
        # Step 2: Run the actual attack
        out_path, logs = _run_phase1(tmp_in, model, attack, epsilon, defense)

        if out_path is None:
            yield None, "<div class='summary-card'><h3>Attack Summary</h3><p class='bad'>❌ Attack failed. See logs.</p></div>", logs, "❌ Attack failed.", None
            return

        # Step 3: Processing result
        yield None, "<div class='summary-card'><h3>Attack Summary</h3><p>🎨 Processing adversarial image...</p></div>", logs, "🎨 Processing results...", None

        adv_only = _crop_adv_from_figure(out_path) or Image.open(out_path).convert("RGB")
        composite = _concat_side_by_side(pil_image.convert("RGB"), adv_only)

        # Save downloadable adversarial-only image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fout:
            tmp_out = fout.name
        adv_only.save(tmp_out)

        # Step 4: Final result
        info = _parse_stdout(logs)
        success = info.get("success")
        success_emoji = "✅" if success else "❌"
        success_class = "ok" if success else "bad"

        summary_html = f"""
        <div class="summary-card">
          <h3>Attack Summary</h3>
          <ul>
            <li>Model: <strong>{model}</strong></li>
            <li>Attack: <strong>{attack}</strong> | ε=<strong>{epsilon:.3f}</strong> | Defense: <strong>{defense}</strong></li>
            <li>Original: <strong>{info.get('orig_name', '?')}</strong> ({info.get('orig_conf', '?')})</li>
            <li>Adversarial: <strong>{info.get('adv_name', '?')}</strong> ({info.get('adv_conf', '?')})</li>
            <li class="{success_class}">Success: <strong>{success_emoji}</strong> | Perturbation: <strong>{info.get('perturbation', '?')}</strong> (linf) | Confidence drop=<strong>{info.get('delta_conf_true', '?')}</strong></li>
          </ul>
        </div>
        """

        yield composite, summary_html, logs, "✅ Attack completed!", tmp_out
    finally:
        # Clean up input temp file
        try:
            os.remove(tmp_in)
        except OSError:
            pass

def benchmark_models(pil_image: Image.Image):
    """
    Runs attack on all 3 models and returns vulnerability ranking artifacts.
    Outputs (in order):
    - bench_rank_md: Markdown title for ranking
    - bench_table: Dataframe with rank, model, success, confidence drop, vulnerability score, perturbation
    - bench_winner_md: Markdown for the most robust model
    - bench_summary_html: HTML summary card with most vulnerable/robust and tips
    - bench_logs: Text logs (progress + raw logs)
    """
    if pil_image is None:
        yield "### 🔍 Vulnerability Ranking", [], "**No image provided**", "<div class='summary-card'><h3>📊 Robustness Analysis</h3><p>Please upload an image.</p></div>", "Please upload an image."
        return

    models = ["ResNet50", "VGG16", "DenseNet121"]
    tmp_in = os.path.join(tempfile.gettempdir(), f"in_{next(tempfile._get_candidate_names())}.png")
    pil_image.save(tmp_in)

    results = []
    logs_accum = []

    # Initial UI update
    yield "### 🔍 Vulnerability Ranking", [], "", "<div class='summary-card'><h3>📊 Robustness Analysis</h3><p>Running benchmark...</p></div>", "⏳ Starting benchmark..."

    for i, m in enumerate(models):
        out_path, logs = _run_phase1(tmp_in, m, "pgd", 0.05, "none")
        logs_accum.append(f"--- {m} ---\n{logs if logs else ''}")
        info = _parse_stdout(logs or "")

        success = info.get("success")
        drop = info.get("delta_conf_true")
        pert = info.get("perturbation")

        # Error handling: if run failed or metrics missing, mark as failed but keep ranking going
        failed_run = (out_path is None)
        if failed_run or success is None or drop is None:
            success = False if success is None else success
            drop = 0.0 if drop is None else drop
            status_str = "Failed" if failed_run else ("✅" if success else "❌")
            logs_accum.append("⚠️ Attack run failed or incomplete metrics; treating as failure for ranking.")
        else:
            status_str = "✅" if success else "❌"

        # Vulnerability score: (1.0 if success else 0.5) * abs(delta_conf_true)
        score = (1.0 if success else 0.5) * abs(drop if isinstance(drop, (int, float)) else 0.0)

        results.append({
            "model": m,
            "success": success,
            "status_str": status_str,
            "drop": drop,
            "pert": pert,
            "score": score
        })

        # Build partial ranking (descending by vulnerability score)
        sr = sorted(results, key=lambda r: (r["score"] if r["score"] is not None else -1), reverse=True)
        medals = ["🥇", "🥈", "🥉"]
        table_rows = []
        for idx, r in enumerate(sr):
            rank_icon = medals[idx] if idx < len(medals) else f"{idx+1}"
            table_rows.append({
                "Rank": rank_icon,
                "Model": r["model"],
                "Attack Success": r["status_str"],
                "Confidence Drop": f"{r['drop']:+.4f}" if isinstance(r["drop"], (int, float)) else "n/a",
                "Vulnerability Score": f"{r['score']:.3f}" if isinstance(r["score"], (int, float)) else "n/a",
                "Perturbation": f"{r['pert']:.4f}" if isinstance(r["pert"], (int, float)) else "n/a"
            })

        progress_msg = f"✅ {m} completed ({i+1}/{len(models)})."
        yield "### 🔍 Vulnerability Ranking", table_rows, "", "<div class='summary-card'><h3>📊 Robustness Analysis</h3><p>📈 Updating ranking...</p></div>", "\n".join(logs_accum + [progress_msg])

    # Final ranking
    sr = sorted(results, key=lambda r: (r["score"] if r["score"] is not None else -1), reverse=True)
    medals = ["🥇", "🥈", "🥉"]
    final_rows = []
    for idx, r in enumerate(sr):
        rank_icon = medals[idx] if idx < len(medals) else f"{idx+1}"
        final_rows.append({
            "Rank": rank_icon,
            "Model": r["model"],
            "Attack Success": r["status_str"],
            "Confidence Drop": f"{r['drop']:+.4f}" if isinstance(r["drop"], (int, float)) else "n/a",
            "Vulnerability Score": f"{r['score']:.3f}" if isinstance(r["score"], (int, float)) else "n/a",
            "Perturbation": f"{r['pert']:.4f}" if isinstance(r["pert"], (int, float)) else "n/a"
        })

    most_vuln = sr[0]["model"] if sr else "n/a"
    most_robust = sr[-1]["model"] if sr else "n/a"
    winner_md = f"**🏆 Winner (Most Robust): {most_robust}**"
    summary_html = f"""
    <div class="summary-card">
      <h3>📊 Robustness Analysis</h3>
      <p><strong>Most Vulnerable:</strong> <span style="color:#ef4444">{most_vuln} 🥇</span></p>
      <p><strong>Most Robust:</strong> <span style="color:#22c55e">{most_robust} 🛡️</span></p>
      <p><em>Tips: Models with dense connectivity (DenseNet) often resist perturbations better.</em></p>
    </div>
    """
    final_logs = "\n".join(logs_accum + ["📊 Generating final ranking...", f"Most Vulnerable: {most_vuln}", f"Most Robust: {most_robust}"])

    yield "### 🔍 Vulnerability Ranking", final_rows, winner_md, summary_html, final_logs
    # ... existing code ...

# -------------------------
# Styling (dark, centered, subtle shadows)
# -------------------------
custom_css = """
:root, .gradio-container {
  --radius-xl: 14px;
}

body, .gradio-container {
  background: radial-gradient(1200px 600px at 50% 0%, #0f172a 0%, #0b1221 50%, #0a0f1a 100%);
  color: #e5e7eb;
}

.container-center {
  max-width: 1100px;
  margin: 0 auto;
}

.section {
  background: rgba(17, 24, 39, 0.7);
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 14px;
  padding: 16px;
  box-shadow:
    0 10px 30px rgba(0, 0, 0, 0.35),
    inset 0 1px 0 rgba(255, 255, 255, 0.02);
}

.summary-card {
  background: rgba(15, 23, 42, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 12px;
  padding: 14px 16px;
}

.summary-card h3 { margin: 4px 0 10px 0; }
.summary-card ul { margin: 0; padding-left: 16px; }
.summary-card li { margin: 6px 0; }

.ok { color: #22c55e; }
.bad { color: #ef4444; }

button, .gr-button {
  background: linear-gradient(180deg, #1f2937 0%, #111827 100%) !important;
  color: #e5e7eb !important;
  border: 1px solid rgba(148, 163, 184, 0.28) !important;
  border-radius: 10px !important;
  box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}

.gr-input, .gr-textbox, .gr-dropdown, .gr-slider, .gr-radio, .gr-image {
  background: rgba(17, 24, 39, 0.55) !important;
  color: #e5e7eb !important;
  border-color: rgba(148, 163, 184, 0.25) !important;
}

footer { display: none; }

/* Better spinner */
.gr-loading {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
}

.gr-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-top: 3px solid #60a5fa;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
"""


# -------------------------
# App
# -------------------------
with gr.Blocks(title="VulnerAI", css=custom_css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="gray")) as demo:
    with gr.Column(elem_classes=["container-center"]):
        gr.Markdown("### 🛡️ VulnerAI", elem_classes=["title"])
        gr.Markdown("Test how easily AI vision models can be fooled by tiny changes.", elem_classes=["subtitle"])

        with gr.Tabs():
            with gr.Tab("Run Attack"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["section"]):
                            inp_img = gr.Image(label="Upload Image", type="pil")
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["section"]):
                            model_dd = gr.Dropdown(label="Model", choices=["ResNet50", "VGG16", "DenseNet121"], value="ResNet50")
                            attack_rd = gr.Radio(label="Attack", choices=["FGSM", "PGD"], value="FGSM")
                            eps_sl = gr.Slider(label="Epsilon (L∞)", minimum=0.01, maximum=0.10, step=0.01, value=0.03)
                            defense_rd = gr.Radio(
                                ["None", "Gaussian Blur", "JPEG Compression", "All"],
                                value="None",
                                label="Defense"
                            )
                            run_btn = gr.Button("Launch Attack", variant="primary")

                with gr.Group(elem_classes=["section"]):
                    out_img = gr.Image(label="Original vs Adversarial", type="pil")
                    out_summary = gr.HTML(value="<div class='summary-card'><h3>Attack Summary</h3><p>Results will appear here...</p></div>")
                    status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
                    out_logs = gr.Textbox(label="Logs", value="", lines=12)
                    download_btn = gr.File(label="Download Adversarial Image")

                run_btn.click(
                    predict_single,
                    inputs=[inp_img, model_dd, attack_rd, eps_sl, defense_rd],
                    outputs=[out_img, out_summary, out_logs, status_text, download_btn],  # 5 outputs (incl. downloadable file)
                    show_progress=True
                )

            # ✅ Benchmark Tab
            with gr.Tab("Model Robustness Benchmark"):
                with gr.Row():
                    with gr.Column():
                        bench_img = gr.Image(label="Upload Image", type="pil")
                        bench_btn = gr.Button("Benchmark All Models", variant="primary")
                    with gr.Column():
                        bench_rank_md = gr.Markdown("### 🔍 Vulnerability Ranking")
                        bench_table = gr.Dataframe(headers=["Rank", "Model", "Attack Success", "Confidence Drop", "Vulnerability Score", "Perturbation"], value=[])
                        bench_winner_md = gr.Markdown("")
                        bench_summary_html = gr.HTML("<div class='summary-card'><h3>📊 Robustness Analysis</h3><p>Results will appear here...</p></div>")
                        bench_logs = gr.Textbox(label="Logs", value="", lines=12)

                bench_btn.click(
                    benchmark_models,
                    inputs=[bench_img],
                    outputs=[bench_rank_md, bench_table, bench_winner_md, bench_summary_html, bench_logs],
                    show_progress=True
                )

if __name__ == "__main__":
    demo.launch()
