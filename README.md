# 🛡️ VulnerAI

**Adversarial Attack Playground for AI Vision Models**

VulnerAI lets you explore how small perturbations can fool powerful computer vision models.
It provides an interactive Gradio UI to run adversarial attacks (FGSM, PGD), apply defenses, and benchmark model robustness.

---

## 🎥 Quick Demo

![VulnerAI Demo](docs/demo.gif) <sub>*Run FGSM on ResNet50 with epsilon=0.03 → model prediction flips instantly!*</sub>

*(Tip: You can record the GIF using [ScreenToGif](https://www.screentogif.com/) or `asciinema` + `agg` for terminal demos. Save it in `docs/demo.gif` and it will auto-show here.)*

---

## 🚀 Features

* Upload any image and generate adversarial examples.
* Choose between **ResNet50, VGG16, DenseNet121** models.
* Supports **FGSM** and **PGD** attacks with configurable epsilon.
* Apply defenses: Gaussian Blur, JPEG Compression, Feature Squeezing.
* Side-by-side visualization of original vs adversarial image.
* Benchmark multiple models → ranks most vulnerable vs most robust.
* Logs & metrics: perturbation norm, success rate, confidence drop.
* Batch mode (CSV export) for multiple images.

---

## 📦 Installation

```bash
git clone https://github.com/ssmadhavan006/Vulner_AI.git
cd Vulner_AI

# Create environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies**

* Python 3.9+
* PyTorch & torchvision
* Gradio
* Pillow, Matplotlib, Pandas

---

## 🖥️ Usage

### 1. Launch the UI

```bash
python model_ui.py
```

This opens a local Gradio app in your browser.

* **Run Attack Tab** → Upload image, select model, attack type, epsilon, defense.
* **Model Benchmark Tab** → Upload image, benchmark across all models.

### 2. Run from CLI

You can also run the attack engine directly:

```bash
python phase1_fgsm.py --image path/to/image.jpg --model resnet50 --attack fgsm --epsilon 0.03 --defense none
```

---

## 📂 Project Structure

```
Vulner_AI/
│── app_ui.py          # Gradio interface
│── model.py           # Core attack logic (FGSM, PGD, metrics, defenses)
│── images             # input images
│── adv_outputs        # Generated adversarial images, CSV logs
│── docs/demo.gif      # Demo recording (add this)
```

---

## 📊 Example Results

| Model       | Attack | ε    | Defense | Success | Perturbation | Confidence Drop |
| ----------- | ------ | ---- | ------- | ------- | ------------ | --------------- |
| ResNet50    | FGSM   | 0.03 | None    | ✅       | 0.0300       | -0.4512         |
| DenseNet121 | PGD    | 0.05 | JPEG    | ❌       | 0.0247       | -0.1234         |

---

## ⚖️ License

MIT License.
For research and educational use only.

---

## 🙌 Contributing

Pull requests and discussions are welcome!
Feel free to add new models, defenses, or attack methods.

---

