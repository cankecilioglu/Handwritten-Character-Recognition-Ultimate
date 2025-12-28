# Handwritten Character Recognition (HCR) — Ultimate
### From Classical ML Baseline to Near State-of-the-Art Deep Learning System

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras&logoColor=white)
![Research](https://img.shields.io/badge/Project-Type%3A%20Research%20%26%20Engineering-success)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 🚀 Project Overview

This repository presents a **research-grade Handwritten Character Recognition (HCR)** system that evolved from a simple classical machine learning experiment into a **near State-of-the-Art (SOTA)** deep learning pipeline.

The project covers the **entire lifecycle** of a real-world ML system:
- Baseline construction
- Deep learning transition
- Hyperparameter optimization
- Ensemble intelligence
- Test-Time Augmentation (TTA)
- Rigorous error and failure-mode analysis

**Final Accuracy:** **93.16%**  
**Classes:** 62 (A–Z, a–z, 0–9)

---

## 🧠 Why This Project Is Different

Most HCR projects stop at “high accuracy”.

This project goes further by answering:
- **Why does the model fail?**
- **Which errors are mathematically unavoidable?**
- **How do real handwriting artifacts affect predictions?**
- **How far can performance be pushed with engineering discipline?**

---

## 📈 Project Evolution & Milestones

### Phase 1 — Classical Machine Learning (Baseline)

**Method**
- HOG (Histogram of Oriented Gradients)
- XGBoost Classifier

**Hardware**
- CPU: AMD Ryzen 5

**Accuracy**
- **68.11%**

**Key Lesson**
Manual feature engineering fails to capture the extreme variability of handwritten characters and reaches an early **performance plateau**, especially for visually similar letters.

---

### Phase 2 — Deep Learning & Optimization (The Ascent)

**Method**
- Deep Convolutional Neural Networks (CNN)

**Progression**
- Simple CNN → **78.74%** (NVIDIA T4)
- VGG-style CNN (single best model)

**Key Engineering Improvements**
- Swish activation (instead of ReLU)
- L2 regularization
- Dropout
- Cosine Decay learning rate scheduler

**Hyperparameter Optimization**
- **Bayesian Optimization** using **Keras Tuner**
- Scientifically selected learning rate (≈ **0.0008**)

**Hardware**
- Google Colab **L4 GPU**

**Best Single-Model Accuracy**
- **89.45%**

---

### Phase 3 — Peak Performance (Ensemble + TTA)

**Core Idea**
> No single model sees everything correctly — multiple perspectives outperform individual intelligence.

**Ensemble Components**
1. **Champion (Swish CNN)**  
   Excels at complex stroke patterns
2. **Stabilizer (ReLU CNN)**  
   Produces balanced and conservative predictions
3. **Nuclear Model (Synthetic Data)**  
   Trained on computer fonts to learn ideal character geometry

**Test-Time Augmentation (TTA)**
- Each test image is evaluated under multiple geometric variations
- Final prediction = averaged probabilities

**Hardware**
- NVIDIA **A100 GPU**

### 🏆 FINAL RESULT
**93.16% Accuracy**

---

## 🔬 Engineering Insights & Failure Analysis

### 1️⃣ Uppercase vs Lowercase Blindness

Confusion matrix analysis reveals:
- Many characters achieve **F1-Score ≈ 1.0**
- Severe drops for ambiguous pairs:
  - `w` vs `W`
  - `s` vs `S`
  - `o` vs `O`

**Conclusion**
Single-character recognition **lacks contextual information**, making these errors **mathematically unavoidable**, regardless of model strength.

---

### 2️⃣ Real-World Handwriting Artifacts

Testing with real handwritten samples revealed:
- Thin pen strokes collapse into pixel noise during **64×64 downscaling**
- Pixel artifacts can mimic zig-zag features
- Model falsely interprets noise as structural edges (e.g., “W”)

**Observed Phenomenon**
> Feature Confusion caused by resolution-induced aliasing

---

## 🧱 Repository Structure

```text
Handwritten-Character-Recognition-Ultimate/
│
├── data/                # Dataset (may be external due to size)
├── src/                 # Training, tuning, evaluation pipelines
├── paper/               # Academic paper (LaTeX / PDF)
├── requirements.txt
└── README.md

```

## 🚀 Installation

Follow these steps to set up the project on a local Linux machine. Tested on Python 3.12.

**1. Clone the Repository**

```bash
git clone https://github.com/cankecilioglu/Handwritten-Character-Recognition-XGBoost.git
cd Handwritten-Character-Recognition-XGBoost
```

**2. Create Virtual Environment**

It is recommended to use a clean virtual environment to avoid dependency conflicts.

```bash
# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```
### 📂 Dataset Setup (Required)

The image dataset (`Img/` folder) is hosted on Kaggle due to GitHub's file size limits. You must download it manually to run the project.

1. **Download:** [👉 Click here to download the dataset from Kaggle](https://www.kaggle.com/datasets/cankeiliolu/handwritten-english-characters)

2. **Extract:** Unzip the downloaded file. Move the `english.csv` file and the `Img` folder into the `data/` directory of this project.
Note: Ensure english.csv and the Img folder are placed inside the data/ directory before running the scripts.

## ▶️ Usage

To train the final (best performing) model, run the following commands:

```bash
# Activate virtual environment
source venv/bin/activate

# Run the final training script
python src/deep_learning/train.py
```

## 📄 Academic Paper

You can review the detailed technical analysis, mathematical models, and literature review in the [Technical Report](paper/report_ing.pdf) located in the paper/ folder.

## 👨‍💻 Author

**Can Keçilioğlu** Department of Mechatronics Engineering
