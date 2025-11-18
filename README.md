# Handwritten Character Recognition with XGBoost & HOG

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Library-XGBoost_v2.0+-green)
![Platform](https://img.shields.io/badge/Platform-Linux_(AMD_Ryzen)-orange?logo=linux&logoColor=white)
![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

## 📖 Overview

This project addresses the problem of recognizing handwritten characters (A-Z, a-z, 0-9) within the scope of **Mechatronics Engineering** studies. The primary objective is to test the limits and performance of **Classical Machine Learning (XGBoost)** methods as an alternative to computationally expensive Deep Learning (CNN) approaches for embedded systems.

The project utilizes **Histogram of Oriented Gradients (HOG)** for feature extraction from raw pixel data and **eXtreme Gradient Boosting (XGBoost)** for multi-class classification.

Model performance was analyzed through 4 controlled **Ablation Studies**, examining factors such as data cleanliness, model capacity, and noise robustness.

## ⚙️ Key Features

- **Feature Engineering:** Extraction of structural features from $64 \times 64$ pixel images using the HOG method.
- **Algorithm:** Optimized XGBoost for Multi-class Classification.
- **Hardware Optimization:** Multi-core parallel training optimization for AMD Ryzen processors.
- **Live Monitoring:** Real-time tracking of training progress and error rates (merror) via terminal.

## 📂 Directory Structure

```text
Handwritten-Character-Recognition-XGBoost/
│
├── data/                        # Dataset directory
│   ├── english.csv              # Label file
│   └── Img/                     # Raw image folder
│
├── src/                         # Source Code
│   ├── exp1_baseline.py         # Experiment 1: Baseline Model
│   ├── exp2_filtered_final.py   # Experiment 2 (Final): Best Performing Model
│   ├── exp3_validation.py       # Experiment 3: Validation Split
│   └── exp4_noise_test.py       # Experiment 4: Noise Robustness Test
│
├── paper/                       # Academic Report
│   └── report.pdf               # Detailed technical paper (PDF)
│
├── models/                      # Trained Models (.joblib format)
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## 🚀 Installation

Follow these steps to set up the project on a local Linux machine. Tested on Python 3.12.

**1. Clone the Repository**

```bash
git clone [https://github.com/cankecilioglu/Handwritten-Character-Recognition-XGBoost.git](https://github.com/cankecilioglu/Handwritten-Character-Recognition-XGBoost.git)
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

Note: Ensure english.csv and the Img folder are placed inside the data/ directory before running the scripts.

## ▶️ Usage

To train the final (best performing) model, run the following commands:

```bash
# Activate virtual environment
source venv/bin/activate

# Run the final training script
python src/exp2_filtered_v3.py
```

## 📄 Academic Paper

You can review the detailed technical analysis, mathematical models, and literature review in the [Technical Report](paper/report_ing.pdf) located in the paper/ folder.

## 👨‍💻 Author

**Can Keçilioğlu** Department of Mechatronics Engineering
