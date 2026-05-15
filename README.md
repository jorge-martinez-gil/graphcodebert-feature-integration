<div align="center">

# Improving Source Code Similarity Detection
### with GraphCodeBERT and Additional Feature Integration

*A novel approach that pushes code clone detection to near-perfect accuracy*

[![arXiv](https://img.shields.io/badge/arXiv-2408.08903-b31b1b.svg?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2408.08903)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorge-martinez-gil/colab-notebooks/blob/main/GraphCodeBERT%2BFeatures.ipynb)
[![Citations](https://img.shields.io/badge/Citations-4-blue?style=for-the-badge&logo=google-scholar)](https://scholar.google.com/citations?view_op=view_citation&hl=en&citation_for_view=X1pRUYcAAAAJ:DOLguN9Lh8sC)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

---

## 📖 Overview

Accurate detection of similar source code fragments is a cornerstone of software quality assurance — enabling plagiarism detection, code deduplication, and clone management at scale. This repository presents an **extended variant of GraphCodeBERT** that integrates an additional output feature into its classification head, achieving near-perfect F-measure scores on a standard benchmark.

> **Key Result:** Our GraphCodeBERT variant reaches **F-Measure = 0.99** on the IR-Plag dataset, outperforming all baselines including vanilla GraphCodeBERT (0.96).

---

## ✨ Highlights

| Feature | Detail |
|---|---|
| 🧠 **Base Model** | GraphCodeBERT (transformer pre-trained on code) |
| 🔧 **Innovation** | Additional output feature integrated into the classifier |
| 📊 **Dataset** | IR-Plag — academic plagiarism benchmark |
| 📈 **Best F-Measure** | **0.99** (Precision: 0.98 · Recall: 1.00) |
| 💻 **Languages** | Python · Jupyter Notebook |
| 📄 **Paper** | [arXiv:2408.08903](https://arxiv.org/abs/2408.08903) |

---

## 🗂️ Repository Structure

```
graphcodebert-feature-integration/
├── 📓 graphcodebert_fint.ipynb                          # Full end-to-end implementation (notebook)
├── 🐍 fine-tunning-graphcodebert-karnalim-with-features.py  # Standalone Python script
├── 📦 requirements.txt                                  # Python dependencies
├── 📁 data/                                             # Dataset directory
└── 🛠️ utils/                                            # Utility functions
```

---

## 🏗️ Methodology

### Model Architecture

The model extends **GraphCodeBERT** — a transformer pre-trained on code corpora that captures both the **textual semantics** and **structural (data-flow graph) properties** of source code. Our key contribution is the integration of an **additional output feature** into the binary classification head, enriching the representation used for similarity judgement.


### Dataset

We use the **IR-Plag dataset**, a widely-used benchmark for source code similarity detection in academic plagiarism contexts. It covers a range of similarity levels across Java source files, making it ideal for stress-testing clone detection models.

### Training & Evaluation

- Random train / validation / test splits
- Evaluated on **Precision**, **Recall**, and **F-Measure**
- Compared against CodeBERT, Output Analysis, XGBoost, Random Forest, and vanilla GraphCodeBERT

---

## 📊 Results

Our approach establishes a new state-of-the-art on the IR-Plag benchmark:

| Approach | Precision | Recall | F-Measure |
|---|:---:|:---:|:---:|
| CodeBERT | 0.72 | 1.00 | 0.84 |
| Output Analysis | 0.88 | 0.93 | 0.90 |
| Boosting (XGBoost) | 0.88 | 0.99 | 0.93 |
| Bagging (Random Forest) | 0.95 | 0.97 | 0.96 |
| GraphCodeBERT | 0.98 | 0.95 | 0.96 |
| 🏆 **Our GraphCodeBERT variant** | **0.98** | **1.00** | **0.99** |

---

## 🚀 Getting Started

### Prerequisites

```bash
git clone https://github.com/jorge-martinez-gil/graphcodebert-feature-integration.git
cd graphcodebert-feature-integration
pip install -r requirements.txt
```

### Run via Jupyter Notebook

Open `graphcodebert_fint.ipynb` in JupyterLab / VS Code, or launch it directly on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorge-martinez-gil/colab-notebooks/blob/main/GraphCodeBERT%2BFeatures.ipynb)

### Run via Python Script

```bash
python fine-tunning-graphcodebert-karnalim-with-features.py
```

---

## 🔬 Works That Cite This Paper

<details>
<summary><strong>View citing works (4)</strong></summary>

1. **[SWE-Fixer: Training Open-Source LLMs for Effective and Efficient GitHub Issue Resolution](https://arxiv.org/pdf/2501.05040)**
   - **Authors:** C. Xie, B. Li, C. Gao, H. Du, W. Lam, D. Zou
   - **Venue:** *arXiv preprint, 2025*
   - Large Language Models (LLMs) have shown exceptional proficiency in various complex tasks. This study explores the application of open-source LLMs in addressing software engineering challenges on GitHub.

2. **[Natural Language Summarization Enables Multi-Repository Bug Localization by LLMs in Microservice Architectures](https://arxiv.org/abs/)**
   - **Authors:** A. R. Oskooei, S. S. Yukcu, M. C. Bozoglan, et al.
   - **Venue:** *arXiv preprint, 2025*
   - Examines how natural language summarization can support LLM-based bug localization across multiple repositories in microservice systems.

</details>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made by [Jorge Martinez-Gil](https://github.com/jorge-martinez-gil)

⭐ *If you find this work useful, please consider giving it a star!*

</div>
