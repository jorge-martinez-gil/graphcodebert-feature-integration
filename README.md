# 🚀 Improving Source Code Similarity Detection with GraphCodeBERT and Additional Feature Integration

This repository contains the implementation of a novel approach for source code similarity detection that integrates an additional output feature into the classification process to enhance model performance. The approach is based on the **GraphCodeBERT** model, which has been extended with a custom output feature layer and a concatenation mechanism to improve feature representation. The model has been trained and evaluated on the **IR-Plag dataset**, demonstrating significant improvements in precision, recall, and f-measure. The full implementation, including model architecture, training strategies, and evaluation metrics, is available in this repository.

[![arXiv](https://img.shields.io/badge/arXiv-2408.08903-b31b1b.svg)](https://arxiv.org/abs/2408.08903)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌍 Introduction

Accurate and efficient detection of similar source code fragments is crucial for maintaining software quality, improving developer productivity, and ensuring code integrity. With the rise of deep learning (DL) and natural language processing (NLP) techniques, transformer-based models have become a preferred approach for understanding and processing source code.

In this project, we extend the capabilities of **GraphCodeBERT**—a transformer model specifically designed to process the structural and semantic properties of programming languages. By integrating an additional output feature layer and using a concatenation mechanism, our approach enhances the model's ability to represent source code, leading to better performance in similarity detection tasks.

### 📂 Repository Contents

- **`graphcodebert_fint.ipynb`**: A Jupyter Notebook that includes the full implementation of the model, from data loading and preprocessing to training, evaluation, and results interpretation. Detailed comments and documentation are provided within the notebook. **It is optimized to be used in Google Colab since the use of a GPU is highly recommended.**
- **`fine-tunning-graphcodebert-karnalim-with-features.py`**: The source code in the form of a standard Python app.

---

## 🛠️ Methodology

### 🔍 Model Architecture

The model is an extension of **GraphCodeBERT**, a transformer-based model pre-trained on large corpora of code and designed to capture both textual and structural properties of code. We introduce a custom output feature layer and concatenate the pooled output of the transformer with this processed feature, allowing the model to learn a richer representation of the source code.

### 📊 Dataset

We utilize the **IR-Plag dataset**, which is specifically designed for benchmarking source code similarity detection techniques, particularly in academic plagiarism contexts. The dataset contains 467 code files, with 355 labeled as plagiarized. The diversity in coding styles and structures within this dataset makes it ideal for evaluating the effectiveness of our model.

### 🏋️ Training and Evaluation

The training process included random splits of the dataset into training, validation, and test sets. Key metrics such as precision, recall, and f-measure were computed to evaluate the model's performance. The notebook documents the training arguments, including batch size, number of epochs, and learning rate adjustments.

---

## 📈 Results

Our experimental results show that the integration of an additional output feature significantly enhances the model's performance. Specifically, our extended version of **GraphCodeBERT** achieved the highest precision, recall, and f-measure compared to other state-of-the-art techniques.

The table below summarizes the performance of various approaches:

| **Approach**                      | **Precision** | **Recall** | **F-Measure** |
|-----------------------------------|:-------------:|:----------:|:-------------:|
| CodeBERT                     | 0.72          | 1.00       | 0.84          |
| Output Analysis               | 0.88          | 0.93       | 0.90          |
| Boosting (XGBoost)            | 0.88          | 0.99       | 0.93          |
| Bagging (Random Forest)       | 0.95          | 0.97       | 0.96          |
| GraphCodeBERT                | 0.98          | 0.95       | 0.96          |
| **Our GraphCodeBERT variant**     | **0.98**      | **1.00**   | **0.99**      |

---

## 📚 Reference

If you use this work, please cite:

```bibtex
@misc{martinezgil2024graphcodebert,
      title={Improving Source Code Similarity Detection Through GraphCodeBERT and Integration of Additional Features}, 
      author={Jorge Martinez-Gil},
      year={2024},
      eprint={2408.08903},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```

---

## 📄 License

This project is licensed under the MIT License. 
