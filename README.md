# COVID-19-Misinformation-Detector-BoW

Detecting COVID-19 misinformation in news articles using classical Bag-of-Words feature extraction paired with neural network classifiers.

---

## 🎯 **Objective**

To build and compare text classification models using **Count Vectorizer** and **TF-IDF** features across three neural architectures — LNN, LSTM, and MLP — with a focus on maximizing recall for the misinformation class.

---

## 🗂 **Dataset**

* **COVID-19 News Dataset**
* Training & Validation: Articles from **May** (reliable + fake)
* Test Set: Articles from **July**
* Binary labels: `0` = Reliable, `1` = Misinformation
* Class imbalance: Reliable news significantly outnumbers misinformation

---

## 🧪 **Experiments**

| Experiment | Model | Feature Extraction | Test Accuracy | Misinformation Recall |
|---|---|---|---|---|
| 1 | LNN | Count Vectorizer | 80.1% | 0.20 |
| 2 | LNN | TF-IDF | 80.1% | 0.23 |
| 3 | LSTM | Count Vectorizer | 78.9% | 0.08 |
| 4 | MLP | Count Vectorizer | 80.1% | 0.03 |
| 5 | MLP | TF-IDF | 80.9% | 0.12 |

**Best model:** Experiment 2 (LNN + TF-IDF) — highest misinformation F1 and most stable learning curves.

---

## 🧠 **Model Architectures**

* **LNN:** Single Dense layer with sigmoid — strictly linear classifier over sparse BoW features
* **LSTM:** 64 units, dropout 0.2 — sequence modeling on count-vector input
* **MLP:** 1 hidden layer (64 ReLU units, dropout 0.5) — non-linear feature compression

---

## 🛡 **Training Details**

* Optimizer: Adam
* Loss: Binary cross-entropy
* Batch size: 32 | Max epochs: 20
* Early stopping to prevent overfitting

---

## 📊 **Evaluation Metrics**

* Training / Validation accuracy & loss curves
* Confusion matrix
* Precision, Recall, F1 per class
* Focus on **misinformation recall** — missing fake news is costlier than a false alarm

---

## 🛠 **Technology Stack**

* **Language:** Python 3.x
* **Libraries:** TensorFlow / Keras, Scikit-learn, NLTK, NumPy, Pandas, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook (Google Colab compatible)

---

## 📁 **Repository Structure**

```
├── notebook.ipynb      # All 5 BoW experiments
└── README.md
```

> Update the dataset file paths in the notebook's data-loading cell to point to your local CSV files before running.
