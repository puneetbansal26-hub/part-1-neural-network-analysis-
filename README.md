# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Overview
This project builds and analyzes a feed-forward neural network to predict customer churn using a structured tabular dataset. It covers the full pipeline — exploration, preprocessing, model training, evaluation, and hyperparameter experimentation.

## Dataset
`Q1_customer_churn_nn.csv` — 2000 customers, 15 features, binary target (`churn`).

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Place `Q1_customer_churn_nn.csv` in the same directory as `notebook.ipynb`
3. Open and run `notebook.ipynb` top to bottom in Jupyter

## Folder Structure
```
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── churn_distribution.png
    ├── loss_curve.png
    ├── confusion_matrix.png
    ├── all_loss_curves.png
    ├── model_comparison_table.png
    ├── model_comparison_table.csv
    └── evaluation_outputs.png
```

## Key Results
| Experiment | Test Acc | F1 (Churn) | AUC-ROC |
|---|---|---|---|
| Exp 1 – Baseline (64,32) ReLU | 0.9750 | 0.17 | 0.71 |
| Exp 2 – Deeper + tanh (128,64,32) | 0.9600 | 0.11 | 0.79 |
| Exp 3 – High LR (64,32) ReLU | 0.9675 | 0.13 | 0.77 |
| **Exp 4 – Smaller + Sigmoid (32)** | **0.9650** | **0.36** | **0.97** |

**Best model:** Exp 4 — single hidden layer (32 neurons), sigmoid activation, lr=0.0005, batch=16.

## Libraries Used
- scikit-learn (MLPClassifier)
- pandas, numpy
- matplotlib, seaborn
