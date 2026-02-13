# ML Assignment 2 - Classification Models + Streamlit Deployment

## a) Problem Statement
The objective of this project is to implement multiple classification machine learning models on a single dataset, evaluate them using standard classification metrics, and deploy an interactive Streamlit web application for model comparison and prediction.

---

## b) Dataset Description (UCI Bank Marketing Dataset)
Dataset: UCI Bank Marketing Dataset  
Type: Binary Classification  
Target Column: `y` (yes/no)

- Number of instances: ~45,000
- Number of features: 16+
- Source: Public UCI dataset (commonly mirrored in open GitHub dataset repositories)

This dataset satisfies the assignment constraints:
- Minimum 12 features
- Minimum 500 instances

---

## c) Models Used + Evaluation Metrics

### Models Implemented
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

---

### Comparison Table

| ML Model Name         | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|-----------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression   | (auto)   | (auto)| (auto)    | (auto) | (auto)| (auto)|
| Decision Tree         | (auto)   | (auto)| (auto)    | (auto) | (auto)| (auto)|
| KNN                   | (auto)   | (auto)| (auto)    | (auto) | (auto)| (auto)|
| Naive Bayes           | (auto)   | (auto)| (auto)    | (auto) | (auto)| (auto)|
| Random Forest         | (auto)   | (auto)| (auto)    | (auto) | (auto)| (auto)|
| XGBoost               | (auto)   | (auto)| (auto)    | (auto) | (auto)| (auto)|

> Note: The actual values are generated automatically after running `train_and_save_models.py`
and saved into `model/model_metrics.csv`.

---

### Observations Table

| ML Model Name         | Observation about model performance |
|-----------------------|-------------------------------------|
| Logistic Regression   | Works as a strong baseline and performs well after one-hot encoding. |
| Decision Tree         | Can overfit and may produce unstable results depending on split depth. |
| KNN                   | Produces reasonable results but is slower on large datasets. |
| Naive Bayes           | Very fast, but performance may be limited due to independence assumption. |
| Random Forest         | Consistently strong and stable due to ensemble averaging. |
| XGBoost               | Typically achieves the best results due to boosting and non-linear learning. |

---

## How to Run (Local / BITS Virtual Lab)

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
