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

(Dataset is downloaded from UCI inside Streamlit during training)

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

| ML Model Name         | Accuracy | AUC     | Precision | Recall  | F1     | MCC    |
|-----------------------|----------|---------|-----------|---------|--------|--------|
| Logistic Regression   | 0.916606 | 0.942437| 0.711775  | 0.436422| 0.541082| 0.516172 |
| Decision Tree         | 0.894635 | 0.732715| 0.532895  | 0.523707| 0.528261| 0.468983 |
| KNN                   | 0.908230 | 0.899342| 0.641914  | 0.419181| 0.507171| 0.471715 |
| Naive Bayes           | 0.820345 | 0.839329| 0.349509  | 0.690733| 0.464156| 0.400918 |
| Random Forest         | 0.919641 | 0.947576| 0.713826  | 0.478448| 0.572903| 0.543405 |
| XGBoost               | 0.923282 | 0.955604| 0.692208  | 0.574353| 0.627797| 0.588613 |


> Note: The actual values are generated automatically after running `train_and_save_models.py`
and saved into `model/model_metrics.csv`.

---

### Observations Table

| ML Model Name         | Observation about model performance |
|-----------------------|-------------------------------------|
| Logistic Regression   | Strong baseline with high accuracy and AUC, but recall is lower due to class imbalance. |
| Decision Tree         | Lower AUC compared to other models; tends to overfit and gives less stable generalization. |
| KNN                   | Performs reasonably well but slightly weaker recall and slower for large datasets. |
| Naive Bayes           | Gives the highest recall among models but low precision, meaning more false positives. |
| Random Forest         | Very strong overall performance with good balance between precision and recall. |
| XGBoost               | Best performing model overall with the highest AUC, F1 score and MCC. |

---

## How to Run (Local / BITS Virtual Lab)

### Step 1: Install dependencies
```bash
pip install -r requirements.txt



