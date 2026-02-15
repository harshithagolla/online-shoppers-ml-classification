# Online Shoppers Purchasing Intention - ML Classification Assignment

## 1. Problem Statement

The objective of this project is to predict whether an online website visitor will complete a purchase (Revenue = 1) or not (Revenue = 0) based on their browsing behavior and session characteristics.

This is a binary classification problem aimed at helping businesses improve marketing strategies, optimize customer targeting, and increase conversion rates by identifying high-intent users.

---

## 2. Dataset Description

Dataset Name: Online Shoppers Purchasing Intention Dataset  
Source: UCI Machine Learning Repository  
Total Instances: 12,330  
Total Features: 17  
Target Variable: Revenue (Binary: 0 = No Purchase, 1 = Purchase)

The dataset contains a mix of numerical and categorical features representing user behavior during an online session.

Some key features include:

- Administrative & Administrative_Duration
- Informational & Informational_Duration
- ProductRelated & ProductRelated_Duration
- BounceRates
- ExitRates
- PageValues
- SpecialDay
- Month
- VisitorType
- Weekend
- OperatingSystems
- Browser
- Region
- TrafficType

The dataset is slightly imbalanced, with approximately 15% positive purchase cases, making evaluation metrics such as AUC, F1 Score, and MCC important for proper performance assessment.

---

## 3. Models Implemented

The following six classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

All models were trained and evaluated using the same train-test split strategy to ensure fair comparison.

---

## 4. Model Evaluation Metrics Comparison

Each model was evaluated using the following performance metrics:

Accuracy
AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression | 0.8812   | 0.8876 | 0.7432    | 0.3560 | 0.4814   | 0.4603 |
| Decision Tree       | 0.8556   | 0.7307 | 0.5330    | 0.5497 | 0.5412   | 0.4557 |
| KNN                 | 0.8674   | 0.7829 | 0.6170    | 0.3796 | 0.4700   | 0.4145 |
| Naive Bayes         | 0.6736   | 0.7939 | 0.2941    | 0.7906 | 0.4287   | 0.3249 |
| Random Forest       | 0.9023   | 0.9198 | 0.7854    | 0.5079 | 0.6169   | 0.5814 |
| XGBoost             | 0.8998   | 0.9275 | 0.7103    | 0.5969 | 0.6486   | 0.5938 |

---

## 5. Model Performance Observations

| ML Model                | Observation                                                                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Logistic Regression** | Achieved strong accuracy and AUC, but recall was relatively low, indicating difficulty in identifying minority purchase cases.                                     |
| **Decision Tree**       | Captured nonlinear relationships but showed lower AUC compared to ensemble methods. Moderate overall performance.                                                  |
| **KNN**                 | Provided balanced performance but struggled with recall due to class imbalance.                                                                                    |
| **Naive Bayes**         | Achieved high recall but very low precision, resulting in many false positives. Overall weaker performance compared to other models.                               |
| **Random Forest**       | Delivered strong performance with high accuracy and balanced precision-recall tradeoff. Significant improvement over a single Decision Tree.                       |
| **XGBoost**             | Achieved the highest AUC and best overall F1 and MCC scores. Effectively handled nonlinear relationships and class imbalance, making it the best-performing model. |


---

## 6. Streamlit Application Features

The deployed Streamlit application includes:

- CSV upload option (test dataset)
- Model selection dropdown
- Display of evaluation metrics
- Confusion Matrix visualization
- Interactive and user-friendly interface

The application demonstrates real-time model training and performance comparison in an interactive environment.

---

## 7. Project Structure

```
online-shoppers-ml/
│
├── app.py
├── requirements.txt
├── README.md
│
└── model/
    ├── online_shoppers_training.ipynb
    ├── Logistic_Regression.pkl
    ├── Decision_Tree.pkl
    ├── KNN.pkl
    ├── Naive_Bayes.pkl
    ├── Random_Forest.pkl
    ├── XGBoost.pkl
```


---

## 8. Conclusion

Among all implemented models, XGBoost demonstrated the best overall performance based on AUC, F1 Score, and MCC metrics. Ensemble models (Random Forest and XGBoost) significantly outperformed individual classifiers, highlighting the importance of combining multiple decision trees for improved predictive performance.

This project successfully demonstrates a complete end-to-end machine learning workflow including:
- Data preprocessing
- Model implementation
- Performance evaluation
- Comparative analysis
- Interactive web application development using Streamlit
- Deployment on Streamlit Community Cloud
