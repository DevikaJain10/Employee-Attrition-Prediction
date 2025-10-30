# 🧠 HR Attrition Prediction using Machine Learning

Predicting whether an employee is likely to leave an organization using machine learning techniques.

## 📌 Project Overview

Employee attrition can cost companies significant resources. This project focuses on predicting attrition based on historical employee data using various machine learning models. The goal is to help HR teams proactively identify at-risk employees and take preventive actions.

---

## 📂 Dataset

The dataset used is based on the **IBM HR Analytics Employee Attrition & Performance** dataset, which contains information such as:
DATASET:- (https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Age
- Job Role
- Job Satisfaction
- Monthly Income
- Years at Company
- OverTime
- BusinessTravel
- Environment Satisfaction
- Work-Life Balance  
...and more.

Target variable: `Attrition` (Yes/No)

---

## 🧰 Tools & Libraries Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- GridSearchCV

---

## 📊 Steps Followed

1. **Data Preprocessing**
   - Handled missing values
   - Encoded categorical variables
   - Scaled numerical features (where needed)

2. **Exploratory Data Analysis (EDA)**
   - Visualized correlations and feature distributions
   - Identified key features contributing to attrition

3. **Model Building**
   - Trained multiple models: Logistic Regression, Decision Tree, Random Forest Classifier, Support Vector Machine(SVM)
   - Used **GridSearchCV** for hyperparameter tuning
   - Evaluated with accuracy, precision, recall, F1-score, and confusion matrix

4. **Performance**
   - Best model: Random Forest Classifier(after tuning)
   - Achieved **88% accuracy** on test data
Classification Report:

-    precision recall f1-score support
-0.0     0.85    0.89    0.87    250
-1.0     0.88    0.84    0.86    244
---

## 📈 Results

- **Accuracy**: 88%
- **Evaluation Metrics**: Confusion matrix, classification report
- **Key Features Influencing Attrition**:
  - Overtime
  - Job Satisfaction
  - Age
  - Years at Company

---

## Novelty 
- **Real-World Focus**: Tackles the business-critical issue of employee attrition using ML for proactive HR
decisions.
- **Class Imbalance Solved with SMOTE**: Balanced dataset using synthetic oversampling to improve minority
class prediction.
- **Comparative Modeling**: Evaluated Logistic Regression, Random Forest, and SVM to identify the
best-performing model.
- **Statistical Feature Selection**: Used SelectKBest with ANOVA F-value to reduce noise and improve model
performance.
- **Visual Insights for Stakeholders**: EDA with meaningful plots like Attrition vs. Job Satisfaction aids
managerial understanding.
- **Reliable Evaluation**: Applied K-Fold Cross-Validation for unbiased performance metrics
- **Managerial Recommendations**: Model insights guide HR teams on retention strategies based on top
influencing factors.

## 🚀 Future Improvements

- Implement XGBoost or LightGBM for possible performance gain
- Handle class imbalance with SMOTE or class weights
- Deploy model using Streamlit for HR dashboard

---
## 🔗 View the Project

Click the link below to open the project directly in Google Colab:

[📎 Open in Google Colab](https://colab.research.google.com/drive/1Xv3AWeMDUJMcL87hZ0ZbGGml68jgoaK8?usp=sharing)

---
## 🤝 Acknowledgements

- IBM HR Analytics Dataset
- scikit-learn documentation
- pandas documentation
- Kaggle and online ML communities for inspiration
- SMOTE articles from MEDIUM
