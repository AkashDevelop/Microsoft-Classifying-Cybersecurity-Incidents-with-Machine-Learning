# 🚨 **Incident Analysis Using Machine Learning** 🚀  
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://www.python.org/)  
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)  
[![Data Size](https://img.shields.io/badge/Dataset-1.3M%20rows-orange)](#dataset-description)  
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen)](#final-results)  

---

## 🌟 **Overview**  

🎯 **Goal**: Analyze and classify incidents to improve decision-making using Machine Learning.  
💡 **Approach**:  
- Preprocess and clean large-scale data for high-quality insights.  
- Train a robust classification model for accurate predictions.  
- Deliver actionable insights to guide decisions.  

📽️ **Visual Representation**  
![download (16)](https://github.com/user-attachments/assets/6904c7de-b60d-4617-a9f2-99708fd32664) 

---

## 🎯 **Objective**  

1️⃣ Preprocess and clean **1.3M+ rows** of incident data.  
2️⃣ Engineer features for improved model performance.  
3️⃣ Train and evaluate a high-performing classification model.  
4️⃣ Provide interpretability and actionable insights.  

---

## 📊 **Dataset Description**  

- **Size**: `1,297,443 rows × 39 columns`  
- **Key Features**:  
  - `Category`, `IncidentGrade`, `EntityType`, `Hour`, `DayOfWeek`, etc.  
- **Target Variable Distribution**:  
  - `BenignPositive`: 2,054,774  
  - `TruePositive`: 1,662,087  
  - `FalsePositive`: 1,015,782  

🚨 **Missing Values**:  
- Dropped columns with >50% missing values: `ActionGrouped`, `ResourceType`.  
- Imputed numerical and categorical columns using median/mode.  

---

## 🛠️ **Preprocessing Steps**  

**Data Cleaning**:  
- Removed duplicates.  
- Handled missing values with imputation.  

**Outlier Removal**:  
- Used IQR method for numerical features.  

**Feature Engineering**:  
- Extracted temporal features like `Year`, `Month`, `Hour`, `DayOfWeek`.  
- Encoded categorical features with Label and One-Hot Encoding.  

**Scaling**:  
- Applied Min-Max Scaling to numerical features.  

---

## 🤖 **Model Selection**  

- **Chosen Model**: `Random Forest Classifier`  
- **Why Random Forest?**  
  - Efficient for large datasets.  
  - Robust to overfitting using ensemble learning.  
  - Provides feature importance for interpretability.  
  - Works well with imbalanced datasets using SMOTE.  

---

## 🏋️‍♂️ **Training Process**  

1️⃣ Split data into train and test sets.  
2️⃣ Addressed class imbalance with **SMOTE**.  
3️⃣ Conducted hyperparameter tuning via **GridSearchCV**.  
4️⃣ Evaluated model using **F1 score** with cross-validation.  

📈 **Performance**:  
- **Training F1 Score**: `93%`  

---


## 📷 **Features importance**  

![download (17)](https://github.com/user-attachments/assets/930cd5dd-a637-4f59-8be7-449d0ec22908)

---

## 🧪 **Test Dataset Workflow**  

- Preprocessed test data following the training pipeline.  
- Aligned features with the training dataset.  
- Loaded the saved model to make predictions.  

📉 **Performance**:  
- **Test F1 Score**: `88%`  

---

## 🌟 **Challenges Faced**  

- **Class Imbalance**: Solved using **SMOTE**.  
- **High Missing Values**: Dropped columns (>50%) and imputed remaining values.  
- **Overfitting**: Addressed using cross-validation and hyperparameter tuning.  
- **Temporal Data Handling**: Extracted features like `Hour`, `DayOfWeek`, etc.  

---

## 🎉 **Final Results**  

| Metric        | Train F1 Score | Test F1 Score |  
|---------------|----------------|---------------|  
| **Accuracy**  | `96%`          | `88%`         |  

🔍 **Insights**:  
- High training accuracy shows the model's strength in learning patterns.  
- A significant drop in test performance suggests data variability challenges.  

---

## 🚀 **Conclusion and Future Work**  

✅ **Conclusion**:  
- Successfully implemented a robust pipeline for data preprocessing and modeling.  
- Demonstrated high training performance, with areas to improve test accuracy.  

🔮 **Future Improvements**:  
- Explore advanced techniques to handle class imbalance.  
- Investigate feature importance and engineering further.  
- Incorporate additional data for better generalization.  

---

## 📦 **Key Libraries Used**  

[![Pandas](https://img.shields.io/badge/Library-Pandas-blue)](https://pandas.pydata.org/)  
[![Scikit-learn](https://img.shields.io/badge/Library-Scikit--learn-orange)](https://scikit-learn.org/stable/)  
[![NumPy](https://img.shields.io/badge/Library-NumPy-green)](https://numpy.org/)  
[![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-purple)](https://matplotlib.org/)  


---

## 🌐 **How to Use**  

1️⃣ Clone the repository.  
2️⃣ Install dependencies using `requirements.txt`.  
3️⃣ Run the pipeline with `python pipeline.py`.  
4️⃣ Visualize results using `analysis.ipynb`.  

🎯 **Commands**:  
```bash
git clone https://github.com/username/incident-analysis.git
cd incident-analysis
pip install -r requirements.txt
python pipeline.py

