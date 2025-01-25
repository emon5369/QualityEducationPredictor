# QualityEducationPredictor

QualityEducationPredictor is a machine learning project aimed at predicting whether a student requires special education support based on socio-demographic, academic, and personal factors. By leveraging various machine learning models, this project strives to aid in identifying students in need and contribute to equitable access to quality education.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Preprocessing Steps](#preprocessing-steps)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [How to Use](#how-to-use)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Access to quality education is essential for sustainable development and individual success. This project uses machine learning algorithms to predict whether a student needs additional educational support, using a dataset that includes features such as attendance rate, parental education, household income, and more. 

## Features
- Predicts special education needs using various machine learning models.
- Provides insights into factors influencing educational outcomes.
- Compares the performance of multiple models for accuracy and reliability.

## Dataset
The dataset includes:
- **Numerical Features:** Attendance rate, average exam scores, etc.
- **Categorical Features:** Gender, household income, parental education, etc.

The dataset has been preprocessed for training and testing purposes. 

## Preprocessing Steps
1. **Handling Missing Values:** Checked and confirmed no missing values.
2. **Data Encoding:**
   - Label Encoding for nominal features like gender and household income.
   - Ordinal Encoding for ordered features like grade level.
3. **Feature Scaling:** Standardized numerical features using `StandardScaler`.
4. **Data Splitting:**
   - Dataset split into training and testing sets with an 80:20 ratio.
   - Stratification used to maintain class proportions.
5. **Data Augmentation:** Dataset was duplicated and concatenated to increase training size.

## Machine Learning Models
The following machine learning algorithms were used:
1. **K-Nearest Neighbors (KNN):** Tuned using GridSearchCV for optimal `n_neighbors` and weighting schemes.
2. **Decision Tree:** Evaluated using accuracy and Mean Absolute Error (MAE).
3. **Support Vector Machine (SVM):** Utilized RBF kernel with hyperparameter tuning for `C` and `gamma` values.
4. **Random Forest:** Ensemble method for robust predictions.
5. **Gradient Boosting:** Sequential decision tree-based model.
6. **XGBoost:** Optimized implementation of gradient boosting with regularization.

## Results
| Model                        | Accuracy |
|------------------------------|----------|
| K-Nearest Neighbors (KNN)    | 0.89     |
| Decision Tree                | 0.90     |
| SVM (RBF kernel)             | 0.60     |
| Random Forest                | 0.89     |
| Gradient Boosting            | 0.56     |
| XGBoost                      | 0.79     |

- Decision Tree achieved the highest accuracy (0.90), making it the most effective model for this task.
- KNN and Random Forest also performed well with an accuracy of 0.89.
- Gradient Boosting and SVM showed lower performance.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/QualityEducationPredictor.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook project.ipynb
   ```
4. Explore the results and modify the code for further experimentation.

## Future Work
- Expand the dataset to include more diverse and representative samples.
- Experiment with deep learning models for better accuracy.
- Develop a user-friendly web interface for real-time predictions.
- Integrate feature importance analysis to provide actionable insights.

## Contributing
Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your proposed changes.
