# Assignment 5: Scikit-Learn Classification

## Project Purpose

The purpose of this project is to demonstrate how machine learning classification can be performed using Python and the Scikit-Learn library. The program uses the built-in Breast Cancer Wisconsin dataset from Scikit-Learn to train and evaluate multiple classification models. These models attempt to predict whether a tumor is malignant or benign based on various measured features.

This assignment helps demonstrate how machine learning can assist in medical detection tasks by identifying patterns in real-world datasets.

---

## Dataset

This project uses the **Breast Cancer Wisconsin Diagnostic dataset** included in Scikit-Learn.

It can be loaded with:
sklearn.datasets.load_breast_cancer()


The dataset contains numerical features derived from images of breast mass samples.

Target classes:
- **0 = malignant**
- **1 = benign**

The dataset includes:
- 569 total samples
- 30 numerical features
- binary classification labels

---

## Models Used

Three different classification models were trained and evaluated:

### Logistic Regression
A linear model commonly used for binary classification. Logistic regression works well when features are scaled and can often produce strong results on structured datasets.

### K-Nearest Neighbors (KNN)
A distance-based model that classifies data based on the closest training samples. The value of **k** (number of neighbors) was tested with multiple values to determine the best performing configuration.

### Decision Tree
A tree-based model that splits data based on feature values. Decision trees do not require feature scaling and can capture nonlinear relationships in the dataset.

---

## Machine Learning Workflow

The program follows a standard machine learning workflow:

1. Load the dataset from Scikit-Learn
2. Split the data into **training + validation** and **test** sets
3. Split the training data again into **training** and **validation** sets
4. Standardize features using **StandardScaler** for models that require scaling
5. Train models on the training data
6. Tune model parameters using the validation data
7. Retrain the best configuration using the combined training data
8. Evaluate final performance on the held-out test set

This approach ensures that the **test set is not used during tuning**, which helps produce more reliable performance results.

---

## Parameter Tuning

A simple validation-based tuning process was used.

Each model tested a small number of possible parameter values:

### Logistic Regression
Different values of the regularization parameter **C** were tested.

### KNN
Multiple values of **k (number of neighbors)** were tested.

### Decision Tree
Different values of:
- `max_depth`
- `min_samples_split`

The best model configuration was selected using **F1-score on the validation dataset**.

---

## Evaluation Metrics

Each model is evaluated using several metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Confusion Matrix
- Classification Report

The **F1-score** is used as the primary comparison metric because it balances precision and recall, which is important in medical classification problems where both false positives and false negatives matter.

---

## How to Run the Program

Install the required library:
conda install scikit-learn

Run the program:
python main.py


The program will print the results of each model in the terminal along with the best performing model.

---

## Output

The script prints:

- validation tuning results
- final evaluation metrics for each model
- confusion matrices
- classification reports
- the best performing model
- a short paragraph summarizing the results

---

## Limitations

This project uses a single train/validation/test split. Results may vary slightly if a different random seed or dataset split is used.

More advanced tuning methods such as cross-validation or grid search could further improve performance, but the goal of this project was to implement a clear and understandable machine learning workflow using core Scikit-Learn tools.

