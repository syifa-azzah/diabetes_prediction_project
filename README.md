# Diabetes Prediction Project

## Project Overview

This project predicts diabetes using machine learning algorithms based on medical diagnostic data.

The original source code was adapted from a Kaggle notebook and reorganized into a modular Python project structure for better readability, maintainability, and software engineering practice.

---

## Features

* Data loading
* Data preprocessing
* Missing value handling
* Feature scaling
* Data visualization
* Machine learning model training
* Model evaluation

---

## Machine Learning Models Used

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

---

## Project Structure

```plaintext id="jlwmb2"
diabetes_prediction_project/
│
├── main.py
├── config.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── diabetes.csv
│   ├── data_loader.py
│   └── __init__.py
│
├── preprocessing/
│   ├── preprocessing.py
│   └── __init__.py
│
├── models/
│   ├── model_training.py
│   └── __init__.py
│
├── evaluation/
│   ├── evaluation.py
│   └── __init__.py
│
├── visualization/
│   ├── visualization.py
│   └── __init__.py
│
├── utils/
│   └── __init__.py
│
└── tests/
    └── __init__.py
```

---

## Dataset

The dataset contains medical diagnostic measurements used to predict diabetes occurrence in patients.

---

## How to Run

Install required libraries:

```bash id="jlwmb3"
python -m pip install -r requirements.txt
```

Run the project:

```bash id="jlwmb4"
python main.py
```

---

## Results

The project compares several machine learning algorithms based on training and testing accuracy.

Among the evaluated models, Logistic Regression achieved the best balance between training and testing performance.

---

## Author

Industrial Physics Programme Coursework Project
