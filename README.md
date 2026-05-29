# Diabetes Prediction Project

## Project Overview

This project predicts whether a patient is likely to have diabetes using machine learning algorithms. The project was adapted from a Kaggle notebook and reorganized into a modular Python project structure following software engineering principles.

The project performs data loading, preprocessing, visualization, model training, evaluation, and testing.

---

## Features

* Data loading from CSV dataset
* Data preprocessing and cleaning
* Missing value handling
* Feature scaling
* Data visualization
* Machine learning model training
* Model evaluation
* Unit testing
* GitHub Codespaces support

---

## Dataset

The dataset contains medical diagnostic measurements used to predict diabetes occurrence in patients.

Attributes include:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age
* Outcome

---

## Project Structure

```plaintext
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
├── tests/
│   ├── test_model.py
│   └── __init__.py
│
├── utils/
│   └── __init__.py
│
└── .devcontainer/
    └── devcontainer.json
```

---

## Setup Instructions

### Clone Repository

```bash
git clone https://github.com/syifa-azzah/diabetes_prediction_project.git
```

### Navigate to Project Folder

```bash
cd diabetes_prediction_project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage Guide

Run the complete project:

```bash
python main.py
```

The system will:

1. Load the dataset
2. Clean and preprocess data
3. Generate visualizations
4. Split training and testing data
5. Scale features
6. Train machine learning models
7. Evaluate model performance

---

## Testing

The project includes unit tests to verify the reliability and correctness of key components.

### Test Files

* `test_data_loader.py` – verifies dataset loading functionality.
* `test_preprocessing.py` – verifies data cleaning and preprocessing operations.
* `test_model_training.py` – verifies machine learning model initialization and training functionality.
* `test_evaluation.py` – verifies evaluation metrics and model performance calculations.

### Running Tests

Run individual tests using:

```bash
python -m tests.test_data_loader
```

```bash
python -m tests.test_preprocessing
```

```bash
python -m tests.test_model_training
```

```bash
python -m tests.test_evaluation
```

Successful execution of these tests confirms that the main project components function correctly and produce expected results.

---

## Machine Learning Models

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

---

## Results

| Model               | Train Accuracy | Test Accuracy |
| ------------------- | -------------- | ------------- |
| Logistic Regression | 77.04%         | 75.32%        |
| Decision Tree       | 100.00%        | 71.43%        |
| Random Forest       | 100.00%        | 73.38%        |
| SVM                 | 82.90%         | 74.68%        |

Logistic Regression achieved the best balance between training and testing performance.

---

## GitHub Codespaces

The project includes a `.devcontainer` configuration and can be executed directly in GitHub Codespaces.

---

## Author

Nur Syifa Azzah Binti Muhd Farouk (BS23110194) 

Computer Programming SF35803 2025/2026 

GitHub:
https://github.com/syifa-azzah
