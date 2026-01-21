Credit Risk & Transaction Analytics
Overview

This repository contains applied data science projects focused on credit risk analysis and transaction classification within a financial services context. The work simulates real-world workflows used in credit scoring, fraud detection, and model validation, with an emphasis on data quality checks, feature validation, and performance monitoring.

The projects are designed to demonstrate practical, production-adjacent skills rather than academic experimentation.

Project 1: Credit Scoring Model Validation
Problem Statement

Financial institutions rely on credit scoring models to assess borrower risk. This project focuses on validating such a model by ensuring data consistency, appropriate feature usage, and reliable predictive performance.

Dataset

Synthetic credit dataset (credit_data.csv) representing customer demographic and financial attributes:

Age

Monthly income

Loan amount

Loan term

Credit score

Number of late payments

Target variable:

default (1 = defaulted, 0 = non-default)

Methodology

Dataset schema validation and integrity checks

Feature and target separation

Stratified train-test split

Feature scaling using standardization

Logistic regression model training

Model evaluation using accuracy metrics, confusion matrix, and classification report

Feature impact analysis using model coefficients

Output

Validated credit scoring model

Interpretable feature importance

Reproducible testing workflow

Project 2: Transaction Classification (Mobile Money)
Problem Statement

Mobile money platforms process large volumes of transactions that must be classified accurately to identify suspicious or fraudulent behavior. This project tests a transaction classification system inspired by M-Pesa-style data.

Dataset

Synthetic transaction dataset (transactions_data.csv) containing:

Transaction amount

Transaction type (send, paybill, buygoods)

Merchant category

Target variable:

is_fraud (1 = suspicious, 0 = normal)

Methodology

Data validation and consistency checks

Encoding of categorical variables

Rule-based classification logic for testing and validation

Accuracy tracking and classification reporting

Output

Transaction classifier performance metrics

Error analysis framework for misclassified transactions

Technology Stack

Python

Pandas

NumPy

Scikit-learn

Git & GitHub

How to Run
pip install pandas numpy scikit-learn
python credit_scoring.py
python generate_transactions_data.py
python transaction_classifier.py

Key Skills Demonstrated

Credit risk analysis

Transaction classification

Data validation and testing

Feature engineering

Model evaluation and reporting

Financial data workflows

Notes

All datasets are synthetically generated for demonstration purposes and do not contain real customer data
