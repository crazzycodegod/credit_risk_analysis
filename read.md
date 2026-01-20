***Credit Risk & Transaction Classification Projects
**Overview

This repository contains two practical data science projects focused on credit risk analysis and transaction classification, designed to simulate real-world financial data workflows. The projects demonstrate skills in data validation, feature analysis, model testing, and documentation, aligned with entry-level data science and analytics roles in financial services.

**Project 1: Credit Scoring Model Validation
Objective

To test and validate a credit scoring model using structured customer financial data, ensuring feature consistency, model reliability, and interpretable outputs.


**Dataset

*credit_data.csv
Synthetic dataset containing customer demographics, income, loan details, credit behavior, and default outcomes.

**Key features:
Age
Monthly income
Loan amount
Loan term
Credit score
Number of late payments
Target variable:
default (1 = defaulted, 0 = non default)

**Approach
Data loading and schema validation

Feature  target separation

Train test split with stratification

Feature scaling using standardization

Logistic regression model training

Model evaluation using accuracy, confusion matrix, and classification report

Feature importance analysis via model coefficients


**Tools
Python
Pandas
NumPy
Scikit-learn
_________________________________________________________________________________________

***Project 2: Transaction Classification (M-Pesa Style)
Objective

To test and validate a transaction classification system simulating mobile money (M-Pesa) transactions, focusing on fraud detection and classifier accuracy monitoring.

**Dataset
transactions_data.csv
Synthetic dataset representing mobile money transactions across multiple categories.

**Key features:
Transaction amount
Transaction type (send, paybill, buygoods)
Merchant category
Target variable:
is_fraud (1 = suspicious, 0 = normal)

**Approach
Data validation and consistency checks
Encoding categorical variables
Train-test split
Rule-based classification logic for testing
Accuracy measurement and classification reporting

**Tools
Python
Pandas
Scikit-learn

How to Run the Projects

  Clone the repository

Install dependencies:

  pip install pandas numpy scikit-learn


Generate datasets:

  python generate_transactions_data.py
  

  Run credit scoring model:

  python credit_scoring.py


  Run transaction classifier:  

  python transaction_classifier.py



***Skills I focused on in these 2 projects
Credit risk analysis
Data validation and testing
Feature engineering
Model evaluation
Documentation and reporting
Financial data analysis workflows