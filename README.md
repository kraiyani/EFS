# Content for the README.md file

This README provides an overview of a machine learning pipeline for credit default prediction. The pipeline includes data preprocessing, feature selection, model selection, and evaluation of various classifiers.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Data Preprocessing](#data-preprocessing)
- [Feature Selection](#feature-selection)
- [Model Selection](#model-selection)
- [Evaluation](#evaluation)

## Introduction

The goal of this project is to predict credit default using a machine learning approach. The following steps are followed:

1. **Data Loading**: loading data from CSV files for client information, client bureau information, loan information, and default data.

2. **Data Cleaning and Preprocessing**: Performing data cleaning and preprocessing to handle missing values, convert data types, and create new features.

3. **Data Merging**: Merging the cleaned and processed data to create a comprehensive dataset for analysis.

4. **Feature Selection**: Selecting relevant features to improve model performance and reduce dimensionality.

5. **Model Selection**: Evaluating various machine learning classifiers, including logistic regression, support vector machine, random forest, gradient boosting, decision tree, and neural network.

6. **Model Evaluation**: Assessing the performance of each classifier using accuracy, precision, recall, and F1-score metrics.

## Requirements

Before running the code, ensure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:

pip install -r requirements.txt

## Getting Started

After installing requirements just run main.py to navigate through API

Clone this repository to your local machine.

Place the following CSV files in the same directory as your Jupyter Notebook or Python script:

Client_Bureau_Information.csv
Client_Information.csv
Default_Data.csv
Loan_Information.csv
Open the Jupyter Notebook or Python script containing the code.

Run the code step by step to perform data preprocessing, feature selection, model selection, and evaluation.

## Data Preprocessing

Data is loaded from CSV files for client information, client bureau information, loan information, and default data.
Data cleaning and preprocessing steps include handling missing values, converting data types, and creating new features.
Data from different sources are merged to create a comprehensive dataset for analysis.

## Feature Selection
Features are selected to improve model performance and reduce dimensionality.
Low-variance features and highly correlated features are removed.
Mutual information is used to identify and eliminate low-information features.

## Model Selection
Various machine learning classifiers are evaluated, including logistic regression, support vector machine, random forest, gradient boosting, decision tree, and neural network.
Grid search is used to optimize hyperparameters for each classifier.
The best classifiers and their parameters are stored for evaluation.

## Evaluation
Model performance is assessed using accuracy, precision, recall, and F1-score metrics.
A bar plot is created to visualize the performance of each classifier.

