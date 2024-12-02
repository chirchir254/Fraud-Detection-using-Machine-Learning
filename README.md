# Fraud Detection using Machine Learning

## Overview
This project aims to build a machine learning model to detect fraudulent transactions in a highly imbalanced credit card dataset. The dataset contains transactions labeled as either normal or fraudulent, with a significant imbalance favoring normal transactions. The main goal is to develop a model that can accurately identify fraudulent transactions despite this imbalance.

[Notebook](https://github.com/chirchir254/Fraud-Detection-using-Machine-Learning/blob/main/Fraud_Craud_Detection.ipynb)

## Dataset
- ### Sources:   
    The dataset used for this project is from kaggle creditcard.csv.
- ### Description:
    The dataset is highly imbalanced:

      0 represents a normal transaction.

      1 represents a fraudulent transaction.

## Data Preprocessing
- ### Data Loading:
    The dataset is loaded into a pandas DataFrame for analysis.
  
- ### Handling Missing Values:
    The dataset is checked for any missing values, and no missing values found.
  
- ### Class Imbalance:
  Given the imbalance in the dataset, under-sampling techniques are applied to create a balanced dataset containing a similar distribution of normal and fraudulent transactions.

## Model Training
- ### Algorithm Used:
    The model is trained using Logistic Regression.
- ### Training Process:
  - The dataset is split into training and test sets using train_test_split.
  - The Logistic Regression model is trained on the training data.

## Model Evaluation
  - ### Evaluation Metrics:
      - #### Accuracy Score:
        - The model's performance is evaluated using the accuracy score on both the training and test datasets.

## Results
  - ### Training Accuracy:
    - The accuracy score on the training and test data is calculated and displayed.

      -  training_data_accuracy: 0.9440914866581956

      -  test_data_accuracy: 0.9441624365482234

## Conclusion
  - This project successfully demonstrates the application of machine learning techniques to detect fraudulent transactions in an imbalanced dataset. The Logistic Regression model, coupled with under-sampling techniques, provides a foundation for developing more sophisticated fraud detection systems.

## Future Work
  - Explore other machine learning algorithms such as Random Forest, SVM, or Neural Networks.
  - Implement over-sampling techniques like SMOTE to balance the dataset.
  - Perform hyperparameter tuning to optimize model performance.
