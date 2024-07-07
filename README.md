# Credit_Card Fraud Detection Capstone Project

## FindDefault (Prediction of Credit Card fraud)


### Introduction
A credit card is one of the most used financial products to make online purchases and payments. Though the Credit cards can be a convenient way to manage your finances, they can also be risky. Credit card fraud is the unauthorized use of someone else's credit card or credit card information to make purchases or withdraw cash.
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. 


### Dataset
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.


### Expected Output
We have to build a classification model to predict whether a transaction is fraudulent or not.


## My focus on this Project
#### My focus in this project are as the following: 
##### The following is recommendation of the steps that should be employed towards attempting to solve this problem statement: 
1.  **Exploratory Data Analysis:** Analyze and understand the data to identify patterns, relationships, and trends in the data by using Descriptive Statistics and Visualizations. 
2.  **Data Cleaning:** This might include standardization, handling the missing values and outliers in the data. 
3. 	**Dealing with Imbalanced data:** This data set is highly imbalanced. The data should be balanced using the appropriate methods before moving onto model building.
4. 	**Feature Engineering:** Create new features or transform the existing features for better performance of the ML Models. 
5.	**Model Selection:** Choose the most appropriate model that can be used for this project. 
6. 	**Model Training:** Split the data into train & test sets and use the train set to estimate the best model parameters. 
7.  **Model Validation:** Evaluate the performance of the model on data that was not used during the training process. The goal is to estimate the model's ability to generalize to new, unseen data and to identify any issues with the model, such as overfitting. 
8.	**Model Deployment:** Model deployment is the process of making a trained machine learning model available for use in a production environment. 


### Data Exploration and Preprocessing
#### Initial Exploration
The dataset is loaded using pandas, and initial exploration involves checking the first few rows, summary statistics, and data types of the columns. The dataset is checked for missing values and found to be complete.


#### Handling Data Imbalance
#### To handle the imbalance, several approaches are employed:
1.  **Undersampling:** Reducing the number of non-fraudulent transactions to match the count of fraudulent transactions.
2.  **Oversampling:** Increasing the number of fraudulent transactions to match the count of non-fraudulent transactions.
3.  **SMOTE (Synthetic Minority Over-sampling Technique):** Creating synthetic data points to increase the number of fraudulent transactions.
4.  **ADASYN (Adaptive Synthetic Sampling):** Similar to SMOTE, but focuses on creating data points in regions with low density of minority class samples. 


#### Data Exploration and Preprocessing
##### Initial Exploration
The dataset is loaded using pandas, and initial exploration involves checking the first few rows, summary statistics, and data types of the columns. The dataset is checked for missing values and found to be complete.


### Model Building and Evaluation
#### Model Selection
##### Various machine learning models are considered:
1.Logistic Regression
2. Decision Trees
3. XGBoost


#### Training and Validation
The dataset is split into training and testing sets. Cross-validation techniques are used to ensure robustness and avoid overfitting.


#### Performance Metrics
Key performance metrics include accuracy, precision, recall, F1-score, and ROC-AUC. A confusion matrix is used to visualize the performance of the classification models.


#### Hyperparameter Tuning
Hyperparameters are tuned using GridSearchCV to find the optimal parameters.


#### Choosing the Best Model
The Logistic Regression model using the SMOTE balanced dataset showed excellent performance with an ROC score of 0.99 on the train set and 0.97 on the test set. It is chosen for its simplicity, ease of interpretation, and lower computational resource requirements.


### Results and Insights
#### Summary of Findings
Key insights from the data exploration phase, including significant patterns or anomalies detected.

#### Feature Importance
Important features impacting the model’s decisions are identified.

#### Model Performance
The final model’s performance is reported on the test set, comparing it against baseline models. The Logistic Regression model with SMOTE balancing demonstrated high recall and a strong ROC-AUC score.


### Model Deployment
#### Saving the Best Model
The best-performing model is serialized using the pickle module for Deployment:

```python
with open('load_model.pkl', 'wb') as file:
    pickle.dump(load_model, file)
