# Credit Risk Analysis Report

## Overview of the Analysis

* The purpose of this analysis was to train and evaluate a model based on loan risk. 
* The dataset was historical lending activity from a lending services company. The model will be used to identify the creditworthiness of borrowers. 
* The original dataframe was broken down into a labels set (variable y) and features set (variable X). The "loan_status" column was assigned to the variable "y" and is the column the model is trying to predict. The remaining columns were assigned to the "X" variable. 
* Steps of the Machine Learning Process I used:
    * First, I read the "lending_data.csv" file and converted into a Pandas DataFrame.
    * The "loan_status" column was separated out as the labels set and assigned to the variable "y". The remaining colunns were assigned to the variable "X" as the features set. 
    * The labels and features set were split into training and testing datasets using the train_test_split module from the sk.learn library.
    * Using the LogisticRegression module from sk.learn, I initiated and fit a LogisticRegression model using the X_train and y_train datasets.
    * I then used the LogisticRegression model to make predictions using the X_test dataset and assigned this to a "testing_predictions" variable. After, I converted the predictions to a Pandas DataFrame with a "Prediction" column using the "testing_predictions" and an "Actual" column using the y_test dataset. 
    * Finally, using the confusion_matrix and classification_report modules from sk.learn, I generated a confusion matrix and classification report based on the "testing_predictions" and y_test datasets to evaluate the logistic regression model's performance. 

## Results

* Logistic Regression Model Results for Class 0 (Healthy Loan):
    * Precision: 1.00
    * Recall: 1.00
    * F1-Score: 1.00
    
* Logistic Regression Model Results for Class 1 (High-Risk Loan): 
    * Precision: 0.87
    * Recall: 0.95
    * F1-Score: 0.91
    
* Accuracy score - showing overall correctness of the model across both classes: 0.99

## Summary

I would recommend using this model to predict healthy vs. high-risk loans. 
-
The logistic regression model predicts both labels well. The healthy loan label (0) had precision and recall scores of 1, meaning the model has perfect precision and recall. This means that all the instances predicted as positive are positive and the model captures all the positive instances correctly. There are no false negatives or false positives. 

The high-risk loan label (1) still had high scores, with precision and recall scores of 0.87 and 0.95 respectively. 

The overall accuracy score across both classes is 0.99. This means that the model not only captures the vast majority of positive instances, but it is also very precise in its positive predictions. 

I believe performance does depend on the problem we are trying to solve. In this case, the model had perfect precision and recall scores for the healthy loan label (1). This means the company can confidently trust that the model is predicting and capturing all positive instances and that there are no false positives. Therefore, the company can trust that all of the healthy loan predictions are actually healthy. 
-
Although the precision and recall scores are not perfect for the high-risk loan label (0), they are still quite high. It is a fair assumption that the model is still capturing the majority of positive instances, and that it is fairly accurate in predicting which loans are unhealthy and high-risk. 

In this case, I think it would be more important to monitor the performance of high-risk loans compared to healthy loans. This would allow the lending company to take preventative actions when lending high-risk loans. While the performance of the high-risk loan label is lower than the healthy loan label, it is still very high and the overall accuracy is 0.99 so the company can confidently trust the model's predicitons of high risk loans. 
