# Lending Club - Credit Risk (Machine Learning)

## Summary

Credit risk is an inherently unbalanced classification problem, as the number of good loans easily outnumber the number of risky loans. 

1. To use different techniques to train and evaluate models with unbalanced classes. 
2. To Evaluate the performance of these models and make a recommendation on whether they should be used to predict credit risk.

## Objectives
The goals:

1. Implement machine learning models.
2. Use resampling to attempt to address class imbalance.
3. Evaluate the performance of machine learning models. 

The Accuracy and Recall (sensitivity) rate are two important factor to select the righ model for the imbalanced dataset while the good data is overwhelming the bad data in the dataset.

## Jupyter Notebooks

1. [credit_risk_resampling.ipynb](Notebooks/credit_risk_resampling.ipynb)
2. [credit_risk_ensemble.ipynb](Notebooks/credit_risk_ensemble.ipynb)

## Naive Random Oversampling

**Balanced Accuracy Score**

0.69

**Confusion Matrix**

| |Predict high risk | Predict low risk |
|-|-|-|
|Actual high risk |78  | 23   |
|Actual low rick  |6811| 10293|
       
**Report Table**

| |pre|rec|spe|f1|geo|iba|sup|
|-|-|-|-|-|-|-|-|
|high_risk  |  0.01   |   0.77 |     0.60  |    0.02   |   0.68   |   0.47  |     101|
|low_risk   |  1.00   |   0.60 |     0.77  |    0.75   |   0.68   |   0.46  |   17104|
|avg / total|  0.99   |   0.60 |     0.77  |    0.75   |   0.68   |   0.46  |   17205|

### Analysis

**Accuracy**

The balanced accuracy score of Naive Random Oversampling is .69 or 69% after resampled the data training with the Random Over Sampler while Accuracy score is .60 or 60% based on the confustion matrix. Accuracy score formula is (TP+TN)/Total, which TP (True Positive) is 78 and TN(True Nagative) is 10293, and total is 17205, so (78+10293)/17205 = .602789. This is a important factor that we need to evaluate the Machine Learning (ML) method becuase of the accuracy prediction of the corrected cases in the dataset. **Naive Random Oversampling is actually improved the accuracy prediction**

**Recall (sensitivity):** (Refer: the confustion matrix and the classification report)

- High risk rate is 77% so every 100 high risk cases the model detects right 77 high risk cases and categorizes 33 high risk cases to become low risk cases. This is a number that we want to see a larger percent becuase we do not want high risk cases to become low risk cases without any caution. 
- Low risk rate is 60% so every 100 low risk cases the model categorizes 40 low risk cases to become high risk cases. This is just alarmed cases that need to be reviewed by business line

**Precision:**

- High risk rate is closed 0.01 (~1%). It tells that the number of low risk cases predicted as high risk cases is much larger than actual high risk cases 
- Low risk rate is closed 1.00 (~100%). It tells that the actual low risk cases are overrule the precision rate. This is very true in the imbalanced classificantion cases.

**For this dateset, Naive Random Oversampling accuracy and recall rate is the best among below models: Naive Random Oversampling, SMOTE,Undersampling,Combination Sampling. However, I recommend to use the Easy Ensemble AdaBoost Classifier model for the dataset because the accuracy and recall rate are very impressive(over 90%) (Refer: ReadMe.md)**
***


## SMOTE Oversampling

**Balanced Accuracy Score**

0.66

**Confusion Matrix**

| |Predict high risk | Predict low risk |
|-|-|-|
|Actual high risk |64  | 37   |
|Actual low rick  |5220| 11884|
       
**Report Table

| |pre|rec|spe|f1|geo|iba|sup|
|-|-|-|-|-|-|-|-|
|high_risk  |  0.01|      0.63|      0.69|      0.02|      0.66|      0.44|       101|
|low_risk   |  1.00|      0.69|      0.63|      0.82|      0.66|      0.44|     17104|
|avg / total|  0.99|      0.69|      0.63|      0.81|      0.66|      0.44|     17205|

### Analysis

**Accuracy**

The balanced accuracy score of SMOTE Oversampling is .66 or 66% after resampled the data training with the SMOTE while Accuracy score is .69 or 69% based on the confustion matrix. Accuracy score formula is (TP+TN)/Total, which TP (True Positive) is 64 and TN(True Nagative) is 11884, and total is 17205, so (64+11884)/17205 = .69444. With SMOTE model, the balanced accuracy rate is lower than the actual accuracy rate. It means the good loan (low risk) cases predicted better. **SMOTE model accuracy is very close and similar to Naive Random Oversampling, but the Naive Random Oversampling is better selection**

**Recall (sensitivity):** (Refer: the confustion matrix and the classification report)

- High risk rate is 63% so every 100 high risk cases the model detects right 63 high risk cases and categorizes 37 high risk cases to become low risk cases. This is a number that we want to see a larger percent becuase we do not want high risk cases to become low risk cases without any caution. **SMOTE model recall (sensitivity) rate is lower than Naive Random Oversampling**

- Low risk rate is 69% so every 100 low risk cases the model categorizes 41 low risk cases to become high risk cases. This is just alarmed cases that need to be reviewed by business line

**Precision:**

- High risk rate is closed 0.01 (~1%). It tells that the number of low risk cases predicted as high risk cases is much larger than actual high risk cases 
- Low risk rate is closed 1.00 (~100%). It tells that the actual low risk cases are overrule the precision rate. This is very true in the imbalanced classificantion cases.

***


## Cluster Centroids

**Balanced Accuracy Score**

0.53

**Confusion Matrix**

| |Predict high risk | Predict low risk |
|-|-|-|
|Actual high risk |67  | 34   |
|Actual low rick  |10217| 6887|
       
**Report Table**

| |pre|rec|spe|f1|geo|iba|sup|
|-|-|-|-|-|-|-|-|
|high_risk  |  0.01|      0.66|      0.40|      0.01|      0.52|      0.27|       101|
|low_risk   |  1.00|      0.40|      0.66|      0.57|      0.52|      0.26|     17104|
|avg / total|  0.99|      0.40|      0.66|      0.57|      0.52|      0.26|     17205|

### Analysis

**Accuracy**

The balanced accuracy score of Undersampling is .53 or 53% after resampled the data training with the ClusterCentroids while Accuracy score is .40 or 40% based on the confustion matrix. Accuracy score formula is (TP+TN)/Total, which TP (True Positive) is 67 and TN(True Nagative) is 6887, and total is 17205, so (67+6887)/17205 = .4041. With Undersampling model, the balanced accuracy rate is higher than the actual accuracy rate. **Undersampling model accuracy is very low. This method may not the righ model for this dataset**

**Recall (sensitivity):**(Refer: The confustion matrix and the classification report) 

- High risk rate is 66% so every 100 high risk cases the model detects right 66 high risk cases and categorizes 34 high risk cases to become low risk cases. This is a number that we want to see a larger percent becuase we do not want high risk cases to become low risk cases without any caution. **Naive Random Oversampling still the best model selection because the accurary and recall rate**

- Low risk rate is 40% so every 100 low risk cases the model categorizes 60 low risk cases to become high risk cases. This is  just alarmed cases that need to be reviewed by business line, but it is **very high alarmed cases**

**Precision:**

- High risk rate is closed 0.01 (~1%). It tells that the number of low risk cases predicted as high risk cases is much larger than actual high risk cases 
- Low risk rate is closed 1.00 (~100%). It tells that the actual low risk cases are overrule the precision rate. This is very true in the imbalanced classificantion cases.

***

## Combination (Over and Under) Sampling

**Balanced Accuracy Score**

0.63

**Confusion Matrix**

| |Predict high risk | Predict low risk |
|-|-|-|
|Actual high risk |67  | 34   |
|Actual low rick  |6736| 10368|
       
**Report Table**

| |pre|rec|spe|f1|geo|iba|sup|
|-|-|-|-|-|-|-|-|
|high_risk  |  0.01|      0.66|      0.61|      0.02|      0.63|      0.40|       101|
|low_risk   |  1.00|      0.61|      0.66|      0.75|      0.63|      0.40|     17104|
|avg / total|  0.99|      0.61|      0.66|      0.75|      0.63|      0.40|     17205|

### Analysis

**Accuracy**

The balanced accuracy score of Combination (Over and Under) Sampling is .63 or 63% after resampled the data training with the SMOTEENN while Accuracy score is .60 or 60% based on the confustion matrix. Accuracy score formula is (TP+TN)/Total, which TP (True Positive) is 67 and TN(True Nagative) is 10368, and total is 17205, so (67+10368)/17205 = .6065. With SMOTEENN model, the balanced accuracy rate is higher than the actual accuracy rate. **SMOTEENN model accuracy is lower than Naive Random Oversampling, but it is much better than Undersampling model(ClusterCentroids)**

**Recall (sensitivity):**(Refer: The confustion matrix and the classification report) 

- High risk rate is 66% so every 100 high risk cases the model detects right 66 high risk cases and categorizes 34 high risk cases to become low risk cases. This is a number that we want to see a larger percent becuase we do not want high risk cases to become low risk cases without any caution. **Naive Random Oversampling is still the highest rate so far**

- Low risk rate is 61% so every 100 low risk cases the model categorizes 39 low risk cases to become high risk cases. This is  just alarmed cases that need to be reviewed by business line.

**Precision:**

- High risk rate is closed 0.01 (~1%). It tells that the number of low risk cases predicted as high risk cases is much larger than actual high risk cases 
- Low risk rate is closed 1.00 (~100%). It tells that the actual low risk cases are overrule the precision rate. This is very true in the imbalanced classificantion cases.

***

# Extension (Machine Learning)


## BalancedRandomForestClassifier

**Accuracy Score

0.90

**Confusion Matrix

| |Predict high risk | Predict low risk |
|-|-|-|
|Actual high risk |68  | 33   |
|Actual low rick  |1749| 15355|
       
**Report Table

|precision|    recall|  f1-score|   support|
|-|-|-|-|
|   high_risk|       0.04|      0.67|      0.07|       101|
|    low_risk|       1.00|      0.90|      0.95|     17104|
|    accuracy|           |          |      0.90|     17205|
|   macro avg|       0.52|      0.79|      0.51|     17205|
|weighted avg|       0.99|      0.90|      0.94|     17205|

### Analysis

**Accuracy**

The accuracy score of Balanced Random Forest Classifier is .90 or 90%. Accuracy score formula is (TP+TN)/Total, which TP (True Positive) is 68 and TN(True Nagative) is 15355, and total is 17205, so (68+15355)/17205 = .8964. **The accuracy score is very good number. It can be a good model for this dataset. Let take a look into Recall (sensitivity) becuase this rate is important to detect high risk cases of this dataset**

**Recall (sensitivity):**(Refer: The confustion matrix and the classification report) 

- High risk rate is 67% so every 100 high risk cases the model detects right 67 high risk cases and categorizes 33 high risk cases to become low risk cases. This is a number that we want to see a larger percent becuase we do not want high risk cases to become low risk cases without any caution. **Recall rate is average number. It is not impressive like the accuracy. It tells that the low risk cases are detected better than the high risk cases**

- Low risk rate is 90% so every 100 low risk cases the model categorizes 10 low risk cases to become high risk cases. This is  just alarmed cases that need to be reviewed by business line.

**Precision:**

- High risk rate is closed 0.04 (~4%). It tells that the number of low risk cases predicted as high risk cases is much larger than actual high risk cases 
- Low risk rate is closed 1.00 (~100%). It tells that the actual low risk cases are overrule the precision rate. This is very true in the imbalanced classificantion cases.

***

## EasyEnsembleClassifier

**Accuracy Score

0.90

**Confusion Matrix

| |Predict high risk | Predict low risk |
|-|-|-|
|Actual high risk |94  | 7   |
|Actual low rick  |1706| 15398|
       
**Report Table

|precision|    recall|  f1-score|   support|
|-|-|-|-|
|   high_risk|       0.05|      0.93|      0.10|       101|
|    low_risk|       1.00|      0.90|      0.95|     17104|
|    accuracy|           |          |      0.90|     17205|
|   macro avg|       0.53|      0.92|      0.51|     17205|
|weighted avg|       0.99|      0.90|      0.94|     17205|


### Analysis (The best model)

**Accuracy**

The accuracy score of Easy Ensemble AdaBoost Classifier is .90 or 90%. Accuracy score formula is (TP+TN)/Total, which TP (True Positive) is 94 and TN(True Nagative) is 15398, and total is 17205, so (94+15398)/17205 = .9004. **The accuracy score is very impressive number. It can be a good model for this dataset. Let take a look into Recall (sensitivity) becuase this rate is important to detect high risk cases of this dataset**

**Recall (sensitivity):**(Refer: The confustion matrix and the classification report) 

- High risk rate is 93% so every 100 high risk cases the model detects right 93 high risk cases and categorizes 7 high risk cases to become low risk cases. This is a number that we want to see a larger percent becuase we do not want high risk cases to become low risk cases without any caution. **Recall rate is also very impressive number. Both the accuracy and the recall rate are over 90%. This is a good model to detect the high risk cases of this dataset**

- Low risk rate is 90% so every 100 low risk cases the model categorizes 10 low risk cases to become high risk cases. This is  just alarmed cases that need to be reviewed by business line.

**Precision:**

- High risk rate is closed 0.05 (~4%). It tells that the number of low risk cases predicted as high risk cases is much larger than actual high risk cases 
- Low risk rate is closed 1.00 (~100%). It tells that the actual low risk cases are overrule the precision rate. This is very true in the imbalanced classificantion cases.

***
