# Credit_Risk_Analysis

## Overview of the analysis

For this analysis we need to classify a group of possible clients to identify which ones could represent a credit risk. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we employ different techniques to train and evaluate models with unbalanced classes using the imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Machine learning models led us to predict credit risk and have more accurate identification of good candidates for loans which will lead to lower default rates. Supervised learning allows us to create different models that we can use to generate predictions about our data. Finally, we evaluate the performance of the models we obtained and generate a recommendation.

## Results of the Credit_Risk_Analysis 

**- Naive Random Oversampling**

<img src="https://github.com/Jponce25/Credit_Risk_Analysis/blob/6befc79ae6bd9044628096e02046d6dab5c41bfb/Image/Imagen1.png" width="550">

Balanced Accuracy: 0.6438627638488825 (64.4%)

Precision: 0.01 (1%)

Recall: 0.69 (69%)

Conclusion: The model give us a high Recall, while the precision is low for High-risk loans. Calculating the F1 score is 0.02, we can note a pronounced imbalance trade-off between sensitivity and precision. So this model will no be able to capture the true positives for high risk loans. In other words, there are many false positives.

**- SMOTE Oversampling**

<img src="https://github.com/Jponce25/Credit_Risk_Analysis/blob/6befc79ae6bd9044628096e02046d6dab5c41bfb/Image/Imagen2.png" width="550">

Balanced Accuracy: 0.6628910844779521 (66.3%)

Precision: 0.01 (1%)

Recall: 0.63 (63%)

Conclusion: The model in general has a better performance (66.4%), however, a low precision (1%) and a high recall (63%) are still observed. This model is not the best at predicting properly.

**- Undersampling**

<img src="https://github.com/Jponce25/Credit_Risk_Analysis/blob/6befc79ae6bd9044628096e02046d6dab5c41bfb/Image/Imagen3.png" width="550">

Balanced Accuracy: 0.5442661782548694 (54.4%)

Precision: 0.01 (1%)

Recall: 0.69 (69%)

Conclusion: For this model we obtained a lower balanced accuracy (54%) compared to previous models, also we have a low precision (1%) and a high recall (69%). This model is nether the best option to predict.

**- SMOTEENN (Combination Under-Over Sampling)**

<img src="https://github.com/Jponce25/Credit_Risk_Analysis/blob/6befc79ae6bd9044628096e02046d6dab5c41bfb/Image/Imagen4.png" width="550">

Balanced Accuracy: 0.6748328802711889 (67.5%)

Precision: 0.01 (1%)

Recall: 0.76 (76%)

Conclusion: Although the accuracy of the model is better compared to the previous ones (67.5%), our precision continues to be low (1%) while our recall is now higher (76%). This model increased the recall but reviewing the result of f1 (0.02) we observe that it is a more unbalanced model than the previous ones.

**- Balanced Random Forest Classifier**

<img src="https://github.com/Jponce25/Credit_Risk_Analysis/blob/6befc79ae6bd9044628096e02046d6dab5c41bfb/Image/Imagen5.png" width="550">

Balanced Accuracy: 0.7885466545953005 (78.9%)

Precision: 0.03 (3%)

Recall: 0.70 (70%)

Conclusion: Using the Balanced Random Forest Classifier model, it is observed that the accuracy of the model improves notably (78.9%), however, we still have a very low precision (3%). The F1 result is now 0.06 which is slightly better than previous models.

**- Easy Ensemble Classifier**

<img src="https://github.com/Jponce25/Credit_Risk_Analysis/blob/6befc79ae6bd9044628096e02046d6dab5c41bfb/Image/Imagen6.png" width="550">

Balanced Accuracy: 0.9316600714093861 (93.2%)

Precision: 0.09 (9%)

Recall: 0.92 (92%)

Conclusion: The Easy Ensemble Classifier model is the one that obtains the best results, a very high balanced accuracy (93.2%), with a slight improvement in accuracy (9%) and a higher recall than previous models (92%).

## Summary of the Credit_Risk_Analysis 

The pronounced imbalance between the two classes (high_risk and low_risk) can cause machine learning models to be biased toward the majority class. In such a case, the model will be much better at predicting low_risk transactions than high_risk ones. In such a case, even a model that blindly classifies every transaction as low_risk will achieve a very high degree of accuracy. 

One strategy to deal with class imbalance is to use oversampling and undersampling, the idea is simple if one class has too few instances in the training set, we choose more instances from that class for training until it's larger or instead of increasing the number of the minority class, the size of the majority class is decreased.

Another strategy is to use appropriate metrics to evaluate a model's performance, such as precision and recall. The F1 score is a good score to notice the trade-off between sensitivity and precision. A useful way to think about the F1 score is that a pronounced imbalance between sensitivity and precision will yield a low F1 score.

In this analysis we perform 6 models to try to predict subprime loans. The first four models focused on undersampling and oversampling tools using a logistic regression model. In such a scenario, the sensitivity is very high, while the precision is very low. Clearly, this is not a useful algorithm. Even calculating the F1 score we can see a pronounced imbalance between sensitivity and precision will yield a low F1 score.

For the last two models, we use ensemble classifiers to try to predict which loans are high or low risk. The concept of ensemble learning is the process of combining multiple models, to help improve the accuracy and robustness, as well as decrease variance of the model, and therefore increase the overall performance of the model. For this analysis we use a Balanced Random Forest Classifier that although it does not have the best performance, it allows us to identify the importance of each feature, being able to determine and, where appropriate, eliminate the features that are not important in the model.

Finally, we used an Easy Ensemble Classifier that had the best balance accuracy, precision, and recall. Of the 6 models analyzed, this would be the best machine learning model to choose for further credit card analysis risk. However, the model could be improved by using stratification at the time of assigning the Train and Test Sets, we could also try running this same model without the important variables identified by the Balanced Random Forest Classifier.
