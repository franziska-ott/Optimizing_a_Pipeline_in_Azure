# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
This dataset contains data about bank customers and their respones to marketing compaigns. We seek to predict whether a customer subscribed to a product or not.

The best performing model was a.

## Scikit-learn Pipeline
Data Cleansing: First the input data is cleaned. One-hot encoding is performed on categorical features. Categorical variables are converted into binary indicators, months and weekdays are mapped to numerical representations and binary varibales are transformed to 0s and 1s. Moreover, the target variable is defined. 

Classification Algorithm: Since we want to perform a binary classification task, logistic regression is used.

Hyperparameter Tuning: Hyperparameter Tuning is performed to find the optimal values for "C" and "max_iter" that result in the best performance accuracy-wise for the logistic regression model. A random sampling is used to explore the hyperparameter search space more efficiently and improve the changes of finding good hyperparameter configurations. 

Early Stopping: A bandit early stopping policy is used since it offers such as efficient resource allocation by terminating underperforming runs early, resulting in cost savings and faster convergence to optimal configurations. It strikes a balance between exploration and exploitation, adapting to performance variations and improving overall tungin efficiency.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
The HyperDrive LogisticRegression model got an accuracy of 0.90561 with the optimal paramters C=10 and max_iter=1000.
The best AutoML Model .

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
