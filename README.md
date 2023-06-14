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
This project revolves around a dataset that captures the outcomes of marketing campaigns conducted by a Portuguese banking institution. 
The campaigns were centered around phone calls, where customers were approached to subscribe to a bank term deposit. 
The aim is to leverage customer attributes such as age, job, education, default status, and balance to predict whether 
a customer will ultimately subscribe to the term deposit. Two strategies were employed: a tuned logistic regression 
model and the Azure AutoML approach. The AutoML approach resulted in a powerful Voting-Ensemble model, achieving the 
highest validation accuracy of 91.75%. The project's objective is to enhance future marketing campaigns by optimizing 
term deposit predictions based on customer characteristics.

## Scikit-learn Pipeline
__Data Cleansing:__ The initial step involves cleaning the input data. Categorical features are encoded using one-hot encoding, 
transforming them into binary indicators. Months and weekdays are mapped to numerical representations, and binary 
variables are converted to 0s and 1s. Additionally, the target variable is defined for the classification task.

__Classification Algorithm:__ Logistic regression is chosen as the classification algorithm since it suits the binary 
classification task.

__Hyperparameter Tuning:__ To optimize the logistic regression model's performance, hyperparameter tuning is conducted. 
The values of "C" (regularization strength) and "max_iter" (maximum number of iterations) are tuned to achieve the best 
accuracy. Random sampling is utilized to efficiently explore the hyperparameter search space and increase the chances 
of finding optimal configurations.

__Early Stopping:__ A bandit early stopping policy is implemented to enhance resource allocation. It terminates 
underperforming runs early, resulting in cost savings and quicker convergence to optimal configurations. 
This policy balances exploration and exploitation, adapting to performance variations and improving the efficiency of
hyperparameter tuning.

By utilizing the estimator, parameter sampler, and an early termination policy, we construct a HyperDrive Config to 
conduct an experiment. After the run is completed, we identify the best model, which achieves a validation accuracy of 
90.56%. The optimal parameter values for this model are determined to be C=10 and max_iter=1000.

## AutoML
The top-performing model produced by AutoML is a Voting-Ensemble, which combines multiple ensemble models. 
Each ensemble is composed of a scaler and a subsequent model, both sourced from the Sci-kit learn library. 
The weights listed in the second column represent the respective contributions of these individual ensembles 
towards the final prediction.

| Model                                     | Ensemble weight |
| ----------------------------------------- | --------------- |
| StandardScalerWrapper + XGBoostClassifier | 8.83%           |
| StandardScalerWrapper + XGBoostClassifier | 16.67%          |
| StandardScalerWrapper + LightGMBClassifier| 8.83%           |
| StandardScalerWrapper + XGBoostClassifier | 8.83%           |
| StandardScalerWrapper + LogisticRegression| 8.83%           |
| StandardScalerWrapper + XGBoostClassifier | 8.83%           |
| MaxAbsScaler + SGD                        | 8.83%           |
| StandardScalerWrapper + XGBoostClassifier | 8.83%           |
| SparseNormalizer + XGBoostClassifier       | 25%             |
| MaxAbsScaler + LightGMB                    | 8.83%           |

## Pipeline comparison
After fine-tuning the hyperparameters of the Logistic Regression model, a validation accuracy of 90.56% was achieved. 
In contrast, the AutoML approach yielded a higher validation accuracy of 91.75% with the Voting-Ensemble model.

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 90.56%   |
| Voting-Ensemble     | 91.75%   |


The utilization of the AutoML approach led to superior outcomes by employing a comprehensive training and evaluation 
process across multiple models, resulting in the selection of the most effective model for generalizing the data. 
Additionally, the Voting-Ensemble model, with its increased parameter complexity compared to logistic regression, 
offers enhanced capability in capturing the intricate relationships between features and the target variable.

Furthermore, the AutoML approach provides notable benefits through various preprocessing steps, encompassing the 
handling of class imbalance, imputation of missing values, assessment of feature cardinality, and application of 
diverse scaling techniques. These additional preprocessing measures contribute to the improved performance and 
robustness of the models generated by AutoML.

## Future work
Given the highly imbalanced nature of the dataset, alternative performance metrics such as Weighted-AUC and F1-Score can be utilized to evaluate the models. In cases where the Weighted-AUC or F1 score falls short of expectations, techniques like Undersampling, Oversampling, or SMOTE can be explored to address the class imbalance issue.

Another strategy involves tuning the hyperparameters of the Voting-Ensemble model generated through the AutoML approach. 
This includes optimizing the list of estimators, the voting method (e.g., hard or soft), and the weights assigned to 
predicted class labels or probabilities before averaging. By fine-tuning these hyperparameters, the overall performance 
and effectiveness of the Voting-Ensemble model can be enhanced.