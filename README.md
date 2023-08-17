# RPCA_hotspot


## Description: 
Codes for the paper RBHS: https://doi.org/10.1002/prot.26047, a robust principal component analysis based method to predict preotein-protein interaction hot spots.


## Background:
Proteins often exert their function by binding to other cellular partners. The hot spots are key residues for protein-protein binding. Their identification may shed light on the impact of disease associated mutations on protein complexes and help design protein-protein interaction inhibitors for therapy. Unfortunately, current machine learning methods to predict hot spots, suffer from limitations caused by gross errors in the data matrices. This has the codes for a novel data pre-processing pipeline that overcomes this problem by recovering a low rank matrix with reduced noise using Robust Principal Component Analysis. It further trains and validates machine learning classifiers to existing databases to show the predictive power of the method RBHS.




## Data provided: 
 - **Training Data** is at:<br>
["train_HB34.xls"](https://github.com/Divya1205/RBHS_Sitani/blob/master/train_HB34.xls) <br>
train_HB34.xls is the file used for training the classifiers. It contains 313 rows indicating 313 interface residues and 59 columns, where 1-58 columns are features and column 59 is the label values.
- **Test Data** is at:<br>
["test_BID18.xlsx"](https://github.com/Divya1205/RBHS_Sitani/blob/master/test_BID18.xlsx) <br>
test_BID18.xlsx is the independent test set file. It contains 126 rows indicating 126 interface residues and 59 columns, where 1-58 columns are features and column 59 is the label values. 

## Environment file is at:<br>
["HotSpots.yml"](https://github.com/Divya1205/RBHS_Sitani/blob/master/HotSpots.yml) <br>:
All the packages used for RBHS.

## File organisation: <br>

## Generating baseline results using original data and PCA
- **PCA Baseline results** is at :<br>
["PCA_baseline.ipynb"](https://github.com/Divya1205/RBHS_Sitani/blob/master/PCA_baseline.ipynb) <br>:
This jupyter notebook can be used to generate the PCA baseline results that are used to compare the efficacy of RBHS with PCA in the paper https://doi.org/10.1002/prot.26047.

- **Random Forests Classifier Baseline results** is at :<br>
["RandomForestClassifier_Baseline.ipynb"](https://github.com/Divya1205/RBHS_Sitani/blob/master/RandomForestClassifier_Baseline.ipynb) <br>:
This jupyter notebook can be used for training and hyperparameter tuning Random forest classifier on HB-34.xls dataset without any preprocessing. It can then be used for testing the tuned random forest classifier on BID-18.xlsx without any preprocessing done on it.

- **Support Vector Machine Baseline results** is at :<br>
["gridsearchSVM_Baseline.ipynb"](https://github.com/Divya1205/RBHS_Sitani/blob/master/gridsearchSVM_Baseline.ipynb) <br>:
This jupyter notebook can be used for training and hyperparameter tuning SVM classifier on HB-34.xls dataset without any preprocessing. It can then be used for testing the hyperparameter-tuned SVM classifier on BID-18.xlsx without any preprocessing done on it.

- **Gradient Boosting Machine Baseline results** is at :<br>
["tuningGBM_Baseline.ipynb"](https://github.com/Divya1205/RBHS_Sitani/blob/master/tuningGBM_Baseline.ipynb) <br>:
This jupyter notebook can be used for training and hyperparameter tuning GBM classifier on HB-34.xls dataset without any preprocessing. It can then be used for testing the trained GBM classifier on BID-18.xlsx without any preprocessing done on it.

- **Extreme Gradient Boosting Baseline results** is at :<br>
["tuningXGB_B.ipynb"](https://github.com/Divya1205/RBHS_Sitani/blob/master/tuningXGB_B.ipynb) <br>:
This jupyter notebook can be used for training and hyperparameter tuning GBM classifier on HB-34.xls dataset without any preprocessing. It can then be used for testing the trained GBM classifier on BID-18.xlsx without any preprocessing done on it.


## Results for RBHS+classifiers
  
## Generating Plots



## Recommended software:
