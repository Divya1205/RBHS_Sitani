# RPCA_hotspot


## Description: 
Codes for the paper RBHS: https://doi.org/10.1002/prot.26047, a robust principal component analysis based method to predict preotein-protein interaction hot spots.


## Background:
Proteins often exert their function by binding to other cellular partners. The hot spots are key residues for protein-protein binding. Their identification may shed light on the impact of disease associated mutations on protein complexes and help design protein-protein interaction inhibitors for therapy. Unfortunately, current machine learning methods to predict hot spots, suffer from limitations caused by gross errors in the data matrices. This has the codes for a novel data pre-processing pipeline that overcomes this problem by recovering a low rank matrix with reduced noise using Robust Principal Component Analysis. It further trains and validates machine learning classifiers to existing databases to show the predictive power of the method RBHS.




## Data provided: 
train_HB34.xls is the file used for training the classifiers. It contains 313 rows indicating 313 interface residues and 59 columns, where 1-58 columns are features and column 59 is the label values.

test_BID18.xlsx is the independent test set file. It contains 126 rows indicating 126 interface residues and 59 columns, where 1-58 columns are features and column 59 is the label values. 

## Environment file:


## Recommended software:
* Fiji or Napari
* If programming in Python, install scikit-image, opencv, Tensorflow, PyTorch
* If programming in MATLAB, install image processing toolbox
