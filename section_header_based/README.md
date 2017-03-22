# section header based structure identification

In this directory, scripts are used to do model training and prediction. 

## experiments
Two methods will be used.

1. Word feature based classification -- based on SVM and word feature.
2. sequence labeling -- CRF based.

#### Train and test models
Train two models: SVM and CRF with following command.

###### To SVM:
    
    python headers_based_trainer.py train -i '../data/sc_headers.csv' -m SVM -o 'models/svm'

trained models will be saved as 'models/svm-model.pkl' and 'models/svm-vec.pkl'.

###### To CRF:

    python headers_based_trainer.py train -i '../data/sc_headers.csv' -m CRF -o 'models/crf'

trained models will be saved as 'models/crf-model.pkl'.

#### Predict new samples
Using these two trained models to predict labels. The line format of input file is:

    doi1,header1,0
    doi1,header2,1
    doi1,header3,2
    doi1,header4,3
    doi1,header5,4

    doi2,header1,0
    doi2,header2,1
    doi2,header3,2
    doi2,header4,3
    doi2,header5,4

The data of two different articles are separated by a blank line.

###### To SVM:
    
     python headers_based_trainer.py predict -i samples.txt -m SVM -o 'models/svm'

###### To CRF:
    
    python headers_based_trainer.py predict -i samples.txt -m CRF -o 'models/crf'

The results will be printed in console.













