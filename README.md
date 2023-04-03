# PanPep tested on Shuffled Negatives
This repository contains the data and code to test PanPep (Gao et al., 2023) on data with negatives generated by shuffling. If you want to 
use or test PanPep on your own data, we recommend using the original repository (https://github.com/bm2-lab/PanPep).

## Installation
First install PanPep. We suggest installing PyTorch by following the instruction on their official website 
(https://pytorch.org/get-started/locally/). Our tests were run with PyTorch 2.0.0.

On top off the requirements in the PanPep README, we also had to install these additional python packages to run PanPep:
* joblib==1.1.1
* matplotlib==3.5.1
* scikit-learn==1.2.2

Finally, for plotting our ROC-AUC curves, installing this package is required:
* seaborn==0.12.2

## Usage
Although the output files of all our tests are included, they can be reproduced by running these scripts from the 
project root directory.

For the first test (cross-validation with shuffled negatives) run

    bash predict_cross-validation_shuffled-negatives.sh

and for the second test (zero-shot with shuffled negatives) run

    bash predict_zeroshot_shuffled-negatives.sh

To print all performance results and to create the ROC curves run

    python shuffled-negatives_roc-auc.py

To print the data overlap statistics run

    python check_data_overlap.py