# Exploring Inductive Transfer Approaches for Machine Learning Based Rainfall-Runoff Modeling
The code in this repository was utilized to generate all the results presented in our paper.
## Repository content
- ```augmentation/``` -- Folder with augmentation methods
- ```model/``` -- Folder containing model used
- ```training/``` -- Folder for model training and validation
- ```evaluation``` -- Folder for model evaluation
- ```environment.yml``` -- Conda environment used to train and evaluate the models
- ```config``` -- configuration file
## Required configuration 
1. Create a conda environment \
`conda env create -f environment_name.yml`
2. Environment activation\
`conda activate pytorch`
3. Install the neuralhydrology package
`pip install`
## Model training and evaluation
Follow the following steps for model training/inference/evaluation:\
1. Activate the conda envvironent\
`conda activate pytorch`
2. Training start\
`nh-run train --config-file config.yml`
3. Inference/evaluation start\
`nh-run evaluate --run-dir <run directory>`
<!-- ## Contact -->

## Citation