# Autoencoder
## Requirements (Important)
A conda environment is provided in "autoencoder/autoencoder_conda_env.yaml" \
Important: tensorflow==2.10.0

https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file \
conda env create -f autoencoder_conda_env.yml

## Config
"config.py" to change the data set between "taxi" and "wind". \
Model parameters can be changed, but last layer in decoder needs to be treated carefully.

## How to run
After installing environment start "main.py".
