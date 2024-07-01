# Contextualized Hybrid Ensemble Q-learning (CHEQ)

***

Code accompanying the paper **"Contextualized Hybrid Ensemble Q-learning:
Learning Fast with Control Priors".**

## Installation
We use [Conda]((https://conda.io/projects/conda/en/latest/user-guide/install/index.html)) to create a Python environment in which the code can be executed.

Run:
````
conda env create --file environment.yml
conda activate cheq-env
````

## Usage
The training for CHEQ and baselines can be started by executing **main.py** in the **code** directory, e.g. for starting the training for CHEQ-UTD20:
````
python main.py -algo "cheq" -G 20
````

For information about the arguments you can pass to main.py you can simply run
````
python main.py --help
````
to display a help page explaining the usage.

## Logging
The logging of the runs is done with [Weights and Biases](https://wandb.ai). It is necessary to create an account and perform an intial login as described on their website.