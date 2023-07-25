# Equal Long-term Benefit Rate (ELBERT)

## Environment set-up
First, install [Anaconda](https://docs.anaconda.com/anaconda/install/) to set up virtual environment. Then, run:
```
conda create -n pocar python=3.8
conda activate pocar
pip install -r requirements.txt
```

## Running ELBERT and other baselines 
The *.scripts/* folder includes bash scripts for ELBERT and other baselines in five enviroments:
### Lending
* ELBERT
```
bash scripts/lending_elbert.sh
```
* Baseline (G-PPO, R-PPO, A-PPO)
```
bash scripts/lending_original.sh
```
### Infectious control, orginal version
* ELBERT
```
bash scripts/infectious_orienv_elbert.sh
```
* Baseline (G-PPO, R-PPO, A-PPO)
```
bash scripts/infectious_orienv_original.sh
```
### Infectious control, harder version
* ELBERT
```
bash scripts/infectious_newenv_elbert.sh
```
* Baseline (G-PPO, R-PPO, A-PPO)
```
bash scripts/infectious_newenv_original.sh
```
### Attention allocation, orginal version
* ELBERT
```
bash scripts/attention_orienv_elbert.sh
```
* Baseline (G-PPO, R-PPO, A-PPO)
```
bash scripts/attention_orienv_original.sh
```
### Attention allocation, harder version
* ELBERT
```
bash scripts/attention_newenv_elbert.sh
```
* Baseline (G-PPO, R-PPO, A-PPO)
```
bash scripts/attention_newenv_original.sh
```


## Original code
Authors: Eric Yang Yu, Zhizhen Qin, Min Kyung Lee, Sicun Gao \
NeurIPS (Conference on Neural Information Processing Systems) 2022

This is the code implementation for the NeurIPS 2022 [paper](https://arxiv.org/abs/2210.12546) above. 
Code and environments are adapted from the original Google ML-fairness-gym [repo](https://github.com/google/ml-fairness-gym).
