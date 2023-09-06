# Equal Long-term Benefit Rate (ELBERT)

## Environment set-up
First, install [Anaconda](https://docs.anaconda.com/anaconda/install/) to set up virtual environment. Then, run:
```
conda env create -f elbert.yaml
conda activate elbert
pip install -r requirements.txt
```

## Running ELBERT and other baselines 
The *.scripts/* folder includes bash scripts for ELBERT and other baselines (G-PPO, R-PPO, A-PPO) in five enviroments:
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
bash scripts/infectious_original_env_elbert.sh
```
* Baseline (G-PPO, R-PPO, A-PPO)
```
bash scripts/infectious_original_env_original.sh
```
### Infectious control, harder version
* ELBERT
```
bash scripts/infectious_harder_env_elbert.sh
```
* Baseline (G-PPO, R-PPO, A-PPO)
```
bash scripts/infectious_harder_env_original.sh
```
### Attention allocation, orginal version
* ELBERT
```
bash scripts/attention_original_env_elbert.sh
```
* Baseline (G-PPO, R-PPO, A-PPO)
```
bash scripts/attention_original_env_original.sh
```
### Attention allocation, harder version
* ELBERT
```
bash scripts/attention_harder_env_elbert.sh
```
* Baseline (G-PPO, R-PPO, A-PPO)
```
bash scripts/attention_harder_env_original.sh
```


## Comments
Our codebase is based on the following repo:
* [Policy Optimization with Constraint Advantage Regularization](https://github.com/ericyangyu/pocar)
* [Google ML-fairness-gym](https://github.com/google/ml-fairness-gym)

Thanks for their efforts to make the code public!
