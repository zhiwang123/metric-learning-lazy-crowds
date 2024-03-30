### Metric Learning from Limited Pairwise Preference Comparisons

Authors: Zhi Wang, Geelon So, and Ramya Korlakai Vinayak<br>
arXiv: <https://arxiv.org/abs/2403.19629>

This implementation is based on the paper code for ["One for All: Simultaneous Metric and Preference Learning over Multiple Users"](https://arxiv.org/pdf/2207.03609.pdf) by Gregory Canal, Blake Mason, Ramya Korlakai Vinayak, and Robert Nowak. Our code, in part, was adapted from their [repository](https://github.com/gregcanal/multiuser-metric-preference).

---
#### Requirements
To set up the required environment with conda, you can use the provided `environment.yml` file:
```
conda env create -f environment.yml
conda activate env
```
Alternatively, you can manually install the following required packages:
- python 3.9.17
- numpy 1.22.4
- scipy 1.7.1
- cvxpy 1.2.0
- scikit-learn 1.0
- matplotlib 3.4.3
- seaborn 0.12.2
- pyyaml 6.0

---
#### File Descriptions
- `basic_simulation.py` serves as the base script for running a single simulation. Given experiment parameters, 
it generates synthetic data, runs our divide-and-conquer algorithm to learn metric(s), and returns the relative error(s).
- `multiprocess_runner.py` is the **main script** for the three experiments discussed in Section 6 of the paper. It uses multiprocessing to run simulations in parallel and record results for varying parameters, such as the number of users per subspace and the number of comparisons per user. 
- `approx_subspace_run.py` contains a script for running Experiment 3, which calls `multiprocess_runner.py` for each level of approximate subspace noise.
- `data_generation.py` contains functions for generating metrics, subspaces, items along with their low-dimensional representations, user ideal points and user binary responses.
- `metric_learning.py` implements the two stages in our divide-and-conquer algorithm: learning subspace metrics and reconstructing the full metric from subspace metrics.
- `utils.py` contains helper functions.
- `analysis/` contains scripts for aggregating and plotting results.

---
#### Usage
We have included example configuration files for running experiments.
Below are example usages for Experiment 1 and Experiment 2:
```
python multiprocess_runner.py --config examples/configs/exp1.yaml 
python multiprocess_runner.py --config examples/configs/exp2.yaml 
```
Below is an example usage for Experiment 3 with configuration files in `examples/configs/exp3/`:
```
python approx_subspace_run.py
```
Results and logs are saved in `examples/exp_results`. Use `python analysis/exp1_analysis.py` and `python analysis/exp2_analysis.py` and `python analysis/exp3_analysis.py` to generate plots (certain parameters including file paths in the analysis scripts may need to be adjusted).

To reproduce the results presented in the paper, switch to the branch `compare_logistic_hinge`. This branch contains slightly modified code, wherein the learner directly compares the performance of the logistic loss (with loss_param = 1) and the hinge loss when learning subspace metrics on the same generated data. See also example configuration files therein.