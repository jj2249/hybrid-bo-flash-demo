# Hybrid BO: flash vessel demo

Demo code for the poster's flash vessel example, built on our internal library
[hybrid-bo](https://github.com/jj2249/hybrid-bo) at v0.3.0. Problem from Cramer et al. (2026),
[arXiv:2603.11199](https://arxiv.org/abs/2603.11199).

## Commands to reproduce
### Create the environment
```bash
pixi install
```
### Run hybrid Bayesian optimisation
Sample three initial datapoints and run seven BO iterations. The script will write to the terminal as it runs and save plots of the surrogate model and the acquisition surface. You should replace <run-name> with the name of the folder in which you want to write the plots.
```bash
pixi run python studies/flash.py <run-name>
```
### Generate regret plot
```bash
pixi run regret   # regret plot  -> results/flash_regret/rerun/
```
### Generate the cost summary table
```bash
pixi run table    # cost table   -> results/flash_regret/rerun/annual_cost.csv
```

The results used in the poster are in [results/flash_regret/poster/](results/flash_regret/poster/).
