# Hybrid BO: flash vessel demo

Code for the poster's flash vessel example, built on
[hybrid-bo](https://github.com/jj2249/hybrid-bo). Problem from Cramer et al. (2026),
[arXiv:2603.11199](https://arxiv.org/abs/2603.11199).

## Reproduce

```bash
pixi install
pixi run regret   # regret plot  -> results/flash_regret/rerun/
pixi run table    # cost table   -> results/flash_regret/rerun/annual_cost.csv
```

The poster's run is in [results/flash_regret/poster/](results/flash_regret/poster/).
