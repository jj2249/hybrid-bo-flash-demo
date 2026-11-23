"""Hybrid BO's GP of gamma1 against x1 at one temperature, at early iterations of a
`flash_regret.py` run: the measured points, the ideal mixture (gamma1 = 1), the GP's
mean with a band of +-2 standard deviations, and the true NRTL curve.

The GP is of ln gamma1 over (T, x1); this is its slice at T_SLICE, the optimum's
temperature, drawn as gamma1 = exp(ln gamma1). x1 runs over [0, z1], where the flash
has two phases: the infeasible experiments' labels, at x1 in [z1, 1], are off it.
Each (seed, iteration) rebuilds its data as `flash_acquisition.py` does. Run it with

    pixi run python studies/flash_gamma.py RUN SEED:ITERATION [SEED:ITERATION ...]

for a run `results/flash_regret/RUN/`; it saves the figures to its `gamma/`.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import flash
from flash import CONFIG, N_STARTS
from flash_acquisition import iteration_data
from flash_regret import IDEAL_COLOR, STYLES, U_OPTIMUM

logging.getLogger("hybrid_bo").setLevel(logging.ERROR)

T_SLICE = U_OPTIMUM[0]


def figure_of(run, seed, iteration):
    """The figure of the GP of gamma1 before `iteration` of `seed`."""
    data, _ = iteration_data(run, seed, iteration)
    rng = np.random.default_rng([seed, iteration])
    fitted = flash.gp.fit(
        data, inputs=[flash.T, flash.x1], output=flash.ln_gamma1, rng=rng,
        n_starts=N_STARTS,
    )
    x1 = np.linspace(0.0, CONFIG[2], 201)
    mean, std = fitted.predict(np.column_stack([np.full_like(x1, T_SLICE), x1]))

    figure, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
    # No background and no titles, as the regret figure
    figure.patch.set_facecolor("none")
    ax.set_facecolor("none")
    color = STYLES["Hybrid BO"]["color"]
    ax.fill_between(
        x1, np.exp(mean - 2 * std), np.exp(mean + 2 * std), color=color, alpha=0.15,
        lw=0,
    )
    ax.plot(x1, np.exp(mean), color=color, lw=2, label="Hybrid BO GP")
    ax.plot(
        x1, np.exp(flash.nrtl_ln_gamma1(T_SLICE, x1)), color="k", lw=1.5, ls="--",
        label="True (NRTL)",
    )
    ax.axhline(
        1.0, color=IDEAL_COLOR, lw=2, alpha=0.7, zorder=0.5,
        label=r"Ideal mixture ($\gamma_1 = 1$)",
    )

    # The feasible experiments: their states are measured, not NaN
    feasible = ~np.isnan(data.x).any(axis=1)
    ax.scatter(
        data[flash.x1][feasible], np.exp(data[flash.ln_gamma1][feasible]), color="k",
        edgecolors="w", s=50, zorder=3, label="measured",
    )
    ax.set_xlim(0.0, CONFIG[2])
    ax.legend(frameon=False)
    return figure, int(feasible.sum()), len(data.f)


if __name__ == "__main__":
    run = Path("results/flash_regret", sys.argv[1])
    folder = run / "gamma"
    folder.mkdir(exist_ok=True)
    for arg in sys.argv[2:]:
        seed, iteration = (int(k) for k in arg.split(":"))
        figure, n_feasible, n = figure_of(run, seed, iteration)
        stem = f"seed{seed:02d}_it{iteration}"
        figure.savefig(folder / f"{stem}.png", dpi=600)
        figure.savefig(folder / f"{stem}.pdf")
        plt.close(figure)
        print(f"Seed {seed}, iteration {iteration}: {n} experiments, {n_feasible} feasible")
