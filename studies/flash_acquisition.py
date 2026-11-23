"""Figures of hybrid BO's acquisition at early iterations of a `flash_regret.py` run.

Each (seed, iteration) rebuilds that iteration from the run's saved hybrid BO data and
its seed's SAA scenarios, and refits the GP of ln gamma1: the fit's multistarts draw
afresh, so the trial point can differ slightly from the run's. A seed the run doesn't
have, such as one whose initial points are all feasible, runs hybrid BO from its own
initial points up to the iteration. Three figures each:

- hybrid: hybrid BO's SoftEI over (T, p)
- vs_standard: standard BO's SoftEI, from a GP of f on the same data, then hybrid's
- vs_true: the true f from NRTL, then hybrid's SoftEI

with the measured points and the true optimum marked. Run it with

    pixi run python studies/flash_acquisition.py RUN SEED:ITERATION [SEED:ITERATION ...]

for a run `results/flash_regret/RUN/`; it saves the figures to its `acquisition/`.
"""

import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import flash
import hybrid_bo as hb
from flash import CONFIG, N_INITIAL, N_SCENARIOS, N_STARTS, TEMPERATURE, measure, nlp
from flash_regret import U_OPTIMUM, as_standard, f_measured, standard_nlp

logging.getLogger("hybrid_bo").setLevel(logging.ERROR)

GRID_POINTS = 201  # along each of T and p, for the true f


def iteration_data(run, seed, iteration):
    """The hybrid BO data before `iteration` of `seed`, and the seed's SAA: from the
    run's saved data, or, for a seed the run doesn't have, by running hybrid BO."""
    rng = np.random.default_rng(seed)
    initial = measure(hb.lhs_points(nlp.u, N_INITIAL, rng), rng)
    saa = hb.SAA(rng.standard_normal((N_SCENARIOS, 1)))
    saved = run / f"hybrid_bo_seed{seed:02d}.npz"
    if not saved.exists():
        data, run_rng = initial, np.random.default_rng([seed, 0])
        for _ in range(iteration - 1):
            solution = hybrid_solution(data, saa, run_rng)
            data = data.extend(measure(solution.u, run_rng))
        return data, saa
    full = hb.Dataset.load(saved)
    n = N_INITIAL + iteration - 1
    data = hb.Dataset(
        full.u[:n], full.c[:n], full.x[:n], full.y[:n], full.f[:n], full.names
    )
    return data, saa


def hybrid_solution(data, saa, rng):
    """Hybrid BO's SoftLogEI solution on `data`."""
    surrogate = [
        flash.gp.fit(
            data, inputs=[flash.T, flash.x1], output=flash.ln_gamma1, rng=rng,
            n_starts=N_STARTS,
        )
    ]
    return nlp.solve(
        saa, rng, surrogate=surrogate, c=CONFIG,
        objective=hb.obj.SoftLogEI(data.incumbents[-1], temperature=TEMPERATURE),
        n_starts=N_STARTS,
    )


def standard_solution(data, saa, rng):
    """Standard BO's SoftLogEI solution on `data`."""
    surrogate = [
        flash.gp.fit(
            as_standard(data), inputs=nlp.u, output=f_measured, rng=rng,
            n_starts=N_STARTS,
        )
    ]
    return standard_nlp.solve(
        saa, rng, surrogate=surrogate,
        objective=hb.obj.SoftLogEI(data.incumbents[-1], temperature=TEMPERATURE),
        n_starts=N_STARTS, unknown_states_as_variables=False,
    )


def true_f():
    """The true f on a grid over (T, p), NaN where the flash is single-phase."""
    t = np.linspace(flash.T.lower, flash.T.upper, GRID_POINTS)
    p = np.linspace(flash.p.lower, flash.p.upper, GRID_POINTS)
    feed, top, z1 = CONFIG
    f = np.full((GRID_POINTS, GRID_POINTS), np.nan)
    for i, pressure in enumerate(p):
        for j, temperature in enumerate(t):
            x1 = flash.oracle_x1(temperature, pressure, CONFIG)
            if x1 is not None and x1 <= z1:
                y1 = (feed * z1 - (feed - top) * x1) / top
                f[i, j] = (
                    flash.ZETA[0] * (y1 - flash.Y1_SET) ** 2
                    + flash.ZETA[1] * temperature**2
                    + flash.ZETA[2] * (pressure - flash.P_ATM) ** 2
                )
    return t, p, f


def mark(ax, data):
    """The measured points and the true optimum on `ax`."""
    ax.scatter(
        data.u[:, 0], data.u[:, 1], color="k", edgecolors="w", s=40, zorder=3,
        label="measured",
    )
    # A ring, so the trial point's star shows through it
    ax.scatter(
        *U_OPTIMUM, facecolors="none", edgecolors="C3", linewidths=2, s=250,
        zorder=4, label="true optimum",
    )


def figures(run, seed, iteration):
    """Saves the three figures of `iteration` of `seed`."""
    data, saa = iteration_data(run, seed, iteration)
    rng = np.random.default_rng([seed, iteration])
    hybrid = hybrid_solution(data, saa, rng)
    standard = standard_solution(data, saa, rng)
    folder = run / "acquisition"
    stem = f"seed{seed:02d}_it{iteration}"

    def save(figure, layout):
        figure.savefig(folder / f"{stem}_{layout}.png", dpi=300)
        figure.savefig(folder / f"{stem}_{layout}.pdf")
        plt.close(figure)

    # Hybrid BO alone
    figure, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    hb.plots.acquisition(nlp, hybrid, ax)
    mark(ax, data)
    ax.set_title("Hybrid BO acquisition")
    ax.legend(fontsize="small", loc="upper right")
    save(figure, "hybrid")

    # Standard BO, then hybrid BO, sharing p. Their acquisitions' scales differ (each
    # its own model's), and what matters is where each peaks, so no color bars: darker
    # is higher
    figure, axes = plt.subplots(
        1, 2, figsize=(8, 4), sharey=True, layout="constrained"
    )
    for ax, problem, solution, title in (
        (axes[0], standard_nlp, standard, "Standard BO"),
        (axes[1], nlp, hybrid, "Hybrid BO"),
    ):
        hb.plots.acquisition(problem, solution, ax)
        mark(ax, data)
        ax.set_title(f"{title} acquisition")
    for color_bar in figure.axes[2:]:
        color_bar.remove()
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    axes[0].legend(fontsize="small", loc="upper right")
    save(figure, "vs_standard")

    # The true f, then hybrid BO
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    t, p, f = true_f()
    filled = axes[0].contourf(t, p, f, levels=10, cmap="Blues")
    axes[0].contour(t, p, f, levels=filled.levels, colors="k", linewidths=0.5, alpha=0.3)
    axes[0].set_facecolor("0.8")
    figure.colorbar(filled, ax=axes[0])
    axes[0].set(title="True cost f (NRTL; gray: single phase)", xlabel="T", ylabel="p")
    mark(axes[0], data)
    hb.plots.acquisition(nlp, hybrid, axes[1])
    mark(axes[1], data)
    axes[1].set_title("Hybrid BO acquisition")
    axes[0].legend(fontsize="small", loc="upper right")
    save(figure, "vs_true")

    print(
        f"Seed {seed}, iteration {iteration}: hybrid trial T = {hybrid.u[0]:.3f} K, "
        f"p = {hybrid.u[1]:.4f} bar; standard trial T = {standard.u[0]:.3f} K, "
        f"p = {standard.u[1]:.4f} bar"
    )


if __name__ == "__main__":
    run = Path("results/flash_regret", sys.argv[1])
    jobs = [tuple(int(k) for k in arg.split(":")) for arg in sys.argv[2:]]
    (run / "acquisition").mkdir(exist_ok=True)
    with ProcessPoolExecutor(min(len(jobs), 10)) as pool:
        list(pool.map(figures, [run] * len(jobs), *zip(*jobs, strict=True)))
