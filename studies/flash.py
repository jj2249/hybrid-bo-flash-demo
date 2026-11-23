"""The flash unit of Cramer et al. (2026), arXiv:2603.11199: hybrid BO with SoftLogEI.

A feed F of water (1) and acetic acid (2) at mole fraction z1 is split into a top
stream D and a bottom stream B at temperature T and pressure p. The mass balances,
Antoine's equation and Raoult's law are known; water's activity coefficient,
ln gamma1 = h(T, x1), is not, and a GP learns it. The oracle is the NRTL model.

Each experiment sets T and p, and measures the liquid's water fraction x1; the known
equations then give B, y1, p1sat and ln gamma1. An experiment with x1 > z1 is
infeasible (a single phase): the paper labels it ln gamma1 = 0 at a random
x1 ~ U[z1, 1], so the GP sees the infeasible region.

This runs hybrid BO with SAA-SoftLogEI: N_INITIAL initial points, then N_ITERATIONS
iterations, each refitting the GP and measuring the point that minimizes -SoftLogEI.
Run it with

    pixi run python studies/flash.py NAME

It prints each iteration, and saves the measured points and a figure of each iteration
to `results/flash/NAME/`.
"""

import logging
import sys
from pathlib import Path

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import hybrid_bo as hb

logging.basicConfig()
logging.getLogger("hybrid_bo").setLevel(logging.INFO)

SEED = 2
N_INITIAL = 3  # Latin hypercube points over (T, p), as in the paper
N_SCENARIOS = 25  # SAA scenarios, as in the paper
N_STARTS = 100  # multistarts of the GP fit and the acquisition, as in the paper
N_ITERATIONS = 7
# SoftLogEI's temperature, in units of f: small next to the f near the optimum (the
# temperature and pressure terms, ~0.1-1), so it smooths max(a, 0) only close to 0
TEMPERATURE = 1e-3

# Parameters (the paper's Table 2)
Y1_SET = 0.66  # purity specification of y1
P_ATM = 1.01325  # ambient pressure, bar
# Costs of purity violation ($), temperature ($/K^2) and pressure ($/bar^2). The
# temperature cost is the authors' code's 1e-6, not the 0.01 of the paper's Table 2,
# which would swamp the purity term
ZETA = (100.0, 1e-6, 0.5)
CONFIG = np.array([1.0, 0.3, 0.5])  # F, D (mol/s) and z1

# The parameters below are the authors' code's: examples/flash_unit/custom_problem.py
# of https://gitlab.git.nrw/rwth-avt-svt/public/hybrid-bo

# Antoine Coefficient A of H2O, to p in Pa and T in Kelvin (-)
A1 = 10.19590302
# Antoine Coefficient B of H2O, to p in Pa and T in Kelvin (-)
B1 = 1730.6
# Antoine Coefficient C of H2O, to p in Pa and T in Kelvin (-)
C1 = -39.75

# log10(p1sat / Pa) = A - B / (T / K + C); p and p1sat are in bar here
ANTOINE = (A1, B1, C1)
PA_PER_BAR = 1e5

# NRTL parameters of water (1) and acetic acid (2), from Aspen Properties: the
# oracle's, never seen by the method
NRTL = {
    "a12": 2.28422,  # NRTL Aspen Parameter aij of H20, CH3COOH (-)
    "a21": -1.56707,  # NRTL Aspen Parameter aji of CH3COOH, H2O (-)
    "b12": -478.146,  # NRTL Aspen Parameter bij of H2O, CH3COOH (K)
    "b21": 542.84,  # NRTL Aspen Parameter bji of CH3COOH, H2O (K)
    "c12": 0.373484,  # NRTL Aspen Parameter cij of H2O, CH3COOH (-)
    "c21": 0.373484,  # NRTL Aspen Parameter cji of CH3COOH, H2O (-)
    "d12": 0.0,  # NRTL Aspen Parameter dij of H2O, CH3COOH (1/K)
    "d21": 0.0,  # NRTL Aspen Parameter dji of CH3COOH, H2O (1/K)
    "e12": 0.0,  # NRTL Aspen Parameter eij of H2O, CH3COOH (-)
    "e21": 0.0,  # NRTL Aspen Parameter eji of CH3COOH, H2O (-)
    "f12": 0.0,  # NRTL Aspen Parameter fij of H2O, CH3COOH (-)
    "f21": 0.0,  # NRTL Aspen Parameter fji of CH3COOH, H2O (-)
}

# What we know: decision variables u, configuration c, mechanistic states x, unknown y
T = hb.Var("T", lower=363.15, upper=403.15)
p = hb.Var("p", lower=0.8, upper=2.6)
F, D, z1 = hb.Var("F"), hb.Var("D"), hb.Var("z1")
B = hb.Var("B", lower=0.0, upper=10.0)
x1 = hb.Var("x1", lower=0.0, upper=1.0)
y1 = hb.Var("y1", lower=0.0, upper=1.0)
p1sat = hb.Var("p1sat", lower=0.1, upper=10.0)
ln_gamma1 = hb.Var("ln_gamma1", start=(-1.0, 1.0))


def known_equations(v):
    """The paper's Eq. 14: 0 = g(x, y, u, c)."""
    a, b, c = ANTOINE
    return [
        v.D + v.B - v.F,
        v.D * v.y1 + v.B * v.x1 - v.F * v.z1,
        a - b / (c + v.T) - cas.log10(v.p1sat * PA_PER_BAR),
        v.p * v.y1 - v.p1sat * v.x1 * cas.exp(v.ln_gamma1),
    ]


def cost(v):
    """The paper's Eq. 13."""
    return (
        ZETA[0] * (v.y1 - Y1_SET) ** 2
        + ZETA[1] * v.T**2
        + ZETA[2] * (v.p - P_ATM) ** 2
    )


nlp = hb.HybridNLP(
    u=[T, p],
    c=[F, D, z1],
    x=[B, x1, y1, p1sat],
    y=[ln_gamma1],
    g=known_equations,
    f=cost,
)


# The oracle: the flash with NRTL, which the methods never see
def nrtl_ln_gamma1(t, x):
    """ln gamma1 of water at temperature t (K) and water fraction x, by Aspen's NRTL."""
    tau12 = NRTL["a12"] + NRTL["b12"] / t + NRTL["e12"] * np.log(t) + NRTL["f12"] * t
    tau21 = NRTL["a21"] + NRTL["b21"] / t + NRTL["e21"] * np.log(t) + NRTL["f21"] * t
    alpha12 = NRTL["c12"] + NRTL["d12"] * (t - 273.15)
    alpha21 = NRTL["c21"] + NRTL["d21"] * (t - 273.15)
    g12, g21 = np.exp(-alpha12 * tau12), np.exp(-alpha21 * tau21)
    x2 = 1 - x
    return x2**2 * (
        tau21 * (g21 / (x + x2 * g21)) ** 2 + tau12 * g12 / (x2 + x * g12) ** 2
    )


def oracle_x1(t, pressure, config):
    """The water fraction x1 of the liquid at (t, pressure), or None if no x1 gives
    0 <= y1 <= 1."""
    feed, top, feed_water = config
    bottom = feed - top
    a, b, c = ANTOINE
    saturation = 10 ** (a - b / (c + t)) / PA_PER_BAR

    def residual(x):
        top_water = (feed * feed_water - bottom * x) / top
        return pressure * top_water - saturation * x * np.exp(nrtl_ln_gamma1(t, x))

    # y1 = 1 and y1 = 0 bracket x1, within [0, 1]
    low = max(0.0, (feed * feed_water - top) / bottom)
    high = min(1.0, feed * feed_water / bottom)
    if residual(low) * residual(high) > 0:
        return None
    return brentq(residual, low, high, xtol=1e-14)


def measure(u, rng):
    """The dataset of the experiments at u (n, 2): each feasible one measured x1, and
    each infeasible one (x1 > z1, or none) labelled ln gamma1 = 0 at x1 ~ U[z1, 1],
    with the other states NaN and f at y1 = 0, as the authors' code does."""
    names = tuple(var.name for var in (*nlp.u, *nlp.c, *nlp.x, *nlp.y))
    data = None
    for row in np.atleast_2d(u):
        measured = oracle_x1(*row, CONFIG)
        if measured is not None and measured <= CONFIG[2]:
            new = hb.Dataset.from_measurements(
                nlp, u=row, measured={x1: measured}, c=CONFIG, rng=rng
            )
        else:
            states = np.full((1, len(nlp.x)), np.nan)
            states[0, nlp.x.index(x1)] = rng.uniform(CONFIG[2], 1.0)
            t, pressure = row
            f = (
                ZETA[0] * Y1_SET**2
                + ZETA[1] * t**2
                + ZETA[2] * (pressure - P_ATM) ** 2
            )
            new = hb.Dataset(
                row[np.newaxis],
                CONFIG[np.newaxis],
                states,
                np.zeros((1, 1)),
                np.array([f]),
                names,
            )
        data = new if data is None else data.extend(new)
    return data


def print_data(data, title):
    """A row per experiment: u, the states and f."""
    print(f"\n{title}")
    columns = (T, p, B, x1, y1, p1sat, ln_gamma1)
    print("".join(f"{var.name:>11}" for var in columns) + f"{'f':>11}")
    for i in range(len(data.f)):
        values = [data[var][i] for var in columns] + [data.f[i]]
        print("".join(f"{value:11.4f}" for value in values))


# The prior GP of ln gamma1 over (T, x1): zero mean, Matern-5/2, no noise
gp = hb.gp.GP(
    kernel=hb.gp.Matern52(lengthscale_prior=True), mean=hb.gp.Zero(), noise_std=0.0
)

if __name__ == "__main__":
    # The study's folder, made first so a name in use fails before the run
    folder = Path("results/flash", sys.argv[1])
    folder.mkdir(parents=True)

    rng = np.random.default_rng(SEED)
    data = measure(hb.lhs_points([T, p], N_INITIAL, rng), rng)
    print_data(data, f"Initial data ({N_INITIAL} points, seed {SEED})")
    saa = hb.SAA(rng.standard_normal((N_SCENARIOS, 1)))  # once, before the loop

    for i in range(N_ITERATIONS):
        surrogate = [
            gp.fit(data, inputs=[T, x1], output=ln_gamma1, rng=rng, n_starts=N_STARTS)
        ]
        solution = nlp.solve(
            saa,
            rng,
            surrogate=surrogate,
            c=CONFIG,
            objective=hb.obj.SoftLogEI(data.incumbents[-1], temperature=TEMPERATURE),
            n_starts=N_STARTS,
        )

        # The panels: E and std of ln gamma1, E and std of f, and the acquisition
        figure, axd = plt.subplot_mosaic(
            [
                ["Ey", "Ey", "sy", "sy"],
                ["Ef", "Ef", "sf", "sf"],
                [".", "acq", "acq", "."]
            ],
            figsize=(10, 11),
            layout="constrained"
        )
        axes = [axd[k] for k in ("Ey", "sy", "Ef", "sf", "acq")]
        hb.plots.iteration(nlp, solution, axes, data)
        figure.suptitle(f"Iteration {i + 1}")
        figure.savefig(folder / f"iteration_{i + 1:02d}.png")
        plt.close(figure)

        data = data.extend(measure(solution.u, rng))
        print(
            f"Iteration {i + 1}: T = {solution.u[0]:.4f} K, p = {solution.u[1]:.4f} "
            f"bar, f = {data.f[-1]:.6g}, incumbent = {data.incumbents[-1]:.6g}"
        )

    print_data(data, "All data")
    data.save(folder / "data.npz")
