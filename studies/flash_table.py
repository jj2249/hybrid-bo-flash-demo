"""The annual cost of the design each method commits to on a trial budget, from a
`flash_regret.py` run.

Each term of the cost f is read as a daily running cost in $k, so f is in $k/day and
a design's annual cost is 365 f. With a budget of N trials (initial points included),
each method commits to its best measured point and runs it for a year; its annual
cost is 365 f and its purity the top stream's water fraction y1, each the median over
the run's seeds, with the optimum's as a reference. LHS is a fresh N-point Latin
hypercube per seed; standard and hybrid BO are the run's first N trials; the ideal
mixture runs no trials. Run it with

    pixi run python studies/flash_table.py RUN [--budgets 10]

It prints the table, and saves it as `annual_cost.csv` in
`results/flash_regret/RUN/`.
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

import flash
import hybrid_bo as hb
from flash import measure, nlp
from flash_regret import F_OPTIMUM, U_OPTIMUM, ideal_mixture_point, slug

logging.getLogger("hybrid_bo").setLevel(logging.ERROR)

DAYS = 365


def print_coefficients():
    """Each cost coefficient as a daily cost, and its term at the optimum."""
    t, p = U_OPTIMUM
    y1 = measure(np.array(U_OPTIMUM), np.random.default_rng(0))[flash.y1][0]
    terms = (
        ("Purity", "(y1 - 0.66)^2", flash.ZETA[0], (y1 - flash.Y1_SET) ** 2),
        ("Temperature", "T^2 (K^2)", flash.ZETA[1], t**2),
        ("Pressure", "(p - 1.013)^2 (bar^2)", flash.ZETA[2], (p - flash.P_ATM) ** 2),
    )
    print("Cost coefficients, read as $k/day per unit of their term\n")
    print("| Term | Per unit of | Coefficient ($k/day) | At the optimum ($/day) |")
    print("|---|---|---|---|")
    for name, unit, zeta, value in terms:
        print(f"| {name} | {unit} | {zeta:g} | {1e3 * zeta * value:,.2f} |")
    print(
        f"\nOptimum: f* = {F_OPTIMUM:.4f} $k/day = ${1e3 * F_OPTIMUM:,.0f}/day = "
        f"${DAYS * F_OPTIMUM:,.1f}k/year\n"
    )


def shown(value):
    """`value` in $k to one decimal, or "< 0.1" if it would round to 0.0."""
    return "< 0.1" if value < 0.05 else f"{value:,.1f}"


def committed(run, method, budget, n_seeds):
    """The f and y1 of the best of the first `budget` trials of each seed, each
    (n_seeds,)."""
    costs, purities = [], []
    for seed in range(n_seeds):
        if method == "LHS":
            rng = np.random.default_rng([seed, budget])
            data = measure(hb.lhs_points(nlp.u, budget, rng), rng)
        else:
            data = hb.Dataset.load(run / f"{slug(method)}_seed{seed:02d}.npz")
        best = np.argmin(data.f[:budget])
        costs.append(data.f[best])
        purities.append(data[flash.y1][best])
    return np.array(costs), np.array(purities)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name")
    parser.add_argument("--budgets", type=int, nargs="+", default=[10])
    args = parser.parse_args()
    run = Path("results/flash_regret", args.name)
    n_seeds = len(list(run.glob("hybrid_bo_seed*.npz")))

    print_coefficients()

    # Each row: the method, and (annual cost, purity) at each budget
    rng = np.random.default_rng(0)
    ideal = measure(ideal_mixture_point(rng), rng)
    optimum = measure(np.array(U_OPTIMUM), rng)
    rows = [
        ("Ideal mixture", [(DAYS * ideal.f[0], ideal[flash.y1][0])] * len(args.budgets))
    ]
    for method in ("LHS", "Standard BO", "Hybrid BO"):
        values = []
        for n in args.budgets:
            costs, purities = committed(run, method, n, n_seeds)
            values.append((DAYS * np.median(costs), np.median(purities)))
        rows.append((method, values))
    rows.append((
        "Optimum (reference)",
        [(DAYS * F_OPTIMUM, optimum[flash.y1][0])] * len(args.budgets),
    ))

    header = [
        column
        for n in args.budgets
        for column in (f"Annual cost, {n} trials ($k)", f"Purity y1, {n} trials")
    ]
    print(
        f"Median annual cost and purity (spec {flash.Y1_SET}) of the committed design, "
        f"{n_seeds} seeds\n"
    )
    print("| Method | " + " | ".join(header) + " |")
    print("|---|" + "---|" * len(header))
    for method, values in rows:
        cells = [c for cost, purity in values for c in (shown(cost), f"{purity:.3f}")]
        print(f"| {method} | " + " | ".join(cells) + " |")

    with (run / "annual_cost.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Method", *header])
        for method, values in rows:
            writer.writerow([method, *(f"{v:.6g}" for pair in values for v in pair)])
