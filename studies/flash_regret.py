"""Regret of three methods on the flash unit of `flash.py`, over several seeds, and of
the ideal mixture.

1. LHS: one Latin hypercube over (T, p) of the whole budget, in its random order.
2. Standard BO: a GP of f over (T, p), and SAA-SoftLogEI.
3. Hybrid BO: a GP of ln gamma1 over (T, x1), and SAA-SoftLogEI, as `flash.py` does.

The ideal mixture is the hybrid NLP with ln gamma1 = 0 (Raoult's law), the hybrid GP's
prior mean, never learned: its minimizer is measured once, and its regret drawn as a
line.

The BO methods differ only in their model: at each seed they start from the same
N_INITIAL Latin hypercube points and share the SAA scenarios, and all minimize
-SoftLogEI at TEMPERATURE. Regret is the incumbent's f minus the oracle optimum
F_OPTIMUM. Run it with

    pixi run python studies/flash_regret.py NAME [--seeds 20] [--iterations 15]

and redraw a run's figure from its saved regrets, cut to its first iterations, with

    pixi run python studies/flash_regret.py NAME --replot [--iterations 15]

It runs the seeds in parallel processes, and saves each run's dataset, the regrets
and a figure of the median regret to `results/flash_regret/NAME/`.
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

import flash
import hybrid_bo as hb
from flash import CONFIG, N_INITIAL, N_SCENARIOS, N_STARTS, TEMPERATURE, measure, nlp
from hybrid_bo.solvers import NLP
from hybrid_bo.utils import Symbolic
from hybrid_bo.variables import start_bounds

logging.getLogger("hybrid_bo").setLevel(logging.WARNING)

# The oracle optimum, at T = 377.2633 K and p = 0.991199 bar, which no method sees:
# a 201 x 201 grid of the true f over (T, p), refined by Nelder-Mead
F_OPTIMUM = 0.14257656504369193
U_OPTIMUM = (377.26330368, 0.99119916)

METHODS = ("LHS", "Standard BO", "Hybrid BO")

# The ideal mixture: ln gamma1 a mechanistic state with the known equation ln gamma1 = 0,
# and no unknown states
ln_gamma1_ideal = hb.Var("ln_gamma1", start=(-1.0, 1.0))
ideal_nlp = hb.HybridNLP(
    u=nlp.u,
    c=nlp.c,
    x=[*nlp.x, ln_gamma1_ideal],
    y=[],
    g=lambda v: [*flash.known_equations(v), v.ln_gamma1],
    f=flash.cost,
)

# Standard BO: f is the one unknown state, of u alone, with flash.gp's prior
f_measured = hb.Var("f_measured", start=(0.0, 50.0))
standard_nlp = hb.HybridNLP(u=nlp.u, y=[f_measured], f=lambda v: v.f_measured)


def as_standard(data):
    """The dataset of standard BO: u, and f as its one unknown state."""
    n = len(data.f)
    return hb.Dataset(
        data.u,
        np.zeros((n, 0)),
        np.zeros((n, 0)),
        data.f[:, np.newaxis],
        data.f,
        tuple(var.name for var in (*standard_nlp.u, *standard_nlp.y)),
    )


def ideal_mixture_point(rng):
    """The u (2,) minimizing f of `ideal_nlp`, by multistart IPOPT: hb.SAA can't solve
    a problem with no unknown states, so this builds its NLP over (u, x) itself."""
    variables = [*ideal_nlp.u, *ideal_nlp.x]
    n_u, n_x = len(ideal_nlp.u), len(ideal_nlp.x)
    w = Symbolic.sym("w", n_u + n_x)
    u, x = w[:n_u], w[n_u:]
    y, c = Symbolic(0, 1), Symbolic(CONFIG)
    problem = NLP(
        cas.Function("f", [w], [ideal_nlp.f(x, y, u, c)]),
        np.array([var.lower for var in variables]),
        np.array([var.upper for var in variables]),
        cas.Function("g", [w], [ideal_nlp.g(x, y, u, c)]),
        start_bounds(variables),
    )
    best, _ = problem.solve(rng, n_starts=N_STARTS)
    return best[:n_u]


def run_seed(seed, n_iterations):
    """The dataset of each method at `seed`, N_INITIAL + n_iterations points each."""
    rng = np.random.default_rng(seed)
    initial = measure(hb.lhs_points(nlp.u, N_INITIAL, rng), rng)
    saa = hb.SAA(rng.standard_normal((N_SCENARIOS, 1)))  # shared by the BO methods
    rngs = dict(zip(METHODS, rng.spawn(len(METHODS)), strict=True))
    runs = {}

    method_rng = rngs["LHS"]
    points = hb.lhs_points(nlp.u, N_INITIAL + n_iterations, method_rng)
    runs["LHS"] = measure(points, method_rng)

    # With no mechanistic states, y is substituted: f depends on u alone
    method_rng, data = rngs["Standard BO"], initial
    for _ in range(n_iterations):
        surrogate = [
            flash.gp.fit(
                as_standard(data),
                inputs=nlp.u,
                output=f_measured,
                rng=method_rng,
                n_starts=N_STARTS,
            )
        ]
        solution = standard_nlp.solve(
            saa,
            method_rng,
            surrogate=surrogate,
            objective=hb.obj.SoftLogEI(data.incumbents[-1], temperature=TEMPERATURE),
            n_starts=N_STARTS,
            unknown_states_as_variables=False,
        )
        data = data.extend(measure(solution.u, method_rng))
    runs["Standard BO"] = data

    method_rng, data = rngs["Hybrid BO"], initial
    for _ in range(n_iterations):
        surrogate = [
            flash.gp.fit(
                data,
                inputs=[flash.T, flash.x1],
                output=flash.ln_gamma1,
                rng=method_rng,
                n_starts=N_STARTS,
            )
        ]
        solution = nlp.solve(
            saa,
            method_rng,
            surrogate=surrogate,
            c=CONFIG,
            objective=hb.obj.SoftLogEI(data.incumbents[-1], temperature=TEMPERATURE),
            n_starts=N_STARTS,
        )
        data = data.extend(measure(solution.u, method_rng))
    runs["Hybrid BO"] = data

    for method, data in runs.items():
        print(f"Seed {seed}, {method}: regret {data.incumbents[-1] - F_OPTIMUM:.3g}")
    return runs


# The reference palette's first four categorical slots, and a marker each, so a
# method is never told by color alone; the ideal mixture is a line with no markers
IDEAL_COLOR = "#eb6834"
STYLES = {
    "LHS": {"color": "#2a78d6", "marker": "o"},
    "Standard BO": {"color": "#1baf7a", "marker": "^"},
    "Hybrid BO": {"color": "#eda100", "marker": "D"},
}


def slug(method):
    """The method's name in files: "Standard BO" -> "standard_bo"."""
    return "_".join(method.lower().replace("(", "").replace(")", "").split())


def plot(regrets, ideal_regret):
    """The median regret of each method against the experiments, with its quartiles,
    and the ideal mixture's regret as a line."""
    figure, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
    # No background, so the poster's shows through; no titles, which the poster adds
    figure.patch.set_facecolor("none")
    ax.set_facecolor("none")
    n = next(iter(regrets.values())).shape[1]
    experiments = np.arange(1, n + 1)
    for method, regret in regrets.items():
        # Regret is >= 0; a floor keeps an exact hit on the log axis
        regret = np.maximum(regret, 1e-12)
        q25, median, q75 = np.quantile(regret, [0.25, 0.5, 0.75], axis=0)
        style = STYLES[method]
        ax.fill_between(experiments, q25, q75, color=style["color"], alpha=0.15, lw=0)
        ax.plot(experiments, median, lw=2, ms=5, label=method, **style)
    # Behind the methods' bands and lines
    ax.axhline(
        ideal_regret,
        color=IDEAL_COLOR,
        lw=2,
        alpha=0.7,
        zorder=0.5,
        label=r"Ideal mixture ($\gamma_1 = 1$)",
    )
    ax.axvline(N_INITIAL + 0.5, color="0.6", lw=1, ls="--")
    ax.text(N_INITIAL + 0.6, 0.98, "BO starts", transform=ax.get_xaxis_transform(),
            va="top", fontsize="small", color="0.4")
    ax.set(yscale="log", xlim=(1, n))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # In the empty space between the baselines and hybrid BO
    ax.legend(frameon=False, loc="center right")
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    folder = Path("results/flash_regret", args.name)

    ideal_u = ideal_mixture_point(np.random.default_rng(0))
    ideal_regret = measure(ideal_u, np.random.default_rng(0)).f[0] - F_OPTIMUM
    print(f"Ideal mixture: T = {ideal_u[0]:.4f} K, p = {ideal_u[1]:.4f} bar, "
          f"regret {ideal_regret:.4g}")

    if args.replot:
        # The saved regrets, cut to N_INITIAL + iterations experiments
        n = N_INITIAL + args.iterations
        with np.load(folder / "regrets.npz") as saved:
            regrets = {m: saved[slug(m)][:, :n] for m in METHODS}
        figure = plot(regrets, ideal_regret)
        figure.savefig(folder / "regret.png", dpi=600)
        figure.savefig(folder / "regret.pdf")
        print(f"Redrew {folder} over {n} experiments")
        raise SystemExit

    # The study's folder, made first so a name in use fails before the run
    folder.mkdir(parents=True)

    seeds = range(args.seeds)
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_seed, seeds, [args.iterations] * args.seeds))

    regrets = {}
    for method in METHODS:
        for seed, runs in zip(seeds, results, strict=True):
            runs[method].save(folder / f"{slug(method)}_seed{seed:02d}.npz")
        regrets[method] = np.array(
            [runs[method].incumbents - F_OPTIMUM for runs in results]
        )
    np.savez(
        folder / "regrets.npz",
        ideal_mixture=ideal_regret,
        **{slug(m): r for m, r in regrets.items()},
    )
    # A PNG for screens and a PDF, vector, for print
    figure = plot(regrets, ideal_regret)
    figure.savefig(folder / "regret.png", dpi=600)
    figure.savefig(folder / "regret.pdf")
    print(f"Saved to {folder}")
