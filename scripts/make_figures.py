#!/usr/bin/env python3
"""Regenerate the result figures (as in the review deck) from the coordinator's database.

    python scripts/make_figures.py --modes-study study_ab12cd34 --privacy-study study_ef56gh78 --out figures/

Produces PNG files at 200 dpi:
    fig_main_comparison.png   final accuracy per method (mean ± sd over seeds)
    fig_convergence.png       mean accuracy per round for every method
    fig_per_client.png        final accuracy per campus for every method
    fig_privacy_tradeoff.png  accuracy vs privacy level (epsilon) for the methods in the privacy study
    fig_trust.png             trust assigned to each campus by round (proposed method)
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABEL = {"local_only": "Local only", "uniform_fd": "Uniform FD", "trust_fixed": "Trust + fixed λ",
         "trust_adaptive": "Proposed (trust + adaptive)"}
COLOR = {"local_only": "#8797a3", "uniform_fd": "#d9822b", "trust_fixed": "#1b9aaa", "trust_adaptive": "#0e8c7f"}
ORDER = ["local_only", "uniform_fd", "trust_fixed", "trust_adaptive"]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "figure.dpi": 100})


def load_study(db: sqlite3.Connection, sid: str):
    rows = db.execute("SELECT id, tags, summary, status FROM experiments WHERE study_id=?", (sid,)).fetchall()
    if not rows:
        raise SystemExit(f"no experiments found for {sid}")
    out = []
    for eid, tags, summary, status in rows:
        if status != "completed":
            continue
        rounds = [json.loads(r[0]) for r in db.execute(
            "SELECT data FROM rounds WHERE experiment_id=? ORDER BY round", (eid,))]
        out.append({"id": eid, "tags": json.loads(tags), "summary": json.loads(summary), "rounds": rounds})
    return out


def modes_figures(exps, out):
    by = defaultdict(list)
    for e in exps:
        by[e["tags"]["mode"]].append(e)
    modes = [m for m in ORDER if m in by]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    means = [np.mean([e["summary"]["final_mean_acc"] for e in by[m]]) * 100 for m in modes]
    sds = [np.std([e["summary"]["final_mean_acc"] for e in by[m]]) * 100 for m in modes]
    mins = [np.mean([e["summary"]["final_min_acc"] for e in by[m]]) * 100 for m in modes]
    x = np.arange(len(modes))
    ax.bar(x - .2, means, .4, yerr=sds, capsize=4, color=[COLOR[m] for m in modes], label="Mean campus accuracy")
    ax.bar(x + .2, mins, .4, color=[COLOR[m] for m in modes], alpha=.45, label="Weakest campus")
    ax.set_xticks(x, [LABEL[m] for m in modes], rotation=12)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Final accuracy by method")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_main_comparison.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in modes:
        curves = np.array([[r["mean_acc"] for r in e["rounds"]] for e in by[m]]) * 100
        mu, sd = curves.mean(0), curves.std(0)
        xs = np.arange(len(mu))
        ax.plot(xs, mu, color=COLOR[m], lw=2.2, marker="o", ms=4, label=LABEL[m])
        ax.fill_between(xs, mu - sd, mu + sd, color=COLOR[m], alpha=.12)
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Mean test accuracy (%)")
    ax.set_title("Convergence across rounds")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_convergence.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    first = by[modes[0]][0]["rounds"][-1]["campuses"]
    names = [c["name"] for c in first]
    w = .8 / len(modes)
    for j, m in enumerate(modes):
        acc = np.array([[c["acc"] for c in e["rounds"][-1]["campuses"]] for e in by[m]]) * 100
        ax.bar(np.arange(len(names)) + j * w - .4 + w / 2, acc.mean(0), w, yerr=acc.std(0), capsize=2,
               color=COLOR[m], label=LABEL[m])
    ax.set_xticks(np.arange(len(names)), [f"{n}\n({c['model']})" for n, c in zip(names, first)])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Per-campus accuracy")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_per_client.png"), dpi=200)
    plt.close(fig)

    if "trust_adaptive" in by:
        runs = by["trust_adaptive"]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        k = len(runs[0]["rounds"][-1]["campuses"])
        for i in range(k):
            tr = np.array([[r["campuses"][i]["trust"] for r in e["rounds"][1:]] for e in runs]) * 100
            ax.plot(np.arange(1, tr.shape[1] + 1), tr.mean(0), lw=2.2, marker="o", ms=4,
                    label=runs[0]["rounds"][-1]["campuses"][i]["name"])
        ax.set_xlabel("Federated round")
        ax.set_ylabel("Trust share (%)")
        ax.set_title("Trust assigned to each campus (proposed method)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_trust.png"), dpi=200)
        plt.close(fig)


def privacy_figure(exps, out):
    by = defaultdict(lambda: defaultdict(list))
    for e in exps:
        by[e["tags"]["mode"]][str(e["tags"]["vary_value"])].append(e["summary"]["final_mean_acc"] * 100)
    vals = sorted({v for m in by.values() for v in m}, key=lambda v: (v == "off", float(v) if v != "off" else 0))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in [m for m in ORDER if m in by]:
        mu = [np.mean(by[m][v]) for v in vals]
        sd = [np.std(by[m][v]) for v in vals]
        ax.errorbar(range(len(vals)), mu, yerr=sd, color=COLOR[m], lw=2.2, marker="o", capsize=3, label=LABEL[m])
    ax.set_xticks(range(len(vals)), ["no DP" if v == "off" else f"ε={float(v):g}" for v in vals])
    ax.set_xlabel("Privacy level per release (smaller ε = more noise)")
    ax.set_ylabel("Mean test accuracy (%)")
    ax.set_title("Privacy–accuracy trade-off")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_privacy_tradeoff.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="instance/fedtrust.db")
    ap.add_argument("--modes-study", help="id of a 'Method comparison' study")
    ap.add_argument("--privacy-study", help="id of a 'Privacy level sweep' study")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    if not (a.modes_study or a.privacy_study):
        ap.error("give --modes-study and/or --privacy-study (ids are shown on the Studies page)")
    os.makedirs(a.out, exist_ok=True)
    db = sqlite3.connect(a.db)
    if a.modes_study:
        modes_figures(load_study(db, a.modes_study), a.out)
    if a.privacy_study:
        privacy_figure(load_study(db, a.privacy_study), a.out)
    print("wrote figures to", a.out, "->", ", ".join(sorted(os.listdir(a.out))))


if __name__ == "__main__":
    main()
