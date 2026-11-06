#!/usr/bin/env python3
"""Run a study against a running coordinator, without the browser.

    python scripts/run_study.py modes   --seeds 42 43 44
    python scripts/run_study.py privacy --epsilons 1 5 10 20 50 off
    python scripts/run_study.py noniid  --alphas 0.1 0.3 1 10

Prints a results table when the study finishes and writes <study_id>.csv.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request


def api(base: str, path: str, body: dict | None = None, raw: bool = False):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("COORDINATOR_ADMIN_PASSWORD"):
        headers["Authorization"] = "Basic " + base64.b64encode(
            f"admin:{os.environ['COORDINATOR_ADMIN_PASSWORD']}".encode()).decode()
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers=headers, method="POST" if body is not None else "GET")
    data = urllib.request.urlopen(req, timeout=60).read()
    return data if raw else json.loads(data)


def parse_val(s: str):
    return "off" if s == "off" else float(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["modes", "privacy", "noniid"])
    ap.add_argument("--coordinator", default="http://127.0.0.1:5000")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epsilons", nargs="+", default=["1", "5", "10", "20", "50", "off"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.3, 1.0, 10.0])
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    nodes = api(args.coordinator, "/api/nodes")
    online = [n for n in nodes if n["online"]]
    if len(online) < 2:
        print("need at least 2 online campus nodes (start them with scripts/start_cluster.py)", file=sys.stderr)
        return 1
    meta = api(args.coordinator, "/api/meta")
    models = meta["models"]
    base = {"n_rounds": args.rounds,
            "campuses": [{"node_id": n["id"], "model": models[i % 4],
                          "label_noise": 0.35 if i == len(online) - 1 else 0.0}
                         for i, n in enumerate(online)]}
    spec = {"name": args.name or f"CLI {args.kind}", "kind": args.kind, "base": base, "seeds": args.seeds}
    if args.kind == "modes":
        spec["modes"] = ["local_only", "uniform_fd", "trust_fixed", "trust_adaptive"]
    elif args.kind == "privacy":
        spec["modes"] = ["trust_adaptive"]
        spec["vary"] = {"dp.epsilon": [parse_val(e) for e in args.epsilons]}
    else:
        spec["modes"] = ["local_only", "uniform_fd", "trust_adaptive"]
        spec["vary"] = {"alpha": args.alphas}
    res = api(args.coordinator, "/api/studies", spec)
    sid = res["id"]
    print(f"study {sid}: {res['runs']} runs queued")
    while True:
        s = api(args.coordinator, f"/api/studies/{sid}")
        done = sum(e["status"] in ("completed", "failed", "cancelled") for e in s["experiments"])
        print(f"\r  {done}/{len(s['experiments'])} finished", end="", flush=True)
        if s["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(3)
    print()
    rows = s["aggregate"]["rows"]
    print(f"\n{'setting':<10}{'method':<18}{'runs':>5}{'accuracy':>12}{'min-campus':>12}{'conv.rnd':>10}{'comm KB':>10}")
    for r in rows:
        print(f"{('-' if r['vary_value'] in ('None', None) else r['vary_value']):<10}{r['mode']:<18}{r['runs']:>5}"
              f"{r['acc_mean']*100:>8.1f}±{r['acc_std']*100:<3.1f}{r['min_acc_mean']*100:>11.1f}%"
              f"{r['convergence_round']:>10.1f}{(r['comm_bytes'] or 0)/1024:>10.0f}")
    csv_bytes = api(args.coordinator, f"/api/studies/{sid}/export.csv", raw=True)
    with open(f"{sid}.csv", "wb") as f:
        f.write(csv_bytes)
    print(f"\nwrote {sid}.csv")
    return 0 if s["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
