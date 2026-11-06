"""End-to-end test: real campus processes + the coordinator, talking over HTTP."""
import os
import socket
import subprocess
import sys
import time

import pytest
import requests

from common.security import TOKEN_HEADER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN = "test-token"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait(url: str, timeout: float = 60) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if requests.get(url, timeout=2).ok:
                return
        except requests.RequestException:
            time.sleep(0.3)
    raise RuntimeError(f"{url} did not come up")


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    os.environ["FEDTRUST_TOKEN"] = TOKEN
    env = {**os.environ, "PYTHONPATH": ROOT, "FEDTRUST_TOKEN": TOKEN}
    procs, urls = [], []
    for i in range(3):
        port = free_port()
        procs.append(subprocess.Popen([sys.executable, "-m", "campus.app", "--name", f"T{i}", "--port", str(port)],
                                      cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        urls.append(f"http://127.0.0.1:{port}")
    for u in urls:
        wait(u + "/api/health")

    from coordinator.app import create_app
    app = create_app(db_path=str(tmp_path_factory.mktemp("db") / "t.db"), data_dir=os.path.join(ROOT, "data"))
    client = app.test_client()
    nodes = [client.post("/api/nodes", json={"url": u}).get_json() for u in urls]
    yield {"client": client, "nodes": nodes, "urls": urls, "procs": procs, "app": app}
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait(10)


def wait_done(client, eid, timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        e = client.get(f"/api/experiments/{eid}").get_json()
        if e["status"] in ("completed", "failed", "cancelled"):
            return e
        time.sleep(1)
    raise TimeoutError(eid)


def cfg_for(nodes, **kw):
    c = {"n_rounds": 3, "warmup_epochs": 3, "local_epochs": 1, "distill_epochs": 1,
         "campuses": [{"node_id": n["id"], "model": m} for n, m in zip(nodes, ["mlp_small", "cnn_small", "mlp_wide"])]}
    c.update(kw)
    return c


def test_nodes_are_online_and_token_is_enforced(cluster):
    nodes = cluster["client"].get("/api/nodes").get_json()
    assert len(nodes) == 3 and all(n["online"] for n in nodes)
    assert requests.get(cluster["urls"][0] + "/api/info", timeout=3).status_code == 401
    assert requests.get(cluster["urls"][0] + "/api/info", headers={TOKEN_HEADER: "wrong"}, timeout=3).status_code == 401
    assert requests.get(cluster["urls"][0] + "/api/info", headers={TOKEN_HEADER: TOKEN}, timeout=3).ok


@pytest.mark.parametrize("mode", ["trust_adaptive", "uniform_fd", "trust_fixed", "local_only"])
def test_every_mode_completes(cluster, mode):
    c = cluster["client"]
    r = c.post("/api/experiments", json=cfg_for(cluster["nodes"], mode=mode, name=mode))
    assert r.status_code == 201, r.get_json()
    e = wait_done(c, r.get_json()["id"])
    assert e["status"] == "completed", e.get("error")
    rounds = e["rounds"]
    assert [x["round"] for x in rounds] == [0, 1, 2, 3]
    assert all(0 <= cm["acc"] <= 1 for x in rounds for cm in x["campuses"])
    s = e["summary"]
    if mode == "local_only":
        assert s["total_comm_bytes"] < 20_000 and s["teacher_acc_mean"] is None
    else:
        t = rounds[-1]["campuses"][0]
        assert abs(sum(cm["trust"] for cm in rounds[-1]["campuses"]) - 1.0) < 1e-6
        assert t["bytes_up"] > 0 and t["bytes_down"] > 0
        assert rounds[-1]["privacy"]["epsilon_total"] > rounds[1]["privacy"]["epsilon_total"] > 0
        if mode == "uniform_fd":
            assert all(abs(cm["trust"] - 1 / 3) < 1e-6 for cm in rounds[-1]["campuses"])
    # collaboration should not leave the federation worse off than where warm-up put it
    if mode != "local_only":
        assert s["final_mean_acc"] >= s["start_mean_acc"] - 0.02


def test_dp_off_reports_no_epsilon(cluster):
    c = cluster["client"]
    eid = c.post("/api/experiments", json=cfg_for(cluster["nodes"], dp={"enabled": False})).get_json()["id"]
    e = wait_done(c, eid)
    assert e["status"] == "completed" and e["summary"]["sigma"] == 0.0 and e["summary"]["epsilon_total"] is None


def test_csv_export_and_compare(cluster):
    c = cluster["client"]
    ids = [x["id"] for x in c.get("/api/experiments").get_json() if x["status"] == "completed"][:2]
    csv = c.get(f"/api/experiments/{ids[0]}/export.csv")
    assert csv.status_code == 200 and csv.data.decode().startswith("experiment,mode,round")
    cmp = c.get("/api/compare?ids=" + ",".join(ids)).get_json()
    assert len(cmp) == 2 and cmp[0]["rounds"]


def test_study_expands_runs_and_aggregates(cluster):
    c = cluster["client"]
    spec = {"name": "mini", "kind": "privacy", "base": cfg_for(cluster["nodes"], n_rounds=2), "modes": ["trust_adaptive"],
            "vary": {"dp.epsilon": [20, "off"]}, "seeds": [1]}
    r = c.post("/api/studies", json=spec)
    assert r.status_code == 201 and r.get_json()["runs"] == 2
    sid = r.get_json()["id"]
    end = time.time() + 240
    while time.time() < end:
        s = c.get(f"/api/studies/{sid}").get_json()
        if s["status"] in ("completed", "failed"):
            break
        time.sleep(1)
    assert s["status"] == "completed"
    assert {row["vary_value"] for row in s["aggregate"]["rows"]} == {"20", "off"}


def test_bad_configs_are_rejected(cluster):
    c = cluster["client"]
    assert c.post("/api/experiments", json={"campuses": []}).status_code == 400
    bad = cfg_for(cluster["nodes"], mode="nope")
    assert c.post("/api/experiments", json=bad).status_code == 400
    bad = cfg_for(cluster["nodes"])
    bad["campuses"][0]["node_id"] = "node_unknown"
    assert c.post("/api/experiments", json=bad).status_code == 400


def test_cancel_stops_a_running_experiment(cluster):
    c = cluster["client"]
    eid = c.post("/api/experiments", json=cfg_for(cluster["nodes"], n_rounds=60, local_epochs=2)).get_json()["id"]
    time.sleep(6)
    assert c.post(f"/api/experiments/{eid}/cancel").get_json()["cancelled"]
    e = wait_done(c, eid, 120)
    assert e["status"] == "cancelled" and e["progress"] < 60


def test_unreachable_campus_fails_cleanly(cluster):
    c = cluster["client"]
    ghost = c.post("/api/nodes", json={"url": "http://127.0.0.1:1", "name": "ghost", "force": True}).get_json()
    cfg = cfg_for(cluster["nodes"][:2])
    cfg["campuses"].append({"node_id": ghost["id"], "model": "linear"})
    e = wait_done(c, c.post("/api/experiments", json=cfg).get_json()["id"])
    assert e["status"] == "failed" and "ghost" in e["error"]
    c.delete(f"/api/nodes/{ghost['id']}")


def test_campus_failing_mid_run_does_not_kill_the_round(cluster):
    """Kill one campus after the run started: the federation continues with the survivors."""
    c = cluster["client"]
    eid = c.post("/api/experiments", json=cfg_for(cluster["nodes"], n_rounds=12, local_epochs=2)).get_json()["id"]
    end = time.time() + 60
    while time.time() < end:
        if c.get(f"/api/experiments/{eid}").get_json()["progress"] >= 1:
            break
        time.sleep(0.5)
    cluster["procs"][2].terminate()
    e = wait_done(c, eid, 240)
    assert e["status"] == "completed", e.get("error")
    assert any(r["errors"] for r in e["rounds"])                              # failure was recorded
    assert e["rounds"][-1]["campuses"][2]["error"]                            # and attributed to the right campus
    assert e["rounds"][-1]["campuses"][0]["acc"] is not None                  # survivors kept learning


def test_self_registration(tmp_path):
    from coordinator.app import create_app
    app = create_app(db_path=str(tmp_path / "r.db"), data_dir=os.path.join(ROOT, "data"))
    cl = app.test_client()
    assert cl.post("/api/nodes/register", json={"name": "x", "url": "http://h:1"}).status_code == 401
    ok = cl.post("/api/nodes/register", json={"name": "x", "url": "http://h:1"}, headers={TOKEN_HEADER: TOKEN})
    assert ok.status_code == 201 and len(cl.get("/api/nodes").get_json()) == 1
