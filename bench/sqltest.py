# sqltest.py <config_name> <model_dir> [extra flags...]: ask the MLX reference server for the SQL migration N times and grade each answer.
# This drives mlx-serve (the "Original MLX" column of the guide), not Strata; MLX_SERVE names the binary (default: on PATH).
import json, os, signal, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sqltask import PROMPT, extract_sql, check

OUT = os.path.expanduser(os.environ.get("STRATA_BENCH_OUT", "~/strata-bench"))
BIN = os.environ.get("MLX_SERVE", "mlx-serve")
URL = "http://127.0.0.1:8000"
TRIALS = int(os.environ.get("TRIALS", "5"))
FLAGS = ["--serve", "--host", "127.0.0.1", "--port", "8000", "--ctx-size", "131072", "--kv-quant", os.environ.get("KV", "8"),
         "--mtp-head-kv-quant", "--wired-margin-gib", "2", "--skip-mem-preflight", "--max-resident-models", "1"]
name, model, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(OUT, exist_ok=True)
log = f"{OUT}/{name}.sql.server.log"

def say(*a): print(time.strftime("%H:%M:%S"), name, *a, flush=True)

def ready():
    try:
        with urllib.request.urlopen(URL + "/v1/models", timeout=5) as r:
            return any(m.get("state") == "ready" for m in json.loads(r.read())["data"])
    except Exception:
        return False

subprocess.run(["pkill", "-f", "mlx-serve --model"])
while subprocess.run(["pgrep", "-f", "mlx-serve --model"], capture_output=True).returncode == 0:
    time.sleep(2)
time.sleep(5)
srv = subprocess.Popen([BIN, "--model", model, *FLAGS, *extra, "--log-file", log],
                       stdout=open(log + ".out", "w"), stderr=subprocess.STDOUT)
t0 = time.time()
while not ready():
    if srv.poll() is not None or time.time() - t0 > 1200:
        say("server failed to start"); sys.exit(1)
    time.sleep(5)

results = []
for k in range(TRIALS):
    body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 16000,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20}
    t1 = time.time()
    try:
        req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            resp = json.loads(r.read())
        msg = resp["choices"][0]["message"]
        text = msg.get("content") or ""
        ok, problems = check(extract_sql(text))
        rec = {"trial": k, "ok": ok, "problems": problems, "s": round(time.time() - t1),
               "tokens": resp["usage"]["completion_tokens"], "answer": text}
    except Exception as e:
        rec = {"trial": k, "ok": False, "problems": [f"request failed: {e!r}"[:300]], "s": round(time.time() - t1)}
    results.append(rec)
    say("trial", k, "PASS" if rec["ok"] else "FAIL", rec["s"], "s", rec.get("tokens"), "tokens", "; ".join(rec["problems"])[:300])

json.dump(results, open(f"{OUT}/{name}.sql.json", "w"), indent=1)
say("SCORE", sum(r["ok"] for r in results), "/", TRIALS)
srv.send_signal(signal.SIGTERM)
try:
    srv.wait(120)
except Exception:
    srv.kill()
