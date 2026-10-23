# sbench.py <name> <exe> [KEY=VAL ...]: one serialized Strata benchmark on an M1 Max / M2 Max.
# Holds $STRATA_BENCH_OUT/gpu.lock, starts the server with <exe> and the env, runs fixed greedy prompts, writes
# $STRATA_BENCH_OUT/<name>.json (decode/prefill tok/s + the answers for exactness), then restarts the baseline server.
# STRATA_ROOT: the repo (default: this file's parent); STRATA_CONFIG: the baseline run config (default: strata-cached.json).
import json, os, subprocess, sys, time, urllib.request, hashlib
R = os.environ.get("STRATA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.expanduser(os.environ.get("STRATA_BENCH_OUT", "~/strata-bench")); LOCK = f"{B}/gpu.lock"
BASE = os.environ.get("STRATA_CONFIG") or f"{R}/strata-cached.json"
name, exe, env = sys.argv[1], os.path.abspath(os.path.expanduser(sys.argv[2])), dict(a.split("=", 1) for a in sys.argv[3:])
os.makedirs(B, exist_ok=True)
while True:
    try: os.mkdir(LOCK); break
    except FileExistsError: time.sleep(5)
open(LOCK + "/owner", "w").write(f"{name} {os.getpid()}\n")
def post(body, t=1800):
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=t) as r: return json.loads(r.read())
def stop():
    subprocess.run(["pkill", "-f", "serve/server.py --engine strata"])
    # match the engine by process name and the server by its exact script args, never a waiter's command line
    while any(subprocess.run(c, capture_output=True).returncode == 0
              for c in (["pgrep", "-x", "strata"], ["pgrep", "-f", "^[^ ]*[Pp]ython[0-9.]* [^ ]*serve/server.py --engine strata"])):
        time.sleep(2)
def start(cfgp, log):
    p = subprocess.Popen([f"{R}/.venv/bin/python", f"{R}/serve/server.py", "--engine", "strata", "--config", cfgp, "--port", "8000"],
                         cwd=R, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        if "ready: http" in open(log).read(): return True
        if p.poll() is not None: return False
        time.sleep(3)
    return False
res = {"name": name, "exe": exe, "env": env}
try:
    base = json.load(open(BASE))
    cfg = dict(base, exe=exe, env={**base.get("env", {}), **env}, log=f"{B}/{name}.engine.log")
    cfgp = f"{B}/{name}.cfg.json"; json.dump(cfg, open(cfgp, "w"), indent=1)
    stop()
    if not start(cfgp, f"{B}/{name}.server.log"): res["error"] = "start failed"; raise SystemExit
    src = open(f"{R}/serve/frontend.py").read()[:6000]
    prompts = [("warm", "Say hi."),
               ("code", "Write a Python function that parses an RFC 3339 timestamp without datetime.fromisoformat, with unit tests."),
               ("prose", "Explain how a hash map handles collisions, in about 300 words."),
               ("edit", "Here is a Python file:\n\n" + src + "\n\nAdd a docstring to every function and show the full changed file.")]
    for label, p in prompts:
        d = post({"messages": [{"role": "user", "content": p}], "max_tokens": 16 if label == "warm" else 512, "temperature": 0})
        t = d.get("timings", {}); m = d["choices"][0]["message"]
        txt = (m.get("reasoning_content") or m.get("reasoning") or "") + "\n<answer>\n" + (m.get("content") or "")
        res[label] = {"decode": t.get("predicted_per_second"), "prefill": t.get("prompt_per_second"), "n": t.get("predicted_n"),
                      "accept": f"{t.get('draft_n_accepted')}/{t.get('draft_n')}", "sha": hashlib.sha256(txt.encode()).hexdigest()[:16],
                      "text": txt[:4000]}
    long_text = "".join(open(os.path.join(dp, f), errors="ignore").read() for dp, _, fs in os.walk(f"{R}/serve") for f in sorted(fs) if f.endswith(".py"))
    for label, chars in (("p4k", 16000), ("p16k", 64000)):
        d = post({"messages": [{"role": "user", "content": f"[{name} {label}]\n" + long_text[:chars] + "\n\nName three functions above."}], "max_tokens": 32, "temperature": 0})
        t = d.get("timings", {}); res[label] = {"prefill": t.get("prompt_per_second"), "n": t.get("prompt_n"), "ms": t.get("prompt_ms")}
except SystemExit: pass
except Exception as e: res["error"] = repr(e)[:500]
finally:
    json.dump(res, open(f"{B}/{name}.json", "w"), indent=1)
    stop(); start(BASE, f"{B}/_baseline_restart.log")
    os.remove(LOCK + "/owner"); os.rmdir(LOCK)
print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "text"} if isinstance(v, dict) else v) for k, v in res.items()}, indent=0))
