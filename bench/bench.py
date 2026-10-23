# bench.py <config_name> <model_dir> [extra flags...]: quality, speed and max-context run for one MLX reference server config.
# This drives mlx-serve (the "Original MLX" column of the guide), not Strata; MLX_SERVE names the binary (default: on PATH).
# BENCH_CORPUS: a folder of .ts/.py/.md source files for the quality and context prompts (default: the repo's serve/).
# BENCH_EDIT_FILE: the source file of the "edit" speed prompt (default: serve/server.py).
import glob, json, os, re, signal, subprocess, sys, time, urllib.request, urllib.error

ROOT = os.environ.get("STRATA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser(os.environ.get("STRATA_BENCH_OUT", "~/strata-bench"))
CORPUS = os.path.expanduser(os.environ.get("BENCH_CORPUS", f"{ROOT}/serve"))
EDIT_FILE = os.path.expanduser(os.environ.get("BENCH_EDIT_FILE", f"{ROOT}/serve/server.py"))
BIN = os.environ.get("MLX_SERVE", "mlx-serve")
URL = "http://127.0.0.1:8000"
BASE_FLAGS = ["--serve", "--host", "127.0.0.1", "--port", "8000", "--ctx-size", "131072", "--kv-quant", os.environ.get("KV", "8"),
              "--mtp-head-kv-quant", "--wired-margin-gib", "2", "--skip-mem-preflight", "--max-resident-models", "1"]
name, model, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(OUT, exist_ok=True)
log_path = f"{OUT}/{name}.server.log"
res = {"config": name, "model": model, "extra": extra}

def say(*a):
    print(time.strftime("%H:%M:%S"), name, *a, flush=True)

def post(path, body, timeout):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def ready():
    try:
        with urllib.request.urlopen(URL + "/v1/models", timeout=5) as r:
            return any(m.get("state") == "ready" for m in json.loads(r.read())["data"])
    except Exception:
        return False

def source_files(lo, hi):
    files = [f for ext in ("ts", "tsx", "py", "md") for f in glob.glob(f"{CORPUS}/**/*.{ext}", recursive=True)
             if "node_modules" not in f and ".test." not in f and lo < os.path.getsize(f) < hi]
    return sorted(files)

def corpus():
    files = source_files(8000, 60000)
    files = files[:: max(1, len(files) // 16)][:16]
    return [(os.path.relpath(f, CORPUS), open(f, errors="ignore").read()[:7000]) for f in files]

subprocess.run(["pkill", "-f", "mlx-serve --model"])
while subprocess.run(["pgrep", "-f", "mlx-serve --model"], capture_output=True).returncode == 0:
    time.sleep(2)
time.sleep(5)
srv = subprocess.Popen([BIN, "--model", model, *BASE_FLAGS, *extra, "--log-file", log_path],
                       stdout=open(log_path + ".out", "w"), stderr=subprocess.STDOUT)
t0 = time.time()
while not ready():
    if srv.poll() is not None or time.time() - t0 > 1200:
        say("server failed to start"); res["error"] = "start"; break
    time.sleep(5)
res["load_s"] = round(time.time() - t0)
say("ready in", res["load_s"], "s")

if "error" not in res and os.environ.get("SKIP_QUALITY") != "1":
    q = []
    for path, text in corpus():
        r = post("/v1/completions", {"prompt": text, "max_tokens": 1, "temperature": 0, "logprobs": 5, "echo": True}, 900)
        lp = r["choices"][0]["logprobs"]
        q.append({"file": path, "tokens": lp["tokens"], "lp": lp["token_logprobs"], "top": lp["top_logprobs"]})
        say("quality", path, len(lp["tokens"]), "tokens")
    json.dump(q, open(f"{OUT}/{name}.quality.json", "w"))

if "error" not in res and os.environ.get("LONGQ") == "1":
    files = source_files(4000, 40000)
    long_text, i = "", 0
    while len(long_text) < 235000 and i < len(files):
        long_text += f"\n// FILE {os.path.relpath(files[i], CORPUS)}\n" + open(files[i], errors="ignore").read(); i += 1
    t1 = time.time()
    lp = None
    for tail in ("", "\n", "\nexport"):  # the server drops logprobs when the model stops at once; nudge it
        r = post("/v1/completions", {"prompt": long_text[:235000] + tail, "max_tokens": 1, "temperature": 0, "logprobs": 5, "echo": True}, 3600)
        lp = r["choices"][0]["logprobs"]
        if lp: break
    json.dump([{"file": "long", "tokens": lp["tokens"], "lp": lp["token_logprobs"], "top": lp["top_logprobs"]}],
              open(f"{OUT}/{name}.longq.json", "w"))
    res["longq"] = {"tokens": len(lp["tokens"]), "s": round(time.time() - t1)}
    say("long quality", res["longq"])

if "error" not in res and os.environ.get("SKIP_SPEED") != "1":
    src = open(EDIT_FILE, errors="ignore").read()[:9000]
    prompts = [
        ("code", "Write a Python function that parses an RFC 3339 timestamp without datetime.fromisoformat, with unit tests."),
        ("prose", "Explain how a hash map handles collisions, in about 300 words."),
        ("edit", "Here is a source file:\n\n" + src + "\n\nAdd a doc comment to every function and show the full changed file."),
    ]
    post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 16, "temperature": 0}, 600)
    speed = {}
    for label, p in prompts:
        r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "max_tokens": 600, "temperature": 0}, 1200)
        time.sleep(1)
        lines = [l for l in open(log_path, errors="ignore") if "decode:" in l and "<-" in l]
        m = re.search(r"prefill: ([\d.]+) tok/s.*decode: ([\d.]+) tok/s", lines[-1]) if lines else None
        speed[label] = {"decode": float(m.group(2)) if m else None, "prefill": float(m.group(1)) if m else None,
                        "tokens": r["usage"]["completion_tokens"]}
        say("speed", label, speed[label])
    res["speed"] = speed

if "error" not in res and os.environ.get("SKIP_CTX") != "1":
    text = "".join(t for _, t in corpus()) * 40
    r = post("/v1/completions", {"prompt": text[:40000], "max_tokens": 1, "temperature": 0}, 900)
    cpt = 40000 / r["usage"]["prompt_tokens"]
    say("chars per token", round(cpt, 3))
    best = None
    for n in range(131072, 40000, -8192):
        body = {"messages": [{"role": "user", "content": f"[{name} {n}]\n" + text[: int(n * cpt)]}],
                "max_tokens": 16384, "temperature": 0}
        try:
            r = post("/v1/chat/completions", body, 30)
            say("ctx", n, "refused", json.dumps(r)[:300]); continue
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="ignore")[:300]
            say("ctx", n, "refused", msg); continue
        except Exception:
            best = n; say("ctx", n, "accepted (prefill running)"); break
    res["max_ctx_admitted"] = best
    time.sleep(10)
    while True:
        tail = open(log_path, errors="ignore").read()[-3000:]
        if "client_disconnect" in tail or "<-" in tail.split("POST")[-1]:
            break
        time.sleep(10)
    if best and os.environ.get("SKIP_CTX_FULL") != "1":
        body = {"messages": [{"role": "user", "content": f"[{name} full {best}]\n" + text[: int(best * cpt)]
                              + "\n\nIn one sentence, what is this text about?"}], "max_tokens": 200, "temperature": 0}
        t1 = time.time()
        try:
            r = post("/v1/chat/completions", body, 3600)
            res["max_ctx_full"] = {"ok": True, "prompt_tokens": r["usage"]["prompt_tokens"], "s": round(time.time() - t1),
                                   "answer": r["choices"][0]["message"].get("content", "")[:300]}
        except urllib.error.HTTPError as e:
            res["max_ctx_full"] = {"ok": False, "error": e.read().decode(errors="ignore")[:300]}
        except Exception as e:
            res["max_ctx_full"] = {"ok": False, "error": repr(e)[:300]}
        say("ctx full", res["max_ctx_full"])

json.dump(res, open(f"{OUT}/{name}.result.json", "w"), indent=2)
srv.send_signal(signal.SIGTERM)
try:
    srv.wait(120)
except Exception:
    srv.kill()
say("done")
