# sbspec.py <name> <exe> <plan> [--args "--spec 8 --mtp-max-t 2 ..."] [--reps N] [KEY=VAL ...]
# Paired, in-run A/B of drafting settings on an M1 Max / M2 Max: ONE engine process, the same prompts, variants
# interleaved (rotated each rep) through the per-request keys strata_tune {mtp_t, sfx_t, lchain, sfx_gate, spec_min_p}
# (an engine without them ignores the keys).  Holds $STRATA_BENCH_OUT/gpu.lock like sbench.py, writes
# $STRATA_BENCH_OUT/<name>.json, restarts the baseline server at the end.
# Every measured request follows the same reset request and asks for no checkpoint, so each reads its prompt the same
# way: the answers of one prompt must have ONE sha across all variants (bit-exact), and match the "stock" plan's.
# STRATA_ROOT: the repo; STRATA_CONFIG: the baseline run config; BENCH_TS_FILE: a TypeScript file for the edit_ts prompt.
import json, os, subprocess, sys, time, urllib.request, hashlib, statistics
R = os.environ.get("STRATA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.expanduser(os.environ.get("STRATA_BENCH_OUT", "~/strata-bench")); LOCK = f"{B}/gpu.lock"
BASE = os.environ.get("STRATA_CONFIG") or f"{R}/strata-cached.json"
a = sys.argv[1:]
name, exe, plan = a[0], os.path.abspath(os.path.expanduser(a[1])), a[2]
rest = a[3:]; xargs = None; reps = 2; env = {}
i = 0
while i < len(rest):
    if rest[i] == "--args": xargs = rest[i + 1].split(); i += 2
    elif rest[i] == "--reps": reps = int(rest[i + 1]); i += 2
    else: k, v = rest[i].split("=", 1); env[k] = v; i += 1

src_py = open(f"{R}/serve/frontend.py").read()[:6000]
ts = os.path.expanduser(os.environ.get("BENCH_TS_FILE", ""))
src_ts = open(ts, errors="ignore").read()[:9000] if ts and os.path.exists(ts) else open(f"{R}/serve/server.py").read()[:9000]
P = {"edit_py": ("Here is a Python file:\n\n" + src_py + "\n\nAdd a docstring to every function and show the full changed file.", {}),
     "edit_ts": ("Here is a TypeScript file:\n\n" + src_ts + "\n\nAdd a doc comment to every exported function and show the full changed file.", {}),
     "edit_nt": ("Here is a Python file:\n\n" + src_py + "\n\nAdd a docstring to every function and show the full changed file.",
                 {"chat_template_kwargs": {"enable_thinking": False}}),
     "code": ("Write a Python function that parses an RFC 3339 timestamp without datetime.fromisoformat, with unit tests.", {}),
     "prose": ("Explain how a hash map handles collisions, in about 300 words.", {})}
# plans: (prompts, variants {label: strata_tune}); the first variant is the reference of the paired ratios
PLANS = {
    # stock config, stock binary or not: the reference shas of this protocol (one variant: no keys)
    "stock": (["edit_py", "edit_ts", "edit_nt", "code", "prose"], {"stock": {}}),
    # Q1, engine --spec 8 --mtp-max-t 2 --lookup-chain 6: lookup window length, the MTP gate, MTP + lookup chain
    "lookup": (["edit_py", "edit_ts", "edit_nt", "code"],
               {"D4": {"mtp_t": 2, "sfx_t": 4, "lchain": 0},          # = the stock default (--spec 2, suffix to 4)
                "M2": {"mtp_t": 2, "sfx_t": 0, "lchain": 0},          # MTP only (--suffix-draft 0)
                "L8": {"mtp_t": 2, "sfx_t": 8, "lchain": 0},          # lookup windows up to 8
                "L8g": {"mtp_t": 2, "sfx_t": 8, "lchain": 0, "sfx_gate": 0},   # ... without the MTP-agrees gate
                "C8": {"mtp_t": 2, "sfx_t": 8, "lchain": 6}}),        # ... plus lookup chained after the MTP draft
    # Q2, engine --spec 8 --mtp-max-t 4: MTP depth (suffix as the stock rule: spec + 2), and the confidence-gated depth
    "depth": (["code", "prose", "edit_py"],
              {"m2": {"mtp_t": 2, "sfx_t": 4},
               "m3": {"mtp_t": 3, "sfx_t": 5},
               "m4": {"mtp_t": 4, "sfx_t": 6},
               "m4p5": {"mtp_t": 4, "sfx_t": 6, "spec_min_p": 0.5},
               "m4p7": {"mtp_t": 4, "sfx_t": 6, "spec_min_p": 0.7}}),
}
prompts, variants = PLANS[plan]
MAXTOK = int(env.pop("MAXTOK", "768"))
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
    while any(subprocess.run(c, capture_output=True).returncode == 0
              for c in (["pgrep", "-x", "strata"], ["pgrep", "-f", "^[^ ]*[Pp]ython[0-9.]* [^ ]*serve/server.py --engine strata"])):
        time.sleep(2)
def start(cfgp, log, server=f"{R}/serve/server.py"):
    p = subprocess.Popen([f"{R}/.venv/bin/python", server, "--engine", "strata", "--config", cfgp, "--port", "8000"],
                         cwd=R, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        if "ready: http" in open(log).read(): return True
        if p.poll() is not None: return False
        time.sleep(3)
    return False
res = {"name": name, "exe": exe, "plan": plan, "env": env, "runs": []}
elog = f"{B}/{name}.engine.log"
try:
    base = json.load(open(BASE))
    args = list(base["args"])
    if xargs:   # replace --spec N, then append the rest (e.g. --mtp-max-t 2 --lookup-chain 6)
        j = args.index("--spec"); del args[j:j + 2]
        args += xargs
    res["args"] = args
    cfg = dict(base, exe=exe, args=args, env={**base.get("env", {}), **env}, log=elog)
    cfgp = f"{B}/{name}.cfg.json"; json.dump(cfg, open(cfgp, "w"), indent=1)
    stop()
    # the patched server passes the keys; it is the exe's own tree's (an unpatched server drops them silently)
    tree_server = os.path.join(os.path.dirname(os.path.dirname(exe)), "serve", "server.py")
    server = tree_server if os.path.exists(tree_server) else f"{R}/serve/server.py"
    res["server"] = server
    if not start(cfgp, f"{B}/{name}.server.log", server): res["error"] = "start failed"; raise SystemExit
    reset = {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 4, "temperature": 0, "strata_checkpoint": False}
    post(reset)
    labels = list(variants)
    for r in range(reps):
        order = labels[r % len(labels):] + labels[:r % len(labels)]
        for pl in prompts:
            text, extra = P[pl]
            for v in order:
                post(reset)
                off = os.path.getsize(elog) if os.path.exists(elog) else 0
                body = {"messages": [{"role": "user", "content": text}], "max_tokens": MAXTOK, "temperature": 0,
                        "strata_checkpoint": False, **extra}
                if variants[v]: body["strata_tune"] = variants[v]
                d = post(body)
                time.sleep(0.5)
                with open(elog, errors="ignore") as f: f.seek(off); tail = f.read()
                keep = [l for l in tail.splitlines() if any(s in l for s in ("suffix drafts", "lookup chain", "spec rounds", "depth", "acceptance"))]
                t = d.get("timings", {}); m = d["choices"][0]["message"]
                txt = (m.get("reasoning_content") or m.get("reasoning") or "") + "\n<answer>\n" + (m.get("content") or "")
                row = {"rep": r, "prompt": pl, "variant": v, "decode": t.get("predicted_per_second"), "n": t.get("predicted_n"),
                       "accept": f"{t.get('draft_n_accepted')}/{t.get('draft_n')}", "sha": hashlib.sha256(txt.encode()).hexdigest()[:16],
                       "log": keep, "think_chars": len(m.get("reasoning_content") or m.get("reasoning") or "")}
                res["runs"].append(row)
                print(json.dumps({k: row[k] for k in ("rep", "prompt", "variant", "decode", "n", "accept", "sha")}), flush=True)
                if r == 0 and v == order[0]: res.setdefault("text", {})[pl] = txt[:3000]
                json.dump(res, open(f"{B}/{name}.json", "w"), indent=1)
except SystemExit: pass
except Exception as e: res["error"] = repr(e)[:500]
finally:
    # summary: per prompt, the median decode per variant, the paired ratio to the first variant (same rep), the shas
    summ = {}
    for pl in prompts:
        rows = [x for x in res["runs"] if x["prompt"] == pl]
        if not rows: continue
        ref = list(variants)[0]
        s = {"shas": sorted({x["sha"] for x in rows})}
        for v in variants:
            dv = [x["decode"] for x in rows if x["variant"] == v and x["decode"]]
            ratios = []
            for rr in range(reps):
                a_ = [x["decode"] for x in rows if x["variant"] == v and x["rep"] == rr and x["decode"]]
                b_ = [x["decode"] for x in rows if x["variant"] == ref and x["rep"] == rr and x["decode"]]
                if a_ and b_: ratios.append(round(a_[0] / b_[0], 3))
            if dv: s[v] = {"median": statistics.median(dv), "all": dv, "vs_" + ref: ratios}
        summ[pl] = s
    res["summary"] = summ
    json.dump(res, open(f"{B}/{name}.json", "w"), indent=1)
    stop(); start(BASE, f"{B}/_baseline_restart.log")
    os.remove(LOCK + "/owner"); os.rmdir(LOCK)
print(json.dumps(res.get("summary"), indent=1)); print("error:", res.get("error"))
