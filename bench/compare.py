# compare.py <ref> <cand>... [--long <ref> <cand>...]: agreement with the reference run, plus speed and context.
# Reads the bench.py results in $STRATA_BENCH_OUT (default ~/strata-bench) and writes summary.txt there.
import json, math, os, sys
OUT = os.path.expanduser(os.environ.get("STRATA_BENCH_OUT", "~/strata-bench"))
def top1(t): return max(t, key=t.get) if t else None
def stats(ref, cand, start=1):
    agree = tot = 0; nll = []; dl = []
    for a, b in zip(ref, cand):
        for i in range(start, min(len(a["top"]), len(b["top"]))):
            if not a["top"][i] or not b["top"][i]: continue
            tot += 1; agree += top1(a["top"][i]) == top1(b["top"][i])
            nll.append(-b["lp"][i]); dl.append(abs(a["lp"][i] - b["lp"][i]))
    return f"agree {100*agree/tot:.2f}% of {tot} | ppl {math.exp(sum(nll)/len(nll)):.4f} | mean|dlogp| {sum(dl)/len(dl):.4f}"
args = sys.argv[1:]; long_ = args[args.index("--long") + 1:] if "--long" in args else []
short = args[: args.index("--long")] if "--long" in args else args
lines = []
for n in short:
    r = json.load(open(f"{OUT}/{n}.result.json")); row = [n]
    if os.path.exists(f"{OUT}/{n}.quality.json") and os.path.exists(f"{OUT}/{short[0]}.quality.json"):
        row.append(stats(json.load(open(f"{OUT}/{short[0]}.quality.json")), json.load(open(f"{OUT}/{n}.quality.json"))))
    if "speed" in r: row += [f"{k} {v['decode']} tok/s" for k, v in r["speed"].items()]
    row += [f"ctx admitted {r.get('max_ctx_admitted')}", f"ctx full {str(r.get('max_ctx_full'))[:120]}"]
    lines.append(" | ".join(map(str, row)))
if long_:
    ref = json.load(open(f"{OUT}/{long_[0]}.longq.json"))
    for n in long_:
        c = json.load(open(f"{OUT}/{n}.longq.json"))
        lines.append(f"LONG {n}: all {stats(ref, c)} || past 40K {stats(ref, c, 40000)}")
open(f"{OUT}/summary2.txt" if long_ else f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
