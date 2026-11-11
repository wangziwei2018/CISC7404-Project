#!/usr/bin/env python3
"""把 raw_outputs.jsonl 汇总成 metrics.json 和 accuracy_vs_d.png。判分用 generate.grade。"""
import json, math, re, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate import grade

RAW = sys.argv[1] if len(sys.argv) > 1 else "raw_outputs.jsonl"
NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"

def parse(text):
    """Strict: last '#### N' line. Fallback (recorded separately): last number in text."""
    m = re.findall(r"####\s*(?:ANSWER\s*=\s*)?(" + NUM + r")", text)
    if m:
        return m[-1], "marker"
    nums = re.findall(NUM, text)
    return (nums[-1], "fallback") if nums else (None, "none")

def wilson(k, n, z=1.959963984540054):
    if n == 0: return [None, None]
    p = k / n; den = 1 + z*z/n; c = (p + z*z/(2*n)) / den
    r = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [max(0.0, c-r), min(1.0, c+r)]

def ca_trend(succ, tot, scores):  # identical to Experiments/task_a/experiment_task_a.ipynb
    n = sum(tot); p = sum(succ)/n; mean = sum(t*s for t, s in zip(tot, scores))/n
    var = p*(1-p)*sum(t*(s-mean)**2 for t, s in zip(tot, scores))
    if var == 0: return {"z": 0.0, "p_two_sided": 1.0, "testable": False}
    z = sum((s-mean)*(x-t*p) for x, t, s in zip(succ, tot, scores))/math.sqrt(var)
    return {"z": z, "p_two_sided": math.erfc(abs(z)/math.sqrt(2)), "testable": True}

def p1_mle(cells):  # acc_d = p^d, binomial MLE by grid search
    best = None
    for i in range(1, 100000):
        p = i/100000; ll = 0.0
        for d, k, n in cells:
            q = p**d
            # p**d 在很小的 p、很大的 d 上会下溢成 0；这些点不可能是 MLE，只避免 log(0)。
            ll += k*math.log(max(q, 1e-300)) + (n-k)*math.log(max(1e-300, 1-q))
        if best is None or ll > best[1]: best = (p, ll)
    return best[0]

rows = [json.loads(l) for l in open(RAW)]
by = defaultdict(list)
for r in rows:
    pred, how = parse(r["text"])
    r["pred"], r["parse_how"] = pred, how
    r["correct"] = pred is not None and grade(pred, r["gold"], r["answer_decimals"])
    r["correct_strict"] = r["correct"] and how == "marker"
    r["trunc"] = r["finish_reason"] == "length"
    lines = [l for l in r["text"].strip().splitlines() if l.strip()]
    r["da_extra_text"] = r["mode"] == "DA" and len(lines) > 1
    by[(r["d"], r["mode"])].append(r)
D = sorted({d for d, _ in by})
table = []
for d in D:
    row = {"d": d}
    for m in ("DA", "CoT"):
        g = by.get((d, m), [])
        k = sum(r["correct"] for r in g); n = len(g)
        row[m] = {"n": n, "correct": k, "acc": k/n if n else None, "wilson95": wilson(k, n),
                  "correct_strict_marker": sum(r["correct_strict"] for r in g),
                  "parse_fallback": sum(r["parse_how"] == "fallback" for r in g),
                  "parse_none": sum(r["parse_how"] == "none" for r in g),
                  "trunc_rate": (sum(r["trunc"] for r in g)/n) if n else None,
                  "da_extra_text": sum(r["da_extra_text"] for r in g),
                  "mean_out_tokens": (sum(r["n_out_tokens"] for r in g)/n) if n else None}
    # paired gap on items run in both modes
    cot = {r["qid"]: r for r in by.get((d, "CoT"), [])}; da = {r["qid"]: r for r in by.get((d, "DA"), [])}
    common = sorted(set(cot) & set(da))
    diffs = [int(cot[q]["correct"]) - int(da[q]["correct"]) for q in common]
    if diffs:
        mu = sum(diffs)/len(diffs); sd = math.sqrt(sum((x-mu)**2 for x in diffs)/max(1, len(diffs)-1))
        se = sd/math.sqrt(len(diffs))
        row["gap"] = {"n_paired": len(diffs), "gap": mu, "ci95_normal": [mu-1.96*se, mu+1.96*se]}
    table.append(row)
T = {r["d"]: r for r in table}
da_cells = [(d, T[d]["DA"]["correct"], T[d]["DA"]["n"]) for d in D]
cot_cells = [(d, T[d]["CoT"]["correct"], T[d]["CoT"]["n"]) for d in D]
ca = {m: ca_trend([T[d][m]["correct"] for d in D], [T[d][m]["n"] for d in D], D) for m in ("DA", "CoT")}
gap_ct = None
# CA on "CoT correct & DA wrong" rate (paired win) as a gap-trend proxy
wins = []
for d in D:
    cot = {r["qid"]: r for r in by[(d, "CoT")]}; da = {r["qid"]: r for r in by[(d, "DA")]}
    c = sorted(set(cot) & set(da)); wins.append((sum(cot[q]["correct"] and not da[q]["correct"] for q in c), len(c)))
ca["CoT_win_over_DA"] = ca_trend([w for w, _ in wins], [n for _, n in wins], D)
da1, cot20, da20 = T[1]["DA"]["acc"], T[20]["CoT"]["acc"], T[20]["DA"]["acc"]
g1, g20 = T[1]["gap"]["gap"], T[20]["gap"]["gap"]
p1_da = p1_mle([c for c in da_cells if c[2]])
p1_cot = p1_mle([c for c in cot_cells if c[2]])
checks = {
    "DA(1) in [0.80,0.95]": {"value": da1, "pass": 0.80 <= da1 <= 0.95},
    "p1 (=DA(1)) in heuristic [0.955,0.986]": {"value": da1, "pass": 0.955 <= da1 <= 0.986},
    "CoT(20) in [0.40,0.75] (hard)": {"value": cot20, "pass": 0.40 <= cot20 <= 0.75},
    "DA(20) <= 0.10": {"value": da20, "pass": da20 <= 0.10},
    "gap(20)-gap(1) >= 0.30": {"value": g20 - g1, "pass": g20 - g1 >= 0.30},
    "CA CoT decreasing p<0.01": {"z": ca["CoT"]["z"], "p": ca["CoT"]["p_two_sided"],
                                 "pass": ca["CoT"]["z"] < 0 and ca["CoT"]["p_two_sided"] < 0.01},
}
out = {"raw": RAW, "n_rows": len(rows), "table": table, "ca_trend": ca,
       "p1_fit_DA_acc_eq_p1_pow_d_mle": p1_da, "p1_fit_CoT_mle": p1_cot,
       "predicted_cot20_from_DA1": da1**20, "checks": checks}
try:
    out["env"] = json.load(open("env.json"))
except FileNotFoundError:
    pass
json.dump(out, open("metrics.json", "w"), indent=2, ensure_ascii=False)
# failure samples
wrong = [r for r in rows if not r["correct"]]
with open("failure_samples.jsonl", "w") as f:
    for r in wrong:
        f.write(json.dumps({k: r[k] for k in ("qid", "d", "mode", "gold", "pred", "parse_how", "trunc", "text")}, ensure_ascii=False) + "\n")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4.5))
for m, c in (("DA", "tab:orange"), ("CoT", "tab:blue")):
    ys = [T[d][m]["acc"] for d in D]; lo = [T[d][m]["acc"]-T[d][m]["wilson95"][0] for d in D]
    hi = [T[d][m]["wilson95"][1]-T[d][m]["acc"] for d in D]
    ax.errorbar(D, ys, yerr=[lo, hi], marker="o", capsize=3, label=m, color=c)
ax.plot(D, [T[d]["gap"]["gap"] for d in D], "k--", marker="s", label="gap (CoT-DA)")
ax.axhspan(0.40, 0.75, color="tab:blue", alpha=0.08, label="CoT(20) target")
ax.set_xlabel("depth d"); ax.set_ylabel("accuracy"); ax.set_ylim(-0.05, 1.05); ax.set_xticks(D)
ax.set_title("Task A v4 cond_wordproblem (int, addsub), Qwen3-4B-Instruct-2507"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("accuracy_vs_d.png", dpi=150)
print(json.dumps({"checks": checks, "p1_mle": p1_da}, indent=2))
for r in table:
    print(r["d"], {m: (r[m]["correct"], r[m]["n"], [round(x, 3) for x in r[m]["wilson95"]], r[m]["trunc_rate"]) for m in ("DA", "CoT")}, round(r["gap"]["gap"], 3))
