#!/usr/bin/env python3
"""Compare new DA v2 outputs with the previous DA run and existing CoT numbers.

Usage (from this directory):
  ../.venv/bin/python analyze_da_v2.py raw_outputs_da_v2.jsonl

Reads:
  raw_outputs_da_v2.jsonl          (new DA)
  ../calib_int/raw_outputs.jsonl   (old DA + CoT)
  ../calib_int/calib_set.jsonl     (initial amounts, questions)
  few_shot.json, env.json          (optional metadata)

Writes:
  metrics_da_v2.json
  da_v2_compare.png
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict
from decimal import Decimal, InvalidOperation

HERE = os.path.dirname(os.path.abspath(__file__))
NEW_RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "raw_outputs_da_v2.jsonl")
OLD_RAW = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "../calib_int/raw_outputs.jsonl")
CALIB = os.path.join(HERE, "../calib_int/calib_set.jsonl")

NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def parse(text):
    m = re.findall(r"####\s*(?:ANSWER\s*=\s*)?(" + NUM + r")", text)
    if m:
        return m[-1], "marker"
    nums = re.findall(NUM, text)
    return (nums[-1], "fallback") if nums else (None, "none")


def parse_number(s):
    if s is None:
        return None
    t = str(s).strip().replace("$", "").replace(",", "").replace(" ", "")
    try:
        return Decimal(t)
    except InvalidOperation:
        return None


def grade(pred, gold, decimals=0):
    gp = [x for x in str(gold).split(",")] if "," in str(gold) and decimals >= 0 else [str(gold)]
    pp = [x for x in re.split(r"[;,]\s*(?=[\$\d])", str(pred).strip())] if len(gp) > 1 else [pred]
    if len(gp) != len(pp):
        return False
    tol = Decimal(5) / (Decimal(10) ** (decimals + 1))
    for p, g in zip(pp, gp):
        a, b = parse_number(p), parse_number(g)
        if a is None or b is None or abs(a - b) > tol:
            return False
    return True


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - r), min(1.0, c + r)]


def score_row(r, initial):
    pred, how = parse(r["text"])
    dec = r.get("answer_decimals", 0)
    gold = r.get("gold")
    return {
        **r,
        "pred": pred,
        "parse_how": how,
        "correct": bool(pred is not None and grade(pred, gold, dec)),
        "correct_strict": bool(pred is not None and how == "marker" and grade(pred, gold, dec)),
        "echo_initial": bool(pred is not None and initial is not None and grade(pred, initial, dec)),
        "trunc": r.get("finish_reason") == "length",
        "initial": initial,
    }


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def fmt_ci(acc, ci):
    if acc is None or ci[0] is None:
        return "—"
    return f"{acc:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"


def cell_stats(rows):
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    acc = (k / n) if n else None
    return {
        "n": n,
        "correct": k,
        "acc": acc,
        "wilson95": wilson(k, n) if n else [None, None],
        "correct_strict_marker": sum(r["correct_strict"] for r in rows),
        "parse_fail": sum(r["parse_how"] == "none" for r in rows),
        "parse_fail_rate": (sum(r["parse_how"] == "none" for r in rows) / n) if n else None,
        "parse_fallback": sum(r["parse_how"] == "fallback" for r in rows),
        "parse_fallback_rate": (sum(r["parse_how"] == "fallback" for r in rows) / n) if n else None,
        "echo_initial": sum(r["echo_initial"] for r in rows),
        "echo_initial_rate": (sum(r["echo_initial"] for r in rows) / n) if n else None,
        "trunc_rate": (sum(r["trunc"] for r in rows) / n) if n else None,
        "mean_out_tokens": (sum(r.get("n_out_tokens", 0) for r in rows) / n) if n else None,
    }


def main():
    calib = {r["qid"]: r for r in load_jsonl(CALIB)}

    def initial_of(r):
        if r.get("initial") is not None:
            return r["initial"]
        c = calib.get(r["qid"])
        if c:
            return c["chains"][0]["initial"]
        return None

    new_rows = [score_row(r, initial_of(r)) for r in load_jsonl(NEW_RAW)]
    new_rows = [r for r in new_rows if r.get("mode", "DA") == "DA"]
    # 上一轮 raw_outputs.jsonl 不一定在仓库里。缺文件时只汇总新的 DA，不中断。
    if os.path.exists(OLD_RAW):
        old_all = load_jsonl(OLD_RAW)
    else:
        print(f"old raw outputs not found, skipping DA/CoT comparison: {OLD_RAW}")
        old_all = []
    old_da = [score_row(r, initial_of(r)) for r in old_all if r.get("mode") == "DA"]
    old_cot = [score_row(r, initial_of(r)) for r in old_all if r.get("mode") == "CoT"]

    new_by = defaultdict(list)
    old_da_by = defaultdict(list)
    cot_by = defaultdict(list)
    for r in new_rows:
        new_by[r["d"]].append(r)
    for r in old_da:
        old_da_by[r["d"]].append(r)
    for r in old_cot:
        cot_by[r["d"]].append(r)

    depths = sorted(new_by)
    table = []
    for d in depths:
        nd = {r["qid"]: r for r in new_by[d]}
        od = {r["qid"]: r for r in old_da_by.get(d, [])}
        cd = {r["qid"]: r for r in cot_by.get(d, [])}
        # comparability: same qids as the new run
        qids = sorted(nd)
        new_s = cell_stats([nd[q] for q in qids])
        old_s = cell_stats([od[q] for q in qids if q in od])
        cot_s = cell_stats([cd[q] for q in qids if q in cd])
        # paired gaps on items present in CoT and the DA run
        def gap(da_map):
            common = [q for q in qids if q in da_map and q in cd]
            if not common:
                return None
            diffs = [int(cd[q]["correct"]) - int(da_map[q]["correct"]) for q in common]
            mu = sum(diffs) / len(diffs)
            sd = math.sqrt(sum((x - mu) ** 2 for x in diffs) / max(1, len(diffs) - 1))
            se = sd / math.sqrt(len(diffs))
            return {
                "n_paired": len(diffs),
                "gap": mu,
                "ci95_normal": [mu - 1.96 * se, mu + 1.96 * se],
            }

        # new vs old DA, paired
        both = [q for q in qids if q in od]
        if both:
            delta = (sum(int(nd[q]["correct"]) for q in both) - sum(int(od[q]["correct"]) for q in both)) / len(both)
        else:
            delta = None

        row = {
            "d": d,
            "DA_new": new_s,
            "DA_old": old_s,
            "CoT": cot_s,
            "DA_new_minus_DA_old": delta,
            "gap_CoT_minus_DA_new": gap(nd),
            "gap_CoT_minus_DA_old": gap(od),
        }
        table.append(row)

    T = {r["d"]: r for r in table}

    def acc(block, key):
        return T[block][key]["acc"] if block in T else None

    checks = {}
    if 1 in T:
        da1 = acc(1, "DA_new")
        checks["DA_new(1) in [0.80,0.95]"] = {"value": da1, "pass": da1 is not None and 0.80 <= da1 <= 0.95}
        checks["DA_new(1) in heuristic p1 window [0.955,0.986]"] = {
            "value": da1, "pass": da1 is not None and 0.955 <= da1 <= 0.986
        }
    if 20 in T:
        da20 = acc(20, "DA_new")
        cot20 = acc(20, "CoT")
        checks["DA_new(20) <= 0.10"] = {"value": da20, "pass": da20 is not None and da20 <= 0.10}
        checks["CoT(20) in [0.40,0.75] (from previous run, hard)"] = {
            "value": cot20, "pass": cot20 is not None and 0.40 <= cot20 <= 0.75
        }
        if 1 in T and T[1]["gap_CoT_minus_DA_new"] and T[20]["gap_CoT_minus_DA_new"]:
            g1 = T[1]["gap_CoT_minus_DA_new"]["gap"]
            g20 = T[20]["gap_CoT_minus_DA_new"]["gap"]
            checks["gap_new(20)-gap_new(1) >= 0.30"] = {
                "value": round(g20 - g1, 6), "pass": (g20 - g1) >= 0.30
            }

    out = {
        "new_raw": os.path.abspath(NEW_RAW),
        "old_raw": os.path.abspath(OLD_RAW),
        "old_raw_present": bool(old_all),
        "n_new": len(new_rows),
        "table": table,
        "checks": checks,
    }
    env_path = os.path.join(os.path.dirname(os.path.abspath(NEW_RAW)), "env.json")
    if not os.path.exists(env_path):
        env_path = os.path.join(HERE, "env.json")
    if os.path.exists(env_path):
        out["env"] = json.load(open(env_path, encoding="utf-8"))
    few_path = os.path.join(HERE, "few_shot.json")
    if os.path.exists(few_path):
        out["few_shot_system"] = json.load(open(few_path, encoding="utf-8")).get("system")

    out_dir = os.path.dirname(os.path.abspath(NEW_RAW)) or HERE
    metrics_path = os.path.join(out_dir, "metrics_da_v2.json")
    plot_path = os.path.join(out_dir, "da_v2_compare.png")
    json.dump(
        out,
        open(metrics_path, "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    # print table
    print("=== DA v2 vs old DA (same items) and existing CoT ===")
    hdr = (f"{'d':>3} {'n':>4}  {'DA_old':>22}  {'DA_new':>22}  {'Δacc':>7}  "
           f"{'CoT':>22}  {'gap_old':>8}  {'gap_new':>8}  {'echo_old':>9}  {'echo_new':>9}  {'fail_new':>8}")
    print(hdr)
    for r in table:
        n = r["DA_new"]["n"]
        dlt = r["DA_new_minus_DA_old"]
        g_old = r["gap_CoT_minus_DA_old"]["gap"] if r["gap_CoT_minus_DA_old"] else None
        g_new = r["gap_CoT_minus_DA_new"]["gap"] if r["gap_CoT_minus_DA_new"] else None
        print(
            f"{r['d']:>3} {n:>4}  {fmt_ci(r['DA_old']['acc'], r['DA_old']['wilson95']):>22}  "
            f"{fmt_ci(r['DA_new']['acc'], r['DA_new']['wilson95']):>22}  "
            f"{(f'{dlt:+.3f}' if dlt is not None else '—'):>7}  "
            f"{fmt_ci(r['CoT']['acc'], r['CoT']['wilson95']):>22}  "
            f"{(f'{g_old:.3f}' if g_old is not None else '—'):>8}  "
            f"{(f'{g_new:.3f}' if g_new is not None else '—'):>8}  "
            f"{r['DA_old']['echo_initial_rate'] if r['DA_old']['echo_initial_rate'] is not None else 0:>9.3f}  "
            f"{r['DA_new']['echo_initial_rate']:>9.3f}  "
            f"{r['DA_new']['parse_fail_rate']:>8.3f}"
        )
    print("\nWilson 95% CIs in brackets. gap = CoT acc − DA acc (paired on the same qids).")
    print("echo_* = fraction of parsed outputs equal to the initial amount.")
    print("fail_new = parse-none rate on the new DA run.")
    print("\nTask A checks (new DA + previous CoT):")
    for k, v in checks.items():
        flag = "PASS" if v["pass"] else "FAIL"
        print(f"  {flag}  {k}: {v['value']}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        xs = [r["d"] for r in table]

        def errbars(accs, cis, color, label, marker, linestyle):
            if any(a is None or not c or c[0] is None for a, c in zip(accs, cis)):
                print(f"plot skipped series {label}: missing accuracy")
                return
            lo = [max(0.0, a - c[0]) for a, c in zip(accs, cis)]
            hi = [max(0.0, c[1] - a) for a, c in zip(accs, cis)]
            ax.errorbar(xs, accs, yerr=[lo, hi], marker=marker, linestyle=linestyle,
                        capsize=3, label=label, color=color)

        errbars(
            [r["DA_old"]["acc"] for r in table],
            [r["DA_old"]["wilson95"] for r in table],
            "tab:orange", "DA old", "s", "--",
        )
        errbars(
            [r["DA_new"]["acc"] for r in table],
            [r["DA_new"]["wilson95"] for r in table],
            "tab:red", "DA v2", "o", "-",
        )
        errbars(
            [r["CoT"]["acc"] for r in table],
            [r["CoT"]["wilson95"] for r in table],
            "tab:blue", "CoT (previous)", "D", "-",
        )
        ax.set_xlabel("depth d")
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(xs)
        ax.set_title("Task A v4 DA prompt revision vs previous DA / CoT")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"wrote {metrics_path} and {plot_path}")
    except Exception as e:
        print(f"wrote {metrics_path} (plot skipped: {e})")


if __name__ == "__main__":
    main()
