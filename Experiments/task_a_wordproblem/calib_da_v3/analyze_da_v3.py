#!/usr/bin/env python3
"""Compare DA v3 (strict + fallback) with DA v2, old DA, and existing CoT.

Usage (from this directory):
  ../.venv/bin/python analyze_da_v3.py raw_outputs_da_v3.jsonl

Reads (comparison files are skipped gracefully if missing):
  raw_outputs_da_v3.jsonl                      (new DA v3)
  ../calib_da_v2/raw_outputs_da_v2.jsonl       (DA v2)
  ../calib_int/raw_outputs.jsonl               (old DA + CoT)
  ../calib_int/calib_set.jsonl                 (initial amounts, optional)

Writes (does not overwrite any v2 outputs):
  metrics_da_v3.json
  da_v3_compare.png

Headline Task A checks use the STRICT metric (last/only #### N after
reconstructing the v3 '#### ' prefill). Fallback last-number accuracy is
reported alongside so it can be compared with the published DA v2 numbers.
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
NEW_RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "raw_outputs_da_v3.jsonl")
V2_RAW = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "../calib_da_v2/raw_outputs_da_v2.jsonl")
OLD_RAW = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "../calib_int/raw_outputs.jsonl")
CALIB = os.path.join(HERE, "../calib_int/calib_set.jsonl")

# Import scoring helpers from the runner so prefill reconstruction stays consistent.
sys.path.insert(0, HERE)
from run_calib_da_v3 import (  # noqa: E402
    PREFILL, parse_marker, parse_fallback, grade, reconstruct_reply,
    is_single_marker_line,
)

NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - r), min(1.0, c + r)]


def score_row(r, initial, apply_prefill=False):
    text = r.get("text") or ""
    full = reconstruct_reply(text) if apply_prefill else text
    pred_s, how_s = parse_marker(full)
    pred_f, how_f = parse_fallback(full)
    dec = r.get("answer_decimals", 0)
    gold = r.get("gold")
    return {
        **r,
        "pred_strict": pred_s,
        "parse_how_strict": how_s,
        "pred_fallback": pred_f,
        "parse_how_fallback": how_f,
        "correct_strict": bool(pred_s is not None and grade(pred_s, gold, dec)),
        "correct_fallback": bool(pred_f is not None and grade(pred_f, gold, dec)),
        # 'correct' kept as fallback so older gap helpers can still run if needed
        "correct": bool(pred_f is not None and grade(pred_f, gold, dec)),
        "echo_initial": bool(pred_f is not None and initial is not None and grade(pred_f, initial, dec)),
        "trunc": r.get("finish_reason") == "length",
        "single_marker_line": is_single_marker_line(full),
        "initial": initial,
        "apply_prefill": apply_prefill,
    }


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def maybe_load_jsonl(path):
    if not path or not os.path.exists(path):
        print(f"note: missing comparison file, skipping: {path}")
        return None
    return load_jsonl(path)


def fmt_ci(acc, ci):
    if acc is None or ci is None or ci[0] is None:
        return "—"
    return f"{acc:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"


def cell_stats(rows):
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "correct_strict": 0,
            "acc_strict": None,
            "wilson95_strict": [None, None],
            "correct_fallback": 0,
            "acc_fallback": None,
            "wilson95_fallback": [None, None],
            "parse_fail_strict": 0,
            "parse_fail_strict_rate": None,
            "parse_fallback": 0,
            "parse_fallback_rate": None,
            "echo_initial": 0,
            "echo_initial_rate": None,
            "trunc_rate": None,
            "mean_out_tokens": None,
            "single_marker_line": 0,
            "single_marker_line_rate": None,
        }
    ks = sum(r["correct_strict"] for r in rows)
    kf = sum(r["correct_fallback"] for r in rows)
    pf = sum(r["parse_how_strict"] == "none" for r in rows)
    fb = sum(r["parse_how_fallback"] == "fallback" for r in rows)
    echo = sum(r["echo_initial"] for r in rows)
    tr = sum(r["trunc"] for r in rows)
    sl = sum(r["single_marker_line"] for r in rows)
    return {
        "n": n,
        "correct_strict": ks,
        "acc_strict": ks / n,
        "wilson95_strict": wilson(ks, n),
        "correct_fallback": kf,
        "acc_fallback": kf / n,
        "wilson95_fallback": wilson(kf, n),
        "parse_fail_strict": pf,
        "parse_fail_strict_rate": pf / n,
        "parse_fallback": fb,
        "parse_fallback_rate": fb / n,
        "echo_initial": echo,
        "echo_initial_rate": echo / n,
        "trunc_rate": tr / n,
        "mean_out_tokens": sum(r.get("n_out_tokens", 0) for r in rows) / n,
        "single_marker_line": sl,
        "single_marker_line_rate": sl / n,
    }


def empty_stats():
    return cell_stats([])


def gap_between(da_map, cot_map, qids, field="correct_strict"):
    common = [q for q in qids if q in da_map and q in cot_map]
    if not common:
        return None
    diffs = [int(cot_map[q]["correct_fallback"]) - int(da_map[q][field]) for q in common]
    mu = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in diffs) / max(1, len(diffs) - 1))
    se = sd / math.sqrt(len(diffs))
    return {
        "n_paired": len(diffs),
        "field": field,
        "gap": mu,
        "ci95_normal": [mu - 1.96 * se, mu + 1.96 * se],
    }


def main():
    calib = {}
    if os.path.exists(CALIB):
        calib = {r["qid"]: r for r in load_jsonl(CALIB)}
    else:
        print(f"note: missing calib_set, initials only from raw rows: {CALIB}")

    def initial_of(r):
        if r.get("initial") is not None:
            return r["initial"]
        c = calib.get(r["qid"])
        if c:
            return c["chains"][0]["initial"]
        return None

    if not os.path.exists(NEW_RAW):
        print(f"ERROR: DA v3 raw file not found: {NEW_RAW}")
        sys.exit(1)

    new_rows = [
        score_row(r, initial_of(r), apply_prefill=True)
        for r in load_jsonl(NEW_RAW)
        if r.get("mode", "DA") == "DA"
    ]

    v2_raw = maybe_load_jsonl(V2_RAW)
    old_raw = maybe_load_jsonl(OLD_RAW)

    v2_rows = [
        score_row(r, initial_of(r), apply_prefill=False)
        for r in (v2_raw or [])
        if r.get("mode", "DA") == "DA"
    ]
    old_da = [
        score_row(r, initial_of(r), apply_prefill=False)
        for r in (old_raw or [])
        if r.get("mode") == "DA"
    ]
    old_cot = [
        score_row(r, initial_of(r), apply_prefill=False)
        for r in (old_raw or [])
        if r.get("mode") == "CoT"
    ]

    new_by = defaultdict(list)
    v2_by = defaultdict(list)
    old_da_by = defaultdict(list)
    cot_by = defaultdict(list)
    for r in new_rows:
        new_by[r["d"]].append(r)
    for r in v2_rows:
        v2_by[r["d"]].append(r)
    for r in old_da:
        old_da_by[r["d"]].append(r)
    for r in old_cot:
        cot_by[r["d"]].append(r)

    depths = sorted(new_by)
    table = []
    for d in depths:
        nd = {r["qid"]: r for r in new_by[d]}
        v2d = {r["qid"]: r for r in v2_by.get(d, [])}
        od = {r["qid"]: r for r in old_da_by.get(d, [])}
        cd = {r["qid"]: r for r in cot_by.get(d, [])}
        qids = sorted(nd)
        new_s = cell_stats([nd[q] for q in qids])
        v2_s = cell_stats([v2d[q] for q in qids if q in v2d]) if v2d else empty_stats()
        old_s = cell_stats([od[q] for q in qids if q in od]) if od else empty_stats()
        cot_s = cell_stats([cd[q] for q in qids if q in cd]) if cd else empty_stats()

        both_v2 = [q for q in qids if q in v2d]
        both_old = [q for q in qids if q in od]

        def paired_delta(other_map, new_field, other_field):
            keys = [q for q in qids if q in other_map]
            if not keys:
                return None
            return (
                sum(int(nd[q][new_field]) for q in keys)
                - sum(int(other_map[q][other_field]) for q in keys)
            ) / len(keys)

        row = {
            "d": d,
            "DA_v3": new_s,
            "DA_v2": v2_s,
            "DA_old": old_s,
            "CoT": cot_s,
            "DA_v3_strict_minus_DA_v2_strict": paired_delta(v2d, "correct_strict", "correct_strict"),
            "DA_v3_fallback_minus_DA_v2_fallback": paired_delta(v2d, "correct_fallback", "correct_fallback"),
            "DA_v3_strict_minus_DA_old_fallback": paired_delta(od, "correct_strict", "correct_fallback"),
            "gap_CoT_minus_DA_v3_strict": gap_between(nd, cd, qids, "correct_strict"),
            "gap_CoT_minus_DA_v3_fallback": gap_between(nd, cd, qids, "correct_fallback"),
            "gap_CoT_minus_DA_v2_fallback": gap_between(v2d, cd, qids, "correct_fallback"),
            "gap_CoT_minus_DA_old": gap_between(od, cd, qids, "correct_fallback"),
            "n_paired_v2": len(both_v2),
            "n_paired_old": len(both_old),
        }
        table.append(row)

    T = {r["d"]: r for r in table}

    def acc_strict(d):
        return T[d]["DA_v3"]["acc_strict"] if d in T else None

    checks = {}
    if 1 in T:
        da1 = acc_strict(1)
        checks["DA_v3_strict(1) in [0.80,0.95]  (headline)"] = {
            "value": da1, "pass": da1 is not None and 0.80 <= da1 <= 0.95
        }
        checks["DA_v3_strict(1) in heuristic p1 window [0.955,0.986]"] = {
            "value": da1, "pass": da1 is not None and 0.955 <= da1 <= 0.986
        }
    if 20 in T:
        da20 = acc_strict(20)
        cot20 = T[20]["CoT"]["acc_fallback"] if T[20]["CoT"]["n"] else None
        checks["DA_v3_strict(20) <= 0.10"] = {
            "value": da20, "pass": da20 is not None and da20 <= 0.10
        }
        checks["CoT(20) in [0.40,0.75] (from previous run, hard)"] = {
            "value": cot20, "pass": cot20 is not None and 0.40 <= cot20 <= 0.75
        }
        g1 = T[1]["gap_CoT_minus_DA_v3_strict"] if 1 in T else None
        g20 = T[20]["gap_CoT_minus_DA_v3_strict"]
        if g1 and g20:
            checks["gap_v3_strict(20)-gap_v3_strict(1) >= 0.30"] = {
                "value": round(g20["gap"] - g1["gap"], 6),
                "pass": (g20["gap"] - g1["gap"]) >= 0.30,
            }

    out = {
        "new_raw": os.path.abspath(NEW_RAW),
        "v2_raw": os.path.abspath(V2_RAW) if v2_raw is not None else None,
        "old_raw": os.path.abspath(OLD_RAW) if old_raw is not None else None,
        "n_new": len(new_rows),
        "prefill": PREFILL,
        "headline_metric": "strict (last/only #### N after reconstructing the '#### ' prefill)",
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
        few = json.load(open(few_path, encoding="utf-8"))
        out["few_shot_system"] = few.get("system")
        out["few_shot_n"] = len(few.get("examples", []))

    out_dir = os.path.dirname(os.path.abspath(NEW_RAW)) or HERE
    metrics_path = os.path.join(out_dir, "metrics_da_v3.json")
    plot_path = os.path.join(out_dir, "da_v3_compare.png")
    json.dump(out, open(metrics_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("=== DA v3 vs DA v2 / old DA (same items) and existing CoT ===")
    print("Headline is DA_v3 STRICT (#### N only). fallback is the v2-style last-number parse.")
    hdr = (f"{'d':>3} {'n':>4}  {'DA_v3_strict':>22}  {'DA_v3_fallback':>22}  "
           f"{'DA_v2_strict':>22}  {'DA_v2_fallback':>22}  {'DA_old':>22}  "
           f"{'CoT':>22}  {'trunc':>7}  {'echo':>7}  {'single':>7}  {'mean_tok':>8}")
    print(hdr)
    for r in table:
        n = r["DA_v3"]["n"]
        print(
            f"{r['d']:>3} {n:>4}  "
            f"{fmt_ci(r['DA_v3']['acc_strict'], r['DA_v3']['wilson95_strict']):>22}  "
            f"{fmt_ci(r['DA_v3']['acc_fallback'], r['DA_v3']['wilson95_fallback']):>22}  "
            f"{fmt_ci(r['DA_v2']['acc_strict'], r['DA_v2']['wilson95_strict']):>22}  "
            f"{fmt_ci(r['DA_v2']['acc_fallback'], r['DA_v2']['wilson95_fallback']):>22}  "
            f"{fmt_ci(r['DA_old']['acc_fallback'], r['DA_old']['wilson95_fallback']):>22}  "
            f"{fmt_ci(r['CoT']['acc_fallback'], r['CoT']['wilson95_fallback']):>22}  "
            f"{(r['DA_v3']['trunc_rate'] if r['DA_v3']['trunc_rate'] is not None else 0):>7.3f}  "
            f"{(r['DA_v3']['echo_initial_rate'] if r['DA_v3']['echo_initial_rate'] is not None else 0):>7.3f}  "
            f"{(r['DA_v3']['single_marker_line_rate'] if r['DA_v3']['single_marker_line_rate'] is not None else 0):>7.3f}  "
            f"{(r['DA_v3']['mean_out_tokens'] if r['DA_v3']['mean_out_tokens'] is not None else 0):>8.1f}"
        )
    print("\nWilson 95% CIs in brackets. gap checks use CoT acc − DA_v3_strict acc (paired qids).")
    print("trunc / echo / single / mean_tok are DA v3 only. DA_old and CoT use the v2 fallback parse.")
    print("\nTask A checks (DA v3 STRICT + previous CoT):")
    for k, v in checks.items():
        flag = "PASS" if v["pass"] else "FAIL"
        print(f"  {flag}  {k}: {v['value']}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        xs = [r["d"] for r in table]

        def errbars(accs, cis, color, label, marker, linestyle):
            # skip series that are entirely missing
            if all(a is None for a in accs):
                return
            ys, lo, hi, xok = [], [], [], []
            for x, a, c in zip(xs, accs, cis):
                if a is None or c is None or c[0] is None:
                    continue
                xok.append(x)
                ys.append(a)
                lo.append(max(0.0, a - c[0]))
                hi.append(max(0.0, c[1] - a))
            if not ys:
                return
            ax.errorbar(xok, ys, yerr=[lo, hi], marker=marker, linestyle=linestyle,
                        capsize=3, label=label, color=color)

        errbars(
            [r["DA_old"]["acc_fallback"] for r in table],
            [r["DA_old"]["wilson95_fallback"] for r in table],
            "tab:orange", "DA old (fallback)", "s", "--",
        )
        errbars(
            [r["DA_v2"]["acc_fallback"] for r in table],
            [r["DA_v2"]["wilson95_fallback"] for r in table],
            "tab:pink", "DA v2 fallback", "v", "--",
        )
        errbars(
            [r["DA_v2"]["acc_strict"] for r in table],
            [r["DA_v2"]["wilson95_strict"] for r in table],
            "tab:purple", "DA v2 strict", "^", ":",
        )
        errbars(
            [r["DA_v3"]["acc_fallback"] for r in table],
            [r["DA_v3"]["wilson95_fallback"] for r in table],
            "tab:red", "DA v3 fallback", "o", "--",
        )
        errbars(
            [r["DA_v3"]["acc_strict"] for r in table],
            [r["DA_v3"]["wilson95_strict"] for r in table],
            "tab:green", "DA v3 strict (headline)", "o", "-",
        )
        errbars(
            [r["CoT"]["acc_fallback"] for r in table],
            [r["CoT"]["wilson95_fallback"] for r in table],
            "tab:blue", "CoT (previous)", "D", "-",
        )
        ax.set_xlabel("depth d")
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(xs)
        ax.set_title("Task A DA v3 (strict headline) vs DA v2 / old DA / CoT")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"wrote {metrics_path} and {plot_path}")
    except Exception as e:
        print(f"wrote {metrics_path} (plot skipped: {e})")


if __name__ == "__main__":
    main()
