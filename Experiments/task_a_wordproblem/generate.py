#!/usr/bin/env python3
"""条件分支小学应用题生成器（cond_wordproblem_v1）。

每一步都是同一种事件：把当前数量与阈值 T 比较，大于 T 则减去 a，小于 T 则加上 b。
生成时保证当前值不等于 T。运算只有加减，没有乘法。金额在内部按 10**decimals
的整数单位计算（例如两位小数用「分」），只在写出题面时格式化成小数。

用法：
  python generate.py --mode samples [--show]     # d∈{1,3,5,10,20} 与 w∈{1,3,5,10} 各 5 题
  python generate.py --mode full                 # 串行 d=1..20×100 + 并行 w=1..10×100
  python generate.py --mode calib --seed 1234    # 校准集：d∈{1,2,3,5,10,15,20}×100
  python generate.py --mode check --n 500        # 大样本捷径统计
选项：--seed 7404 --decimals 0|1|2 --out PATH
运算只允许 addsub。--ops mixed / 乘法已禁用。
"""
import argparse, hashlib, json, random, re
from collections import Counter
from decimal import Decimal, InvalidOperation

VERSION = "cond_wordproblem_v1"
BASE_CFG = {
    "family": "cond_wordproblem",
    "ops": "addsub",            # 唯一运算：加减。乘法 / mixed 已删除
    "decimals": 0,              # 0 = integers; 1/2 = money with 1/2 decimal places
    "start_min": 20, "start_max": 120,   # in whole units
    "val_min": 0, "val_max": 200,
    "delta_min": 2, "delta_max": 29,
    "margin_min": 3, "margin_max": 40,   # |amount - T| in whole units
    "p_distractor": 1.0,
    # 不再参与生成。保留该键是为了与已交付样例、校准集 config 字段逐字节一致。
    "p_mul2": 0.2,
}

NAMES = [("Mia","she","her"),("Leo","he","his"),("Ava","she","her"),("Noah","he","his"),
         ("Emma","she","her"),("Liam","he","his"),("Zoe","she","her"),("Omar","he","his"),
         ("Lily","she","her"),("Ethan","he","his"),("Grace","she","her"),("Ryan","he","his"),
         ("Nora","she","her"),("Sam","he","his"),("Ella","she","her"),("Jack","he","his"),
         ("Ruby","she","her"),("Owen","he","his"),("Ivy","she","her"),("Ben","he","his"),
         ("Chloe","she","her"),("Max","he","his"),("Hana","she","her"),("Kai","he","his")]
ITEMS = ["stickers","marbles","apples","stamps","pencils","cookies","seashells","baseball cards",
         "beads","coins","crayons","toy cars","candies","postcards","shells","buttons"]
OTHERS = ["a friend","a classmate","POS neighbor","POS cousin","a younger kid","grandma","POS sister"]
SEQ = ["First", "Then", "Next", "After that", "Later", "After this"]

LOSS = ["{p} gives {a} of them to {o}", "{p} sells {a} of them", "{p} gives away {a}",
        "{p} uses up {a}", "{p} trades away {a}", "{p} donates {a} to {o}", "{p} loses {a}"]
GAIN = ["{p} buys {b} more", "{p} gets {b} more from {o}", "{p} finds {b} more",
        "{p} receives {b} more", "{p} wins {b} more", "{p} collects {b} more"]
# money (decimal) phrasing
M_LOSS = ["{p} spends ${a} on a snack", "{p} pays ${a} for a book", "{p} gives ${a} to {o}",
          "{p} spends ${a} on stickers", "{p} pays ${a} for a bus ticket", "{p} donates ${a} to charity"]
M_GAIN = ["{p} earns ${b} more", "{p} gets ${b} from {o}", "{p} finds ${b} more",
          "{p} earns ${b} by washing cars", "{p} receives ${b} from {o}", "{p} saves ${b} more"]

COND = [
    "If {n} has more than {T} {it}, {loss}; otherwise, {gain}.",
    "If {n} has fewer than {T} {it}, {gain}; otherwise, {loss}.",
    "{n} counts {pos} {it}. If there are more than {T}, {loss}. If not, {gain}.",
    "{n} checks {pos} {it}: if {p} has fewer than {T}, {gain}; if {p} has more, {loss}.",
]
M_COND = [
    "If {n} has more than ${T}, {loss}; otherwise, {gain}.",
    "If {n} has less than ${T}, {gain}; otherwise, {loss}.",
    "{n} counts {pos} money. If it is more than ${T}, {loss}. If not, {gain}.",
    "{n} checks {pos} wallet: if {p} has less than ${T}, {gain}; if {p} has more, {loss}.",
]
DISTRACT = ["{n}'s brother has {x} {it2}.", "The store is {x} blocks from {pos} house.",
            "{n}'s class has {x} students.", "{n} is {age} years old.",
            "A box of {it2} at the shop has {x} pieces.", "{n}'s best friend owns {x} {it2}."]


# ---------------- exact number helpers ----------------
class Num:
    def __init__(self, decimals):
        self.dec = decimals
        self.scale = 10 ** decimals

    def fmt(self, x):  # x: integer in scaled units
        if self.dec == 0:
            return str(x)
        return f"{x // self.scale}.{x % self.scale:0{self.dec}d}"

    def rand_amount(self, rng, lo, hi):
        """Random amount in [lo, hi] whole units; with decimals, fractional part is
        never zero (so the decimal really has to be handled)."""
        if self.dec == 0:
            return rng.randint(lo, hi)
        whole = rng.randint(lo, hi - 1)
        frac = rng.randint(1, self.scale - 1)
        return whole * self.scale + frac


def parse_number(s):
    """Normalize a model/printed number: strips $, commas, spaces; returns Decimal or None."""
    if s is None:
        return None
    t = str(s).strip().replace("$", "").replace(",", "").replace(" ", "")
    try:
        return Decimal(t)
    except InvalidOperation:
        return None


def grade(pred, gold, decimals=0):
    """Numeric grading with normalization: 3.5 == 3.50 == $3.50; tolerance = half the
    smallest printed unit (0.5 for integers, 0.005 for cents). Parallel answers are
    comma-separated -> every element must match, in order."""
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


# ---------------- generation ----------------
def assert_addsub(cfg):
    """乘法路径已删除。mixed 或任何非 addsub 设置都直接拒绝。"""
    if cfg.get("ops") != "addsub":
        raise ValueError("仅支持 ops='addsub'（加减）。乘法 / mixed 已禁用。")


def gen_chain(rng, d, cfg, N):
    assert_addsub(cfg)
    S = N.scale
    v = N.rand_amount(rng, cfg["start_min"], cfg["start_max"])
    start, steps = v, []
    for _ in range(d):
        for _try in range(500):
            branch = rng.choice(["loss", "gain"])
            m = rng.randint(cfg["margin_min"], cfg["margin_max"])
            # threshold is always a whole number (readable "$45"), strictly on one side of v
            T = (v // S - m + (1 if v % S else 0)) if branch == "loss" else (v // S + m)
            T *= S
            if not (5 * S <= T <= (cfg["val_max"] - 1) * S) or T == v:
                continue
            if (branch == "loss") != (v > T):
                continue
            a = N.rand_amount(rng, cfg["delta_min"], cfg["delta_max"])
            b = N.rand_amount(rng, cfg["delta_min"], cfg["delta_max"])
            if branch == "loss":
                op, nv = "sub", v - a
                if nv < cfg["val_min"] * S:
                    continue
            else:
                op = "add"
                nv = v + b
                if nv > cfg["val_max"] * S:
                    continue
            break
        else:
            raise RuntimeError("could not place step")
        steps.append({"T": T, "branch": branch, "a": a, "b": b, "op": op, "before": v, "after": nv})
        v = nv
    return {"initial": start, "steps": steps, "trace": [s["after"] for s in steps], "final": v}


def solve_chain(start, steps, ignore=None):
    """Independent solver from printed parameters only (T, a, b, gain_op)."""
    v = start
    for s in steps:
        if ignore == "always_loss":
            take = "loss"
        elif ignore == "always_gain":
            take = "gain"
        else:
            assert v != s["T"]
            take = "loss" if v > s["T"] else "gain"
        if take == "loss":
            v -= s["a"]
        else:
            if s.get("gain_op") != "add":
                raise ValueError("乘法步骤已禁用")
            v = v + s["b"]
    return v


def render_chain(rng, chain, person, item, cfg, N):
    n, p, pos = person
    money = cfg["decimals"] > 0
    f = N.fmt
    sents = [f"{n} has ${f(chain['initial'])}." if money else f"{n} has {f(chain['initial'])} {item}."]
    sol, v = [], chain["initial"]
    for k, s in enumerate(chain["steps"]):
        tpl = rng.choice(M_COND if money else COND)
        o = rng.choice(OTHERS).replace("POS", pos)
        loss = rng.choice(M_LOSS if money else LOSS).format(p=p, a=f(s["a"]), o=o)
        # 增加分支（含未走到的那个子句）只能是加法。
        gop = s["op"] if s["branch"] == "gain" else "add"
        if gop != "add":
            raise ValueError("乘法步骤已禁用")
        gain = rng.choice(M_GAIN if money else GAIN).format(p=p, b=f(s["b"]), o=o)
        s["gain_op"] = gop
        Tstr = str(s["T"] // N.scale)
        body = tpl.format(n=n, T=Tstr, it=item, loss=loss, gain=gain, p=p, pos=pos)
        lead = SEQ[k] if k < 2 else rng.choice(SEQ[1:])
        if body.startswith(n):
            sents.append(f"{lead}, {body}")
        else:
            sents.append(f"{lead}, {body[0].lower()}{body[1:]}")
        b4, af = f(v), f(s["after"])
        if s["branch"] == "loss":
            sol.append(f"{b4} > {Tstr}, so {n} loses {f(s['a'])}: {b4}-{f(s['a'])}=<<{b4}-{f(s['a'])}={af}>>{af}")
        else:
            sol.append(f"{b4} < {Tstr}, so {n} gains {f(s['b'])}: {b4}+{f(s['b'])}=<<{b4}+{f(s['b'])}={af}>>{af}")
        v = s["after"]
    return sents, sol


def distractor(rng, person, item):
    n, p, pos = person
    it2 = rng.choice([i for i in ITEMS if i != item])
    return rng.choice(DISTRACT).format(n=n, x=rng.randint(3, 60), age=rng.randint(7, 12), it2=it2, pos=pos)


def build_item(rng, kind, x, i, cfg, seed):
    N = Num(cfg["decimals"])
    money = cfg["decimals"] > 0
    item = "dollars" if money else rng.choice(ITEMS)
    lengths = [x] if kind == "serial" else [1] + [2] * (x - 1)
    people = rng.sample(NAMES, len(lengths))
    paras, sols, chains = [], [], []
    for j, (L, person) in enumerate(zip(lengths, people)):
        for _ in range(10000):   # chain-level rejection: see chain_ok()
            ch = gen_chain(rng, L, cfg, N)
            sents, sol = render_chain(rng, ch, person, item, cfg, N)
            if chain_ok(ch, set(numbers_in(" ".join(sents), N.scale))):
                break
        else:
            raise RuntimeError("chain rejection exhausted")
        if j == 0 and rng.random() < cfg["p_distractor"]:   # ~1 distractor per problem
            sents.insert(rng.randint(1, len(sents)), distractor(rng, person, item))
        paras.append(" ".join(sents))
        sols.append((person[0], sol))
        chains.append(ch)
    unit = "how much money" if money else f"how many {item}"
    if kind == "serial":
        n = people[0][0]
        q = paras[0] + (f"\nHow much money does {n} have at the end?" if money
                        else f"\nHow many {item} does {n} have at the end?")
        solution = "\n".join(sols[0][1]) + f"\n#### {N.fmt(chains[0]['final'])}"
    else:
        names = ", ".join(p[0] for p in people)
        q = "\n".join(paras) + (f"\nAt the end, {unit} does each person have? "
                                f"Give the numbers in this order: {names}.")
        lines = []
        for n, sol in sols:
            lines += [f"{n}:"] + sol
        solution = "\n".join(lines) + "\n#### " + ",".join(N.fmt(c["final"]) for c in chains)
    # self-check: independent re-solve, bounds, no ties, no negatives
    for ch in chains:
        assert solve_chain(ch["initial"], ch["steps"]) == ch["final"], "solver mismatch"
        allv = [ch["initial"]] + ch["trace"]
        assert all(cfg["val_min"] * N.scale <= v <= cfg["val_max"] * N.scale for v in allv), "bounds"
        assert all(s["before"] != s["T"] for s in ch["steps"]), "tie"
    ans_str = [N.fmt(c["final"]) for c in chains]
    W = sum(lengths)
    return {
        "version": VERSION, "seed": seed, "config": dict(cfg), "split": "sample",
        "kind": kind, "d": x if kind == "serial" else max(lengths), "w": 1 if kind == "serial" else x,
        "i": i, "task_family": cfg["family"],
        "qid": hashlib.sha256(q.encode()).hexdigest(),
        "W": W, "W_atomic": W, "matched_serial_d": W, "chain_lengths": lengths,
        "ans": ",".join(ans_str),
        "answers": [int(a) if N.dec == 0 else float(a) for a in ans_str],
        "answer_decimals": N.dec,
        "grading": "numeric; strip $ and commas; |pred-gold| <= 0.5*10^-decimals; parallel: all in order",
        "q": q, "solution": solution,
        "chains": [{"name": p[0], "item": item, "initial": N.fmt(c["initial"]), "steps": len(c["steps"]),
                    "events": [{"T": N.fmt(s["T"]), "a": N.fmt(s["a"]), "b": N.fmt(s["b"]),
                                "gain_op": s["gain_op"], "taken": s["branch"],
                                "before": N.fmt(s["before"]), "after": N.fmt(s["after"])}
                               for s in c["steps"]],
                    "trace": [N.fmt(t) for t in c["trace"]], "final": N.fmt(c["final"]),
                    "ans": N.fmt(c["final"])}
                   for p, c in zip(people, chains)],
        "meta": {"START": [N.fmt(c["initial"]) for c in chains], "OPS": lengths,
                 "N_LOSS": sum(s["branch"] == "loss" for c in chains for s in c["steps"]),
                 "N_GAIN": sum(s["branch"] == "gain" for c in chains for s in c["steps"])},
        "_raw": chains,  # scaled-int internals, stripped before writing
    }


# ---------------- shortcut checks ----------------
def numbers_in(q, scale):
    return [int((Decimal(x) * scale).to_integral_value()) for x in re.findall(r"\d+(?:\.\d+)?", q)]


def chain_ok(c, nums):
    ans, ev = c["final"], c["steps"]
    if ans in nums:
        return False
    if len(ev) >= 3:   # for d<=2 an 'ignore the condition' path is a legit 1/2..1/4 coincidence
        if ans == c["initial"]:
            return False
        for mode in ("always_loss", "always_gain"):
            if solve_chain(c["initial"], ev, mode) == ans:
                return False
    return True


def acceptable(r):
    """Item-level: no chain's answer equals a number printed in its own paragraph
    (incl. the distractor). Chain-level checks already ran in build_item."""
    sc = 10 ** r["answer_decimals"]
    paras = r["q"].split("\n")[:len(r["_raw"])]
    return all(c["final"] not in set(numbers_in(p, sc)) for c, p in zip(r["_raw"], paras))


def shortcut_stats(recs):
    st = Counter()
    for r in recs:
        if r["kind"] != "serial":
            continue
        c = r["_raw"][0]
        ans, ev = c["final"], c["steps"]
        nums = numbers_in(r["q"], 10 ** r["answer_decimals"])
        st["n"] += 1
        st["sum_all_numbers"] += (sum(nums) == ans)
        both = c["initial"] + sum(e["b"] for e in ev if e["gain_op"] == "add") - sum(e["a"] for e in ev)
        st["apply_both_clauses"] += (both == ans)
        st["always_loss"] += (solve_chain(c["initial"], ev, "always_loss") == ans)
        st["always_gain"] += (solve_chain(c["initial"], ev, "always_gain") == ans)
        st["ans_in_text"] += (ans in nums)
        T, a, b = ev[-1]["T"], ev[-1]["a"], ev[-1]["b"]
        st["last_threshold_guess"] += (ans in (T - a, T + b, T))
        st["start_unchanged"] += (ans == c["initial"])
        for k in range(len(ev)):  # one wrong branch at step k
            v = c["initial"]
            for j, e in enumerate(ev):
                take = "loss" if v > e["T"] else "gain"
                if j == k:
                    take = "gain" if take == "loss" else "loss"
                if e["gain_op"] != "add" and take == "gain":
                    raise ValueError("乘法步骤已禁用")
                v = v - e["a"] if take == "loss" else v + e["b"]
            st["flip_still_correct"] += (v == ans)
        st["flip_events"] += len(ev)
        st["loss"] += sum(e["branch"] == "loss" for e in ev)
        st["steps"] += len(ev)
    n = max(1, st["n"])
    out = {k: round(st[k] / n, 4) for k in ["sum_all_numbers", "apply_both_clauses", "always_loss",
                                            "always_gain", "ans_in_text", "last_threshold_guess",
                                            "start_unchanged"]}
    out["single_branch_error_still_correct"] = round(st["flip_still_correct"] / max(1, st["flip_events"]), 4)
    out["loss_branch_share"] = round(st["loss"] / max(1, st["steps"]), 4)
    out["n_serial"] = st["n"]
    return out


def make(seed, cfg, plan, split, filt=True):
    recs, seen = [], set()
    for kind, x, n in plan:
        for i in range(n):
            rng = random.Random(f"{seed}|{VERSION}|{cfg['name']}|{kind}|{x}|{i}")
            for _ in range(10000):
                r = build_item(rng, kind, x, i, cfg, seed)
                if r["qid"] not in seen and (not filt or acceptable(r)):
                    break
            else:
                raise RuntimeError(f"rejection loop exhausted at {kind} {x} {i}")
            seen.add(r["qid"])
            r["split"] = split
            recs.append(r)
    return recs


def strip(r):
    return {k: v for k, v in r.items() if k != "_raw"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="samples", choices=["samples", "full", "calib", "check"])
    ap.add_argument("--seed", type=int, default=7404)
    ap.add_argument("--ops", default="addsub", choices=["addsub"],
                    help="只允许加减。乘法 / mixed 已禁用。")
    ap.add_argument("--decimals", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-filter", action="store_true", help="disable rejection filter (to measure raw rates)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    assert_addsub({"ops": a.ops})
    cfg = dict(BASE_CFG, ops=a.ops, decimals=a.decimals, name=f"cond_{a.ops}_dec{a.decimals}")
    if a.mode == "samples":
        plan = [("serial", d, 5) for d in (1, 3, 5, 10, 20)] + [("parallel", w, 5) for w in (1, 3, 5, 10)]
        out = a.out or "samples.jsonl"
    elif a.mode == "full":
        plan = [("serial", d, 100) for d in range(1, 21)] + [("parallel", w, 100) for w in range(1, 11)]
        out = a.out or "full.jsonl"
    elif a.mode == "calib":
        # 与 calib_int/ 已交付的 700 题相同：seed 须为 1234 才能复现该文件。
        plan = [("serial", d, 100) for d in (1, 2, 3, 5, 10, 15, 20)]
        out = a.out or "calib_set.jsonl"
    else:
        plan = [("serial", d, a.n) for d in range(1, 21)]
        out = None
    recs = make(a.seed, cfg, plan, a.mode, filt=not a.no_filter)
    if a.mode != "check":
        again = make(a.seed, cfg, plan, a.mode, filt=not a.no_filter)
        assert [r["qid"] for r in again] == [r["qid"] for r in recs], "non-deterministic"
        # grading self-test on gold answers
        assert all(grade(r["ans"], r["ans"], r["answer_decimals"]) for r in recs)
    if out:
        with open(out, "w") as f:
            for r in recs:
                f.write(json.dumps(strip(r), ensure_ascii=False) + "\n")
        print(f"wrote {len(recs)} records -> {out} (deterministic, all gold re-solved)")
    print("overall (serial):", json.dumps(shortcut_stats(recs)))
    per_d = {}
    for r in recs:
        if r["kind"] == "serial":
            per_d.setdefault(r["d"], []).append(r)
    for d in sorted(per_d):
        rs = per_d[d]
        s = shortcut_stats(rs)
        print(f" d={d:2d} n={len(rs)} chars~{sum(len(r['q']) for r in rs)//len(rs)} "
              f"both={s['apply_both_clauses']:.3f} aL={s['always_loss']:.3f} aG={s['always_gain']:.3f} "
              f"in_text={s['ans_in_text']:.3f} lastT={s['last_threshold_guess']:.3f} "
              f"flip_ok={s['single_branch_error_still_correct']:.3f}")
    par = [r for r in recs if r["kind"] == "parallel"]
    if par:
        print("parallel (w,W,chars):", sorted({(r["w"], r["W"], len(r["q"]) // 50 * 50) for r in par})[:12])
    if a.show:
        for d in (1, 3, 5, 10, 20):
            r = next((r for r in recs if r["kind"] == "serial" and r["d"] == d), None)
            if r:
                print(f"\n===== serial d={d} ans={r['ans']} =====\n{r['q']}\n--- gold ---\n{r['solution']}")
        r = next((x for x in par if x["w"] == 3), None)
        if r:
            print(f"\n===== parallel w=3 W={r['W']} ans={r['ans']} =====\n{r['q']}\n--- gold ---\n{r['solution']}")


if __name__ == "__main__":
    main()
