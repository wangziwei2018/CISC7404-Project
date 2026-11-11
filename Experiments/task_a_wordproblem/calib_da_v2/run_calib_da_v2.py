#!/usr/bin/env python3
"""DA-only recalibration with the v2 prompt and two d=1 few-shot examples.

Input : calib_da_v2_slim.jsonl, few_shot.json  (cwd)
Output: raw_outputs_da_v2.jsonl, env.json
Also prints per-d accuracy, parse-fail rate, and the fraction of outputs
equal to the item's initial amount.

Model : Qwen/Qwen3-4B-Instruct-2507 pinned to
        cdbee75f17c01a7cc42f958dc650907174af0554
Sampling matches calib_int / experiment_task_a.ipynb:
        temperature 0.6, top_p 0.95, DA max_tokens 64.
L4 / compute capability >= 8 uses bfloat16; T4 falls back to float16.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import re
import sys
import time
from collections import defaultdict
from decimal import Decimal, InvalidOperation

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-4B-Instruct-2507")
REVISION = os.environ.get(
    "MODEL_REVISION", "cdbee75f17c01a7cc42f958dc650907174af0554"
)
def data_args(argv):
    """只保留数据文件路径。Colab 把脚本当单元格执行时，sys.argv 里会有 `-f <kernel.json>`。"""
    args = []
    skip_next = False
    for arg in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "-f":
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        args.append(arg)
    return args


_DATA_ARGS = data_args(sys.argv)
IN_PATH = _DATA_ARGS[0] if len(_DATA_ARGS) > 0 else "calib_da_v2_slim.jsonl"
OUT_PATH = _DATA_ARGS[1] if len(_DATA_ARGS) > 1 else "raw_outputs_da_v2.jsonl"
FEW_PATH = os.environ.get("FEW_SHOT", "few_shot.json")

PARAMS = {"temperature": 0.6, "top_p": 0.95, "top_k_recorded_not_passed": 20}
MAX_TOKENS = 64
SYSTEM = "Answer with only the final number, in the form #### N."
PROMPT_VERSION = "da_v2"

NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def parse(text):
    """Strict: last '#### N' line. Fallback (recorded separately): last number in text."""
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
    """Same numeric grading as generate.grade (tol = 0.5 * 10^{-decimals})."""
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


def seed_of(qid, mode="DA"):
    return int(hashlib.sha256(f"{qid}|{mode}".encode()).hexdigest()[:8], 16)


def build_messages(question, few_shot):
    msgs = [{"role": "system", "content": few_shot.get("system", SYSTEM)}]
    for ex in few_shot["examples"]:
        msgs.append({"role": "user", "content": ex["q"]})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": question})
    return msgs


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["d"]].append(r)
    print("\n=== DA v2 inline metrics ===")
    print(f"{'d':>3} {'n':>4} {'correct':>8} {'acc':>7} {'parse_fail':>11} "
          f"{'fallback':>9} {'echo_init':>10} {'trunc':>7} {'mean_tok':>9}")
    for d in sorted(by):
        g = by[d]
        n = len(g)
        k = sum(r["correct"] for r in g)
        pf = sum(r["parse_how"] == "none" for r in g)
        fb = sum(r["parse_how"] == "fallback" for r in g)
        echo = sum(r["echo_initial"] for r in g)
        tr = sum(r["trunc"] for r in g)
        mt = sum(r["n_out_tokens"] for r in g) / n if n else 0
        print(f"{d:>3} {n:>4} {k:>8} {k/n if n else 0:>7.3f} {pf/n if n else 0:>11.3f} "
              f"{fb/n if n else 0:>9.3f} {echo/n if n else 0:>10.3f} {tr/n if n else 0:>7.3f} {mt:>9.1f}")
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    echo = sum(r["echo_initial"] for r in rows)
    pf = sum(r["parse_how"] == "none" for r in rows)
    print(f"ALL {n:>4} {k:>8} {k/n if n else 0:>7.3f} {pf/n if n else 0:>11.3f} "
          f"{'':>9} {echo/n if n else 0:>10.3f}")
    print("parse_fail = no number parsed; echo_init = parsed value equals the initial amount.")


def main():
    import torch
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(IN_PATH, encoding="utf-8")]
    few_shot = json.load(open(FEW_PATH, encoding="utf-8"))
    sys_prompt = few_shot.get("system", SYSTEM)
    if sys_prompt != SYSTEM:
        print("WARNING: few_shot.json system differs from built-in SYSTEM", file=sys.stderr)
    for ex in few_shot["examples"]:
        if ex["assistant"] != f"#### {ex['ans']}":
            raise ValueError(f"few-shot assistant must be exactly '#### N', got {ex['assistant']!r}")

    cap = torch.cuda.get_device_capability(0)
    dtype = "bfloat16" if cap[0] >= 8 else "float16"
    print(f"GPU={torch.cuda.get_device_name(0)} cc={list(cap)} dtype={dtype}")
    print(f"model={MODEL} revision={REVISION} n={len(rows)} max_tokens={MAX_TOKENS}")
    print(f"system={sys_prompt!r}")

    llm = LLM(
        model=MODEL,
        revision=REVISION,
        tokenizer_revision=REVISION,
        dtype=dtype,
        max_model_len=4096,
        gpu_memory_utilization=0.88,
        seed=0,
    )
    tok = llm.get_tokenizer()
    messages = [build_messages(r["q"], few_shot) for r in rows]
    prompts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    sps = [
        SamplingParams(
            temperature=PARAMS["temperature"],
            top_p=PARAMS["top_p"],
            max_tokens=MAX_TOKENS,
            seed=seed_of(r["qid"], "DA"),
        )
        for r in rows
    ]
    t0 = time.time()
    outs = llm.generate(prompts, sps)
    el = time.time() - t0

    scored = []
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            pred, how = parse(c.text)
            item = {
                "qid": r["qid"],
                "d": r["d"],
                "i": r["i"],
                "mode": "DA",
                "gold": r["ans"],
                "answer_decimals": r["answer_decimals"],
                "initial": r["initial"],
                "text": c.text,
                "finish_reason": c.finish_reason,
                "n_out_tokens": len(c.token_ids),
                "n_prompt_tokens": len(o.prompt_token_ids),
                "seed": seed_of(r["qid"], "DA"),
                "prompt_version": PROMPT_VERSION,
            }
            item["pred"] = pred
            item["parse_how"] = how
            item["correct"] = bool(pred is not None and grade(pred, r["ans"], r["answer_decimals"]))
            item["trunc"] = c.finish_reason == "length"
            item["echo_initial"] = bool(
                pred is not None and grade(pred, r["initial"], r["answer_decimals"])
            )
            # raw file keeps the same core fields as calib_int, plus initial / prompt_version
            raw = {k: item[k] for k in (
                "qid", "d", "i", "mode", "gold", "answer_decimals", "initial",
                "text", "finish_reason", "n_out_tokens", "n_prompt_tokens",
                "seed", "prompt_version",
            )}
            f.write(json.dumps(raw, ensure_ascii=False) + "\n")
            scored.append(item)

    try:
        from huggingface_hub import model_info
        rev = model_info(MODEL, revision=REVISION).sha
    except Exception as e:
        rev = f"unresolved: {e}"

    env = {
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "model_revision": rev,
        "requested_model_revision": REVISION,
        "dtype": dtype,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(cap),
        "loading": {
            "dtype": dtype,
            "gpu_memory_utilization": 0.88,
            "max_model_len": 4096,
            "seed": 0,
            "revision": REVISION,
            "tokenizer_revision": REVISION,
        },
        "vllm": md.version("vllm"),
        "torch": torch.__version__,
        "transformers": md.version("transformers"),
        "sampling": PARAMS,
        "max_tokens": {"DA": MAX_TOKENS},
        "system_prompts": {"DA": sys_prompt},
        "few_shot": {
            "path": FEW_PATH,
            "seed": few_shot.get("seed"),
            "n_examples": len(few_shot["examples"]),
            "qids": [ex["qid"] for ex in few_shot["examples"]],
            "assistants": [ex["assistant"] for ex in few_shot["examples"]],
        },
        "depths": sorted({r["d"] for r in rows}),
        "n_jobs": len(rows),
        "generate_seconds": round(el, 1),
        "input": IN_PATH,
        "output": OUT_PATH,
    }
    json.dump(env, open("env.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(env, indent=2, ensure_ascii=False))
    summarize(scored)


if __name__ == "__main__":
    main()
