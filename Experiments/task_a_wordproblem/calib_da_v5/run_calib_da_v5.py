#!/usr/bin/env python3
"""DA-only recalibration with a role prompt and a continued assistant prefill.

Input : calib_da_v5_slim.jsonl, few_shot.json  (cwd)
Output: raw_outputs_da_v5.jsonl, env.json
Prints per-d strict accuracy (headline), fallback accuracy, finish_reason=length
rate, mean tokens, echo-of-initial rate, and single-line #### N rate.

The prompt, few-shots, prefill, and sampling match da_v4. The model changes:
  - Qwen/Qwen3-4B (April 2025 hybrid). Not Qwen3-4B-Base and not Instruct-2507.
  - apply_chat_template(..., enable_thinking=False, add_generation_prompt=False,
    continue_final_message=True). The template closes one empty <think></think>
    and then continues the prefill '#### '. Generation does not start inside <think>.
  - Sampling stays temperature 0.6, top_p 0.95, stop on newline, max_tokens=16.

Model : Qwen/Qwen3-4B pinned to
        1cfa9a7208912126459214e8b04321603b3df60c
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

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-4B")
REVISION = os.environ.get(
    "MODEL_REVISION", "1cfa9a7208912126459214e8b04321603b3df60c"
)
# Jupyter/Colab passes kernel flags like "-f <file>" in sys.argv; only accept real file args.
_ARGS = [a for a in sys.argv[1:] if a.endswith(".jsonl")]
IN_PATH = _ARGS[0] if len(_ARGS) > 0 else "calib_da_v5_slim.jsonl"
OUT_PATH = _ARGS[1] if len(_ARGS) > 1 else "raw_outputs_da_v5.jsonl"
FEW_PATH = os.environ.get("FEW_SHOT", "few_shot.json")

PARAMS = {"temperature": 0.6, "top_p": 0.95, "top_k_recorded_not_passed": 20}
MAX_TOKENS = 16
STOP = ["\n"]
PREFILL = "#### "
ROLE = "You are a math problem solver. Solve the following math problem."
FORMAT = "Your entire reply must be exactly one line in the form #### N."
SYSTEM = ROLE + " " + FORMAT
USER_REMINDER = "\n\nReply with only one line: #### N."
PROMPT_VERSION = "da_v5"
CONTINUE_FINAL_MESSAGE = True
ENABLE_THINKING = False
# Qwen3-4B's template closes an empty <think></think> before a final assistant
# prefill. Generation continues after '#### ', not inside an open think block.
ASSISTANT_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n" + PREFILL

NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"
_MARKER_RE = re.compile(r"####\s*(?:ANSWER\s*=\s*)?(" + NUM + r")")
_SINGLE_LINE_RE = re.compile(r"^####\s*" + NUM + r"$")


def with_user_reminder(question: str) -> str:
    if question.endswith(USER_REMINDER):
        return question
    return question + USER_REMINDER


def reconstruct_reply(text: str) -> str:
    """Recover the full assistant message after a '#### ' prefill.

    vLLM returns only the continuation, so a bare '77' is the intended
    completion of '#### 77'. If the model itself emitted a leading ####,
    do not prepend a second copy.
    """
    if text.lstrip().startswith("####"):
        return text
    return PREFILL + text


def parse_marker(text):
    """Strict: last/only #### N marker. No fallback."""
    m = _MARKER_RE.findall(text)
    return (m[-1], "marker") if m else (None, "none")


def parse_fallback(text):
    """Same as da_v2: last #### N if present, else last number in the text."""
    pred, how = parse_marker(text)
    if pred is not None:
        return pred, how
    nums = re.findall(NUM, text)
    return (nums[-1], "fallback") if nums else (None, "none")


def parse(text, prefill_applied=False):
    """Parser used by the runner and the dry-run.

    If prefill_applied is True, a bare number such as '77' is treated as the
    completion of the prefilled '#### ' (strict marker). A leaked paragraph
    with no #### N still falls back to the last number.
    """
    full = reconstruct_reply(text) if prefill_applied else text
    return parse_fallback(full)


def is_single_marker_line(full_text: str) -> bool:
    return _SINGLE_LINE_RE.fullmatch(full_text.strip()) is not None


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
    """Completed few-shot assistants stay '#### N'. Only the last turn is the prefill."""
    sys_prompt = few_shot.get("system", SYSTEM)
    prefill = few_shot.get("prefill", PREFILL)
    msgs = [{"role": "system", "content": sys_prompt}]
    for ex in few_shot["examples"]:
        msgs.append({"role": "user", "content": with_user_reminder(ex["q"])})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": with_user_reminder(question)})
    msgs.append({"role": "assistant", "content": prefill})
    return msgs


def render_prompt(tokenizer, messages):
    """Continue the final assistant message. add_generation_prompt stays False."""
    if messages[-1]["role"] != "assistant":
        raise ValueError("the last message must be the assistant prefill")
    if ENABLE_THINKING:
        raise RuntimeError("DA v5 requires enable_thinking=False")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=CONTINUE_FINAL_MESSAGE,
        enable_thinking=ENABLE_THINKING,
    )
    return prompt


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["d"]].append(r)
    print("\n=== DA v5 inline metrics (headline = strict #### N, no fallback) ===")
    hdr = (f"{'d':>3} {'n':>4} {'strict':>8} {'strict_acc':>11} {'fallback_acc':>13} "
           f"{'trunc':>7} {'mean_tok':>9} {'echo_init':>10} {'single_ln':>10} {'parse_fail':>11}")
    print(hdr)
    for d in sorted(by):
        g = by[d]
        n = len(g)
        ks = sum(r["correct_strict"] for r in g)
        kf = sum(r["correct_fallback"] for r in g)
        tr = sum(r["trunc"] for r in g)
        mt = sum(r["n_out_tokens"] for r in g) / n if n else 0
        echo = sum(r["echo_initial"] for r in g)
        sl = sum(r["single_marker_line"] for r in g)
        pf = sum(r["parse_how_strict"] == "none" for r in g)
        print(f"{d:>3} {n:>4} {ks:>8} {ks/n if n else 0:>11.3f} {kf/n if n else 0:>13.3f} "
              f"{tr/n if n else 0:>7.3f} {mt:>9.1f} {echo/n if n else 0:>10.3f} "
              f"{sl/n if n else 0:>10.3f} {pf/n if n else 0:>11.3f}")
    n = len(rows)
    ks = sum(r["correct_strict"] for r in rows)
    kf = sum(r["correct_fallback"] for r in rows)
    echo = sum(r["echo_initial"] for r in rows)
    sl = sum(r["single_marker_line"] for r in rows)
    pf = sum(r["parse_how_strict"] == "none" for r in rows)
    tr = sum(r["trunc"] for r in rows)
    print(f"ALL {n:>4} {ks:>8} {ks/n if n else 0:>11.3f} {kf/n if n else 0:>13.3f} "
          f"{tr/n if n else 0:>7.3f} {'':>9} {echo/n if n else 0:>10.3f} "
          f"{sl/n if n else 0:>10.3f} {pf/n if n else 0:>11.3f}")
    print("strict_acc = last/only #### N (after reconstructing the '#### ' prefill); "
          "fallback_acc = v2-style last-number parse, for comparison only.")
    print("trunc = finish_reason==length; echo_init = parsed strict/fallback value equals "
          "the initial amount; single_ln = full reply is exactly one #### N line.")
    print("parse_fail = no #### N marker after prefill reconstruction.")


def main():
    import torch
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(IN_PATH, encoding="utf-8")]
    few_shot = json.load(open(FEW_PATH, encoding="utf-8"))
    sys_prompt = few_shot.get("system", SYSTEM)
    if sys_prompt != SYSTEM:
        print("WARNING: few_shot.json system differs from built-in SYSTEM", file=sys.stderr)
    reminder = few_shot.get("user_reminder", USER_REMINDER)
    if reminder != USER_REMINDER:
        print("WARNING: few_shot.json user_reminder differs from built-in", file=sys.stderr)
    prefill = few_shot.get("prefill", PREFILL)
    if prefill != PREFILL:
        print("WARNING: few_shot.json prefill differs from built-in PREFILL", file=sys.stderr)
    if len(few_shot["examples"]) != 4:
        raise ValueError(f"expected 4 few-shot examples, got {len(few_shot['examples'])}")
    for ex in few_shot["examples"]:
        if ex["assistant"] != f"#### {ex['ans']}":
            raise ValueError(f"few-shot assistant must be exactly '#### N', got {ex['assistant']!r}")

    cap = torch.cuda.get_device_capability(0)
    dtype = "bfloat16" if cap[0] >= 8 else "float16"
    print(f"GPU={torch.cuda.get_device_name(0)} cc={list(cap)} dtype={dtype}")
    if MODEL != "Qwen/Qwen3-4B" or "Base" in MODEL or "Instruct-2507" in MODEL:
        raise RuntimeError(f"DA v5 model must be Qwen/Qwen3-4B, got {MODEL}")
    print(f"model={MODEL} revision={REVISION} enable_thinking={ENABLE_THINKING} n={len(rows)} max_tokens={MAX_TOKENS} stop={STOP}")
    print(f"system={sys_prompt!r}")
    print(f"user_reminder={USER_REMINDER!r}")
    print(f"role={ROLE!r}")
    print(f"prefill={PREFILL!r}  continue_final_message={CONTINUE_FINAL_MESSAGE} add_generation_prompt=False")

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
    prompts = [render_prompt(tok, m) for m in messages]
    for prompt in prompts:
        if not prompt.endswith(ASSISTANT_TAIL):
            raise RuntimeError("rendered prompt must end with a closed empty think block then '#### '")
        if prompt.rstrip().endswith("<think>"):
            raise RuntimeError("thinking block was left open")
        tail = prompt[prompt.rfind("<|im_start|>assistant\n"):]
        if "<|im_end|>" in tail:
            raise RuntimeError("continue_final_message closed the prefill")
        if "<|im_start|>assistant\n<|im_start|>assistant" in prompt:
            raise RuntimeError("a second assistant header was added after the prefill")
        if prompt.count("<think>") != 1 or "<think>\n\n</think>\n\n#### " not in prompt:
            raise RuntimeError("expected exactly one empty closed think span before the prefill")
    sps = [
        SamplingParams(
            temperature=PARAMS["temperature"],
            top_p=PARAMS["top_p"],
            max_tokens=MAX_TOKENS,
            stop=STOP,
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
            full = reconstruct_reply(c.text)
            pred_s, how_s = parse_marker(full)
            pred_f, how_f = parse_fallback(full)
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
            item["pred_strict"] = pred_s
            item["parse_how_strict"] = how_s
            item["pred_fallback"] = pred_f
            item["parse_how_fallback"] = how_f
            item["correct_strict"] = bool(
                pred_s is not None and grade(pred_s, r["ans"], r["answer_decimals"])
            )
            item["correct_fallback"] = bool(
                pred_f is not None and grade(pred_f, r["ans"], r["answer_decimals"])
            )
            item["trunc"] = c.finish_reason == "length"
            # echo uses the fallback parse so we still flag an echoed initial
            # even when the model never closed a #### N marker
            item["echo_initial"] = bool(
                pred_f is not None and grade(pred_f, r["initial"], r["answer_decimals"])
            )
            item["single_marker_line"] = is_single_marker_line(full)
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
        "sampling": {
            **PARAMS,
            "stop": STOP,
            "max_tokens": MAX_TOKENS,
        },
        "max_tokens": {"DA": MAX_TOKENS},
        "stop": STOP,
        "prefill": PREFILL,
        "enable_thinking": ENABLE_THINKING,
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
        "continue_final_message": CONTINUE_FINAL_MESSAGE,
        "add_generation_prompt": False,
        "user_reminder": USER_REMINDER,
        "role_prompt": ROLE,
        "system_prompts": {"DA": sys_prompt},
        "few_shot": {
            "path": FEW_PATH,
            "seed": few_shot.get("seed"),
            "n_examples": len(few_shot["examples"]),
            "qids": [ex["qid"] for ex in few_shot["examples"]],
            "assistants": [ex["assistant"] for ex in few_shot["examples"]],
            "taken": [ex.get("taken") for ex in few_shot["examples"]],
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
