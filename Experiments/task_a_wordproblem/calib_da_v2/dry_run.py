#!/usr/bin/env python3
"""GPU-free dry run: prompt construction, official Qwen3 chat template rendering,
parser/grader on synthetic outputs, overlap checks, and notebook validation.
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_calib_da_v2 import (  # noqa: E402
    SYSTEM, build_messages, parse, grade, seed_of,
)

try:
    from jinja2 import Environment, BaseLoader
except ImportError:
    Environment = None


def apply_qwen_chat_template(messages, add_generation_prompt=True, template=None):
    """Render with the official Qwen3-4B-Instruct-2507 chat template (no tools)."""
    if template is None:
        path = os.path.join(HERE, "qwen3_chat_template.jinja")
        template = open(path, encoding="utf-8").read()
    if Environment is None:
        # Fallback identical to the no-tools branch of the official template.
        parts = []
        if messages and messages[0]["role"] == "system":
            parts.append("<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n")
        for i, message in enumerate(messages):
            role, content = message["role"], message.get("content") or ""
            if role == "user" or (role == "system" and i != 0):
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            elif role == "assistant":
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)
    env = Environment(loader=BaseLoader())
    env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
    tmpl = env.from_string(template)
    return tmpl.render(messages=messages, add_generation_prompt=add_generation_prompt, tools=None)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    errors = []
    few = json.load(open(os.path.join(HERE, "few_shot.json"), encoding="utf-8"))
    slim = load_jsonl(os.path.join(HERE, "calib_da_v2_slim.jsonl"))
    calib = load_jsonl(os.path.join(HERE, "../calib_int/calib_set.jsonl"))

    # --- basic invariants ---
    if few["system"] != SYSTEM:
        errors.append(f"system mismatch: {few['system']!r}")
    if few["system"] != "Answer with only the final number, in the form #### N.":
        errors.append("system prompt wording changed")
    if few["seed"] == 1234:
        errors.append("few-shot seed must not be 1234")
    if len(few["examples"]) != 2:
        errors.append(f"expected 2 few-shot examples, got {len(few['examples'])}")
    if len(slim) != 500:
        errors.append(f"expected 500 slim items, got {len(slim)}")
    from collections import Counter
    dc = Counter(r["d"] for r in slim)
    if dict(dc) != {1: 100, 2: 100, 3: 100, 5: 100, 20: 100}:
        errors.append(f"unexpected depth counts: {dc}")

    calib_by = {(r["d"], r["i"]): r for r in calib}
    for r in slim:
        c = calib_by[(r["d"], r["i"])]
        if (r["qid"], r["q"], r["ans"]) != (c["qid"], c["q"], c["ans"]):
            errors.append(f"slim item does not match calib_set: d={r['d']} i={r['i']}")
            break

    # overlap vs calib_set
    qid_set = {r["qid"] for r in calib}
    q_set = {r["q"] for r in calib}
    num_pats = {tuple(re.findall(r"\d+(?:\.\d+)?", r["q"])) for r in calib}
    param_pats = set()
    for r in calib:
        ch = r["chains"][0]
        for ev in ch["events"]:
            param_pats.add((ch["initial"], ev["T"], ev["a"], ev["b"], r["ans"]))
    nsets = []
    for ex in few["examples"]:
        if ex["assistant"] != f"#### {ex['ans']}":
            errors.append(f"assistant is not exactly '#### N': {ex['assistant']!r}")
        if re.search(r"=<<|step|because|so ", ex["assistant"], re.I):
            errors.append(f"assistant looks like it contains reasoning: {ex['assistant']!r}")
        if ex["qid"] in qid_set or ex["q"] in q_set:
            errors.append(f"few-shot overlaps calib_set text/qid: {ex['qid']}")
        if tuple(re.findall(r"\d+(?:\.\d+)?", ex["q"])) in num_pats:
            errors.append(f"few-shot number sequence overlaps calib_set: i={ex['i']}")
        nsets.append(set(re.findall(r"\d+", ex["q"])) | {ex["ans"]})
    if nsets and nsets[0] & nsets[1]:
        errors.append(f"few-shot pair shares numbers: {nsets[0] & nsets[1]}")

    # --- render chat templates ---
    print("SYSTEM:", few["system"])
    print("\n===== few-shot examples =====")
    for i, ex in enumerate(few["examples"], 1):
        print(f"\n--- example {i} (d={ex['d']}, taken={ex['taken']}, {ex['name']}/{ex['item']}) ---")
        print(ex["q"])
        print("assistant:", ex["assistant"])

    samples = [slim[0], slim[100]]  # first d=1 and first d=2
    for r in samples:
        msgs = build_messages(r["q"], few)
        roles = [m["role"] for m in msgs]
        expect = ["system", "user", "assistant", "user", "assistant", "user"]
        if roles != expect:
            errors.append(f"role sequence {roles} != {expect}")
        if msgs[0]["content"] != SYSTEM:
            errors.append("system message mismatch")
        if msgs[-1]["content"] != r["q"]:
            errors.append("last user message is not the test question")
        rendered = apply_qwen_chat_template(msgs, add_generation_prompt=True)
        print("\n===== rendered chat template (d={}, i={}, qid={}…) =====".format(
            r["d"], r["i"], r["qid"][:12]
        ))
        print(rendered)
        print("----- end render, chars={} -----".format(len(rendered)))
        for token in ("<|im_start|>system", "<|im_start|>user", "<|im_start|>assistant"):
            if token not in rendered:
                errors.append(f"missing {token} in rendered prompt")
        if not rendered.endswith("<|im_start|>assistant\n"):
            errors.append("generation prompt not at end")
        # few-shot assistant answers must appear, gold solution traces must not
        for ex in few["examples"]:
            if ex["assistant"] not in rendered:
                errors.append("few-shot assistant missing from prompt")
            # the hidden gold solution (with << >>) must not leak
            if "solution_not_shown_to_model" in ex and "<<" in ex["solution_not_shown_to_model"]:
                leak = ex["solution_not_shown_to_model"].split("\n")[0]
                if leak in rendered:
                    errors.append("gold solution leaked into the prompt")
        # test-item gold must not be in the few-shot portion; it may coincidentally
        # appear as a distractor number, so we only check the assistant turns.
        assistant_bodies = [m["content"] for m in msgs if m["role"] == "assistant"]
        if r["ans"] in assistant_bodies or f"#### {r['ans']}" in assistant_bodies:
            errors.append(f"test gold {r['ans']} leaked in few-shot assistants")

        # seed function still stable
        s = seed_of(r["qid"], "DA")
        if not isinstance(s, int) or s < 0:
            errors.append("bad seed")

    # --- parser / grader on synthetic outputs ---
    print("\n===== parser / grader cases =====")
    cases = [
        ("#### 127", "127", 0, True, "marker"),
        ("#### 127\n", "127", 0, True, "marker"),
        ("####127", "127", 0, True, "marker"),
        ("blah\n#### 127", "127", 0, True, "marker"),
        ("#### ANSWER = 127", "127", 0, True, "marker"),
        ("#### 99", "127", 0, False, "marker"),
        ("127", "127", 0, True, "fallback"),
        ("I think the answer is 127", "127", 0, True, "fallback"),
        ("#### 108", "127", 0, False, "marker"),  # echoed initial of Mia example, wrong
        ("", "127", 0, False, "none"),
        ("no number here", "127", 0, False, "none"),
        ("#### 77.0", "77", 0, True, "marker"),  # int gold, decimal pred within 0.5
        ("#### $88", "88", 0, True, "marker"),
    ]
    for text, gold, dec, should, how in cases:
        pred, got_how = parse(text)
        ok = pred is not None and grade(pred, gold, dec)
        echo_init = pred is not None and grade(pred, "108", 0)
        status = "OK" if (ok == should and got_how == how) else "MISMATCH"
        if status != "OK":
            errors.append(f"parse case failed: {text!r} -> pred={pred} how={got_how} correct={ok}")
        print(f"  {status:9} text={text!r:35} pred={pred!s:6} how={got_how:8} correct={ok} (expect {should}/{how})")

    # echo-initial helper: d=1 first slim item
    r0 = slim[0]
    pred, how = parse(f"#### {r0['initial']}")
    if not (pred and grade(pred, r0["initial"], r0["answer_decimals"])):
        errors.append("echo-initial detection failed on synthetic #### initial")
    if grade(pred, r0["ans"], r0["answer_decimals"]) and r0["initial"] != r0["ans"]:
        errors.append("initial graded as gold unexpectedly")

    # --- notebook validation ---
    nb_path = os.path.join(HERE, "colab_calib_da_v2.ipynb")
    try:
        import nbformat
        nb = nbformat.read(nb_path, as_version=4)
        nbformat.validate(nb)
        src = "\n".join(
            ("".join(c.source) if isinstance(c.source, list) else c.source)
            for c in nb.cells
        )
        needed = [
            "raw_outputs_da_v2.jsonl",
            "files.download",
            SYSTEM,
            "Qwen/Qwen3-4B-Instruct-2507",
            "cdbee75f17c01a7cc42f958dc650907174af0554",
            "max_tokens",
            "echo_initial",
            "parse_fail",
        ]
        for s in needed:
            if s not in src:
                errors.append(f"notebook missing expected string: {s}")
        print(f"\nnotebook ok: {nb_path}  cells={len(nb.cells)}  nbformat={nb.nbformat}.{nb.nbformat_minor}")
    except Exception:
        errors.append("notebook validation failed:\n" + traceback.format_exc())

    print("\n===== result =====")
    if errors:
        print(f"FAIL ({len(errors)} issues)")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
