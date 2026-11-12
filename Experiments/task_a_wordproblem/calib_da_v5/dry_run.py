#!/usr/bin/env python3
"""GPU-free dry run: prompt construction, official Qwen3 chat template rendering,
prefill, parser/grader on synthetic outputs, overlap checks, and notebook validation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import traceback
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_calib_da_v5 import (  # noqa: E402
    MODEL, REVISION, SYSTEM, ROLE, USER_REMINDER, PREFILL, MAX_TOKENS, STOP, PARAMS,
    PROMPT_VERSION, CONTINUE_FINAL_MESSAGE, ENABLE_THINKING, ASSISTANT_TAIL,
    build_messages, with_user_reminder, reconstruct_reply,
    parse, parse_marker, parse_fallback, grade, seed_of, is_single_marker_line,
    render_prompt,
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


def render_with_prefill(messages):
    prompt = apply_qwen_chat_template(messages, add_generation_prompt=True)
    return prompt + PREFILL


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    errors = []
    few = json.load(open(os.path.join(HERE, "few_shot.json"), encoding="utf-8"))
    slim = load_jsonl(os.path.join(HERE, "calib_da_v5_slim.jsonl"))
    calib = load_jsonl(os.path.join(HERE, "../calib_int/calib_set.jsonl"))
    v2_slim_path = os.path.join(HERE, "../calib_da_v2/calib_da_v2_slim.jsonl")
    v3_slim_path = os.path.join(HERE, "../calib_da_v3/calib_da_v3_slim.jsonl")
    v2_slim = load_jsonl(v2_slim_path) if os.path.exists(v2_slim_path) else None
    v3_slim = load_jsonl(v3_slim_path) if os.path.exists(v3_slim_path) else None

    # --- basic invariants ---
    if few["system"] != SYSTEM:
        errors.append(f"system mismatch: {few['system']!r}")
    if not few["system"].startswith(ROLE):
        errors.append("role sentence is not at the front of the system prompt")
    if "Your entire reply must be exactly one line in the form #### N." not in few["system"]:
        errors.append("format reminder was dropped from the system prompt")
    if few.get("user_reminder") != USER_REMINDER:
        errors.append(f"user_reminder mismatch: {few.get('user_reminder')!r}")
    if few.get("prefill") != PREFILL:
        errors.append(f"prefill mismatch: {few.get('prefill')!r}")
    if few["seed"] in (1234, 20261112):
        errors.append(f"few-shot seed must not be 1234 or 20261112, got {few['seed']}")
    if few["seed"] != 20261113:
        errors.append(f"expected reused v3 few-shot seed 20261113, got {few['seed']}")
    if PROMPT_VERSION != "da_v5":
        errors.append(f"prompt version {PROMPT_VERSION}")
    if CONTINUE_FINAL_MESSAGE is not True:
        errors.append("continue_final_message must stay True")
    if len(few["examples"]) != 4:
        errors.append(f"expected 4 few-shot examples, got {len(few['examples'])}")
    taken = [ex["taken"] for ex in few["examples"]]
    if taken.count("gain") != 2 or taken.count("loss") != 2:
        errors.append(f"expected 2 add / 2 subtract, got {taken}")
    if len(slim) != 500:
        errors.append(f"expected 500 slim items, got {len(slim)}")
    dc = Counter(r["d"] for r in slim)
    if dict(dc) != {1: 100, 2: 100, 3: 100, 5: 100, 20: 100}:
        errors.append(f"unexpected depth counts: {dc}")

    slim_sha = hashlib.sha256(open(os.path.join(HERE, "calib_da_v5_slim.jsonl"), "rb").read()).hexdigest()
    expect_sha = "c3820a69e072fa6d4a6211c704dd64640af7674a85b32b2e0d06ac6b0aa57c51"
    if slim_sha != expect_sha:
        errors.append(f"slim SHA256 changed: {slim_sha} != {expect_sha}")
    if v2_slim is not None:
        if [r["qid"] for r in slim] != [r["qid"] for r in v2_slim]:
            errors.append("slim qids do not match v2 slim")
    if v3_slim is not None:
        if [r["qid"] for r in slim] != [r["qid"] for r in v3_slim]:
            errors.append("slim qids do not match v3 slim")
    v3_few_path = os.path.join(HERE, "../calib_da_v3/few_shot.json")
    if os.path.exists(v3_few_path):
        v3_few = json.load(open(v3_few_path, encoding="utf-8"))
        if [ex["qid"] for ex in few["examples"]] != [ex["qid"] for ex in v3_few["examples"]]:
            errors.append("few-shot qids were not reused from v3")
        if [ex["q"] for ex in few["examples"]] != [ex["q"] for ex in v3_few["examples"]]:
            errors.append("few-shot questions were not reused from v3")

    # stop / max_tokens
    if STOP != ["\n"]:
        errors.append(f"stop must be ['\\n'], got {STOP}")
    if MAX_TOKENS != 16:
        errors.append(f"max_tokens must be 16, got {MAX_TOKENS}")
    if PARAMS["temperature"] != 0.6 or PARAMS["top_p"] != 0.95:
        errors.append(f"sampling changed: {PARAMS}")

    calib_by = {(r["d"], r["i"]): r for r in calib}
    for r in slim:
        c = calib_by[(r["d"], r["i"])]
        if (r["qid"], r["q"], r["ans"]) != (c["qid"], c["q"], c["ans"]):
            errors.append(f"slim item does not match calib_set: d={r['d']} i={r['i']}")
            break

    # overlap vs calib_set / slim
    qid_set = {r["qid"] for r in calib} | {r["qid"] for r in slim}
    q_set = {r["q"] for r in calib} | {r["q"] for r in slim}
    num_pats = {tuple(re.findall(r"\d+(?:\.\d+)?", r["q"])) for r in calib}
    param_pats = set()
    for r in calib:
        ch = r["chains"][0]
        for ev in ch["events"]:
            param_pats.add((ch["initial"], ev["T"], ev["a"], ev["b"], r["ans"]))
    slim_golds = {r["ans"] for r in slim}
    nsets = []
    names, items = [], []
    gold_collisions = []
    for ex in few["examples"]:
        if ex["assistant"] != f"#### {ex['ans']}":
            errors.append(f"assistant is not exactly '#### N': {ex['assistant']!r}")
        if re.search(r"=<<|step|because|so ", ex["assistant"], re.I):
            errors.append(f"assistant looks like it contains reasoning: {ex['assistant']!r}")
        if ex["qid"] in qid_set or ex["q"] in q_set:
            errors.append(f"few-shot overlaps calib_set text/qid: {ex['qid']}")
        if tuple(re.findall(r"\d+(?:\.\d+)?", ex["q"])) in num_pats:
            errors.append(f"few-shot number sequence overlaps calib_set: i={ex['i']}")
        tup = (ex["initial"], ex["T"], ex["a"], ex["b"], ex["ans"])
        if tup in param_pats:
            errors.append(f"few-shot (initial,T,a,b,ans) overlaps calib_set: i={ex['i']}")
        # distractor: at least 3 sentences in the body
        body = ex["q"].split("\n")[0]
        sents = re.split(r"(?<=\.)\s+", body.strip())
        if len(sents) < 3:
            errors.append(f"few-shot missing distractor sentence: i={ex['i']}")
        if ex["ans"] in slim_golds:
            gold_collisions.append(ex["ans"])
        nsets.append(set(re.findall(r"\d+", ex["q"])) | {ex["ans"]})
        names.append(ex["name"])
        items.append(ex["item"])
    if gold_collisions:
        errors.append(f"few-shot answers collide with 500-item golds: {gold_collisions}")
    shared = set()
    for i in range(len(nsets)):
        for j in range(i + 1, len(nsets)):
            shared |= (nsets[i] & nsets[j])
    if shared:
        errors.append(f"few-shot examples share numbers: {shared}")
    if len(set(names)) < 4:
        errors.append(f"few-shot names not all distinct: {names}")
    if len(set(items)) < 4:
        errors.append(f"few-shot items not all distinct: {items}")

    # --- render chat templates ---
    print("SYSTEM:", few["system"])
    print("USER_REMINDER:", repr(USER_REMINDER))
    print("PREFILL:", repr(PREFILL))
    print("STOP:", STOP, "MAX_TOKENS:", MAX_TOKENS)
    print("\n===== few-shot examples =====")
    for i, ex in enumerate(few["examples"], 1):
        print(f"\n--- example {i} (d={ex['d']}, taken={ex['taken']}, {ex['name']}/{ex['item']}) ---")
        print(ex["q"])
        print("assistant:", ex["assistant"])

    expect_roles = ["system"] + ["user", "assistant"] * 4 + ["user", "assistant"]
    from transformers import AutoTokenizer
    if MODEL != "Qwen/Qwen3-4B" or "Base" in MODEL or MODEL.endswith("Instruct-2507"):
        errors.append(f"model id must be Qwen/Qwen3-4B, got {MODEL}")
    if ENABLE_THINKING:
        errors.append("enable_thinking must be False")
    if REVISION != "1cfa9a7208912126459214e8b04321603b3df60c":
        errors.append(f"unexpected revision {REVISION}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    if getattr(tokenizer, "name_or_path", "") not in (MODEL, MODEL.split("/")[-1]) and "Qwen3-4B" not in str(getattr(tokenizer, "name_or_path", "")):
        errors.append(f"tokenizer name_or_path {tokenizer.name_or_path}")
    if "Base" in str(getattr(tokenizer, "name_or_path", "")):
        errors.append("loaded a Base tokenizer")
    samples = [slim[0], slim[100]]  # first d=1 and first d=2
    for r in samples:
        msgs = build_messages(r["q"], few)
        roles = [m["role"] for m in msgs]
        if roles != expect_roles:
            errors.append(f"role sequence {roles} != {expect_roles}")
        if msgs[0]["content"] != SYSTEM:
            errors.append("system message mismatch")
        last_user = [m for m in msgs if m["role"] == "user"][-1]
        if not last_user["content"].endswith(USER_REMINDER):
            errors.append("last user message missing reminder")
        if not last_user["content"].startswith(r["q"]):
            errors.append("last user message does not start with the test question")
        for m in msgs:
            if m["role"] == "user" and not m["content"].endswith(USER_REMINDER):
                errors.append("a user message is missing the one-line reminder")
        completed = [m for m in msgs if m["role"] == "assistant"][:-1]
        for m in completed:
            if not re.fullmatch(r"####\s*-?\d+", m["content"].strip()):
                errors.append(f"few-shot assistant is not a bare #### N: {m['content']!r}")
        if msgs[-1] != {"role": "assistant", "content": PREFILL}:
            errors.append(f"final message is not the prefill: {msgs[-1]!r}")

        rendered = render_prompt(tokenizer, msgs)
        print("\n===== apply_chat_template continue_final_message (d={}, i={}, qid={}…) =====".format(
            r["d"], r["i"], r["qid"][:12]
        ))
        print(rendered)
        print("----- end render, chars={} -----".format(len(rendered)))
        for token in ("<|im_start|>system", "<|im_start|>user", "<|im_start|>assistant"):
            if token not in rendered:
                errors.append(f"missing {token} in rendered prompt")
        if not rendered.endswith(ASSISTANT_TAIL):
            errors.append(
                f"prompt must end with closed empty think then prefill; ends with {rendered[-80:]!r}"
            )
        if rendered.rstrip().endswith("<think>"):
            errors.append("thinking block left open at the end")
        if rendered.count("<think>") != 1:
            errors.append(f"expected one empty think span, got {rendered.count('<think>')}")
        if "<think>\n\n</think>\n\n#### " not in rendered:
            errors.append("empty closed think span is not immediately before the prefill")
        tail = rendered[rendered.rfind("<|im_start|>assistant\n"):]
        if "<|im_end|>" in tail:
            errors.append("continue_final_message closed the prefill with <|im_end|>")
        if "<|im_start|>assistant\n<|im_start|>assistant" in rendered:
            errors.append("a second assistant header was added")
        if rendered.count("<|im_start|>assistant\n") != 5:
            errors.append(
                f"expected 4 closed few-shots plus one continued prefill, got "
                f"{rendered.count('<|im_start|>assistant\\n')} assistant headers"
            )
        for ex in few["examples"]:
            closed = ex["assistant"] + "<|im_end|>"
            if closed not in rendered:
                errors.append(f"completed few-shot was not closed: {ex['assistant']!r}")
        if USER_REMINDER.strip() not in rendered:
            errors.append("user reminder not present in rendered prompt")
        # few-shot assistant answers must appear, gold solution traces must not
        for ex in few["examples"]:
            if ex["assistant"] not in rendered:
                errors.append("few-shot assistant missing from prompt")
            if "solution_not_shown_to_model" in ex and "<<" in ex["solution_not_shown_to_model"]:
                leak = ex["solution_not_shown_to_model"].split("\n")[0]
                if leak in rendered:
                    errors.append("gold solution leaked into the prompt")
        assistant_bodies = [m["content"] for m in msgs if m["role"] == "assistant"]
        if r["ans"] in assistant_bodies or f"#### {r['ans']}" in assistant_bodies:
            errors.append(f"test gold {r['ans']} leaked in few-shot assistants")

        s = seed_of(r["qid"], "DA")
        if not isinstance(s, int) or s < 0:
            errors.append("bad seed")

    # --- parser / grader on synthetic outputs ---
    print("\n===== parser / grader cases =====")
    # (text, prefill_applied, gold, dec, should_strict, should_fallback, how_fallback, single_line)
    cases = [
        # full marker, no prefill needed
        ("#### 77", False, "77", 0, True, True, "marker", True),
        ("#### 77\n", False, "77", 0, True, True, "marker", True),
        # prefill already consumed: bare number
        ("77", True, "77", 0, True, True, "marker", True),
        ("77\n", True, "77", 0, True, True, "marker", True),
        (" 77", True, "77", 0, True, True, "marker", True),
        # model ignored prefill and wrote its own marker
        ("#### 77", True, "77", 0, True, True, "marker", True),
        # wrong number, still a clean DA line
        ("12", True, "77", 0, False, False, "marker", True),
        # leaked paragraph after prefill: no valid marker at the start
        (
            "First I compare 101 with 66, then 101-24=77, so the answer is 77.",
            True, "77", 0, False, True, "fallback", False,
        ),
        (
            "The child starts with 101 shells and ends with 77",
            True, "77", 0, False, True, "fallback", False,
        ),
        # empty generation after prefill -> '#### ' only, no number
        ("", True, "77", 0, False, False, "none", False),
        ("no number here", True, "77", 0, False, False, "none", False),
        ("#### 77.0", False, "77", 0, True, True, "marker", True),
        ("#### $88", False, "88", 0, True, True, "marker", True),
    ]
    for text, pref, gold, dec, should_s, should_f, how, single in cases:
        full = reconstruct_reply(text) if pref else text
        pred_s, how_s = parse_marker(full)
        pred_f, how_f = parse_fallback(full)
        # parse(..., prefill_applied=) should agree with fallback on the reconstructed text
        pred_p, how_p = parse(text, prefill_applied=pref)
        ok_s = pred_s is not None and grade(pred_s, gold, dec)
        ok_f = pred_f is not None and grade(pred_f, gold, dec)
        sl = is_single_marker_line(full)
        status = "OK"
        if ok_s != should_s or ok_f != should_f or how_f != how or sl != single:
            status = "MISMATCH"
            errors.append(
                f"parse case failed: text={text!r} pref={pref} -> "
                f"strict={pred_s}/{ok_s} fallback={pred_f}/{how_f}/{ok_f} single={sl}"
            )
        if (pred_p, how_p) != (pred_f, how_f):
            status = "MISMATCH"
            errors.append(f"parse() disagrees with fallback: {text!r} -> {pred_p}/{how_p} vs {pred_f}/{how_f}")
        print(
            f"  {status:9} pref={pref!s:5} text={text!r:70} "
            f"strict={pred_s!s:>6}/{ok_s!s:5} fallback={pred_f!s:>6}/{how_f:8}/{ok_f!s:5} "
            f"single={sl} (expect s={should_s} f={should_f}/{how} sl={single})"
        )

    # echo-initial helper
    r0 = slim[0]
    pred, how = parse(str(r0["initial"]), prefill_applied=True)
    if not (pred and grade(pred, r0["initial"], r0["answer_decimals"])):
        errors.append("echo-initial detection failed on synthetic prefilled initial")
    if grade(pred, r0["ans"], r0["answer_decimals"]) and r0["initial"] != r0["ans"]:
        errors.append("initial graded as gold unexpectedly")

    # --- notebook validation ---
    nb_path = os.path.join(HERE, "colab_calib_da_v5.ipynb")
    try:
        src = None
        try:
            import nbformat
            nb = nbformat.read(nb_path, as_version=4)
            nbformat.validate(nb)
            src = "\n".join(
                ("".join(c.source) if isinstance(c.source, list) else c.source)
                for c in nb.cells
            )
            print(f"\nnotebook ok (nbformat): {nb_path}  cells={len(nb.cells)}  "
                  f"nbformat={nb.nbformat}.{nb.nbformat_minor}")
        except ImportError:
            nb = json.load(open(nb_path, encoding="utf-8"))
            cells = nb["cells"]
            src = "\n".join(
                ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
                for c in cells
            )
            for i, c in enumerate(cells):
                if c.get("cell_type") == "code":
                    body = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                    compile(body, f"notebook_cell_{i}", "exec")
            print(f"\nnotebook ok (json+compile): {nb_path}  cells={len(cells)}")
        needed = [
            "raw_outputs_da_v5.jsonl",
            "files.download",
            SYSTEM,
            "Qwen/Qwen3-4B",
            "1cfa9a7208912126459214e8b04321603b3df60c",
            "enable_thinking=False",
            "max_tokens",
            "stop",
            "#### ",
            "Reply with only one line: #### N.",
            "You are a math problem solver. Solve the following math problem.",
            "continue_final_message",
            "add_generation_prompt=False",
            "echo_initial",
            "correct_strict",
            "a.endswith(\".jsonl\")",
        ]
        for s in needed:
            if s not in src:
                errors.append(f"notebook missing expected string: {s}")
        # the argv bug must not come back
        if re.search(r"sys\.argv\[1\]", src) and "endswith(\".jsonl\")" not in src:
            errors.append("notebook appears to read raw sys.argv[1] without a .jsonl filter")
        if 'os.environ.get("MODEL", "Qwen/Qwen3-4B-Base")' in src or 'os.environ.get("MODEL", "Qwen/Qwen3-4B-Instruct-2507")' in src:
            errors.append("notebook default model is Base or Instruct-2507")
        if 'from_pretrained("Qwen/Qwen3-4B-Base"' in src or 'from_pretrained("Qwen/Qwen3-4B-Instruct-2507"' in src:
            errors.append("notebook loads Base or Instruct-2507")
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
