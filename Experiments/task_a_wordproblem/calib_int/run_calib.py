#!/usr/bin/env python3
"""整数加减版条件应用题的校准推理（cond_wordproblem_v1）。

在有 GPU 的机器上运行（例如 Colab）。本机没有 CUDA 时不要执行。
输入：calib_slim.jsonl（qid, d, i, q, ans, answer_decimals）
      seed 1234，串行 d ∈ {1,2,3,5,10,15,20} × 100
输出：raw_outputs.jsonl（每题 × DA/CoT 一行），env.json

采样与 Experiments/task_a 一致：temperature 0.6、top_p 0.95。
top_k=20 只记录，不传入 SamplingParams。
Qwen3-4B-Instruct-2507 不是 thinking 模型，不传 enable_thinking。
DA / CoT 的 max_tokens 按本校准的输出长度设定（DA 64，CoT 2048），
不是全矩阵实验的 6144。
"""
import hashlib
import importlib.metadata as md
import json
import os
import re
import sys
import time

# 必须在 import vllm 之前设置。关闭 FlashInfer 采样器，避免没有 nvcc 时 JIT 失败。
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-4B-Instruct-2507")
REQUESTED_REVISION = os.environ.get("MODEL_REVISION") or None
IN = sys.argv[1] if len(sys.argv) > 1 else "calib_slim.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "raw_outputs.jsonl"
# d 大于该值时，CoT 只跑前 COT_N_HIGH 条（i 从 0 起）。默认 20，即 700 题全部跑 CoT。
COT_MAX_D = int(os.environ.get("COT_FULL_UP_TO", "20"))
COT_N_HIGH = int(os.environ.get("COT_N_HIGH", "100"))
PARAMS = {"temperature": 0.6, "top_p": 0.95, "top_k_recorded_not_passed": 20}
MAX_TOKENS = {"DA": 64, "CoT": 2048}
SYSTEM = {
    "DA": ("Solve silently. Do not show calculations, intermediate values, explanations or comments. "
           "Output only the final answer line.\n"
           "The final line must be #### followed by the final number, with nothing after it."),
    "CoT": ("Solve step by step. Explicitly write the amount after every event, one event per line. "
            "Then give the final answer.\n"
            "The final line must be #### followed by the final number, with nothing after it."),
}


def seed_of(qid, mode):
    return int(hashlib.sha256(f"{qid}|{mode}".encode()).hexdigest()[:8], 16)


def resolve_model_revision(model, requested):
    """在创建 LLM 之前把远端 main/tag 解析成 commit。不读 vLLM 的私有 commit 字段。"""
    from pathlib import Path

    local = Path(model)
    if local.is_dir():
        source = local.resolve()
        if source.parent.name == "snapshots" and re.fullmatch(r"[0-9a-fA-F]{40}", source.name):
            commit = source.name.lower()
            if requested is not None and requested.lower() != commit:
                raise ValueError("本地 snapshot 与 MODEL_REVISION 不一致。")
            return {"source": str(source), "model_commit": commit,
                    "requested_model_revision": requested, "revision_kind": "local_snapshot",
                    "loading_kwargs": {}}
        if requested is None:
            raise RuntimeError("普通本地权重目录没有可验证的 commit；请设置 MODEL_REVISION。")
        return {"source": str(source), "model_commit": None,
                "requested_model_revision": requested, "revision_kind": "local_label",
                "loading_kwargs": {}}

    if requested is not None and re.fullmatch(r"[0-9a-fA-F]{40}", requested):
        commit = requested.lower()
    else:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id=model, revision=requested or "main")
        commit = info.sha
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
            raise RuntimeError("Hugging Face 未返回有效 commit。可设置 MODEL_REVISION 为完整 commit。")
        commit = commit.lower()
    return {"source": model, "model_commit": commit,
            "requested_model_revision": requested, "revision_kind": "hub_commit",
            "loading_kwargs": {"revision": commit, "tokenizer_revision": commit}}


def loading_kwargs(torch, source, revision):
    """T4（算力 < 8）沿用 AGENTS.md 3.4；更好的 GPU 用 bf16，并保留本校准原来的显存长度。"""
    cap = torch.cuda.get_device_capability(0)
    visible_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if cap[0] < 8:
        if visible_gib < 14:
            raise RuntimeError("Turing / T4 上的 4B 至少需要 14 GiB 可见显存。")
        dtype = "half"
        loading = dict(model=source, dtype="half", gpu_memory_utilization=0.80,
                       max_model_len=8192, max_num_seqs=1, enforce_eager=True, seed=0)
    else:
        dtype = "bfloat16"
        loading = dict(model=source, dtype="bfloat16", gpu_memory_utilization=0.88,
                       max_model_len=4096, seed=0)
    loading.update(revision["loading_kwargs"])
    return dtype, list(cap), loading


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("本机没有 CUDA。请在 Colab GPU 上运行，不要把缺 GPU 当成代码错误。")
    revision = resolve_model_revision(MODEL, REQUESTED_REVISION)
    dtype, cap, loading = loading_kwargs(torch, revision["source"], revision)
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in open(IN)]
    llm = LLM(**loading)
    tok = llm.get_tokenizer()
    jobs = []
    for row in rows:
        for mode in ("DA", "CoT"):
            if mode == "CoT" and row["d"] > COT_MAX_D and row["i"] >= COT_N_HIGH:
                continue
            jobs.append((row, mode))
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM[mode]}, {"role": "user", "content": row["q"]}],
        tokenize=False, add_generation_prompt=True) for row, mode in jobs]
    params = [SamplingParams(temperature=PARAMS["temperature"], top_p=PARAMS["top_p"],
                             max_tokens=MAX_TOKENS[mode], seed=seed_of(row["qid"], mode))
              for row, mode in jobs]
    t0 = time.time()
    outs = llm.generate(prompts, params)
    elapsed = time.time() - t0
    with open(OUT, "w") as f:
        for (row, mode), out in zip(jobs, outs):
            choice = out.outputs[0]
            f.write(json.dumps({
                "qid": row["qid"], "d": row["d"], "i": row["i"], "mode": mode, "gold": row["ans"],
                "answer_decimals": row["answer_decimals"], "text": choice.text,
                "finish_reason": choice.finish_reason, "n_out_tokens": len(choice.token_ids),
                "n_prompt_tokens": len(out.prompt_token_ids), "seed": seed_of(row["qid"], mode),
            }) + "\n")
    env = {
        "model": MODEL,
        "model_commit": revision["model_commit"],
        "requested_model_revision": revision["requested_model_revision"],
        "revision_kind": revision["revision_kind"],
        "dtype": dtype,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": cap,
        "loading": {k: loading[k] for k in loading if k != "model"},
        "model_source": loading["model"],
        "vllm": md.version("vllm"),
        "torch": torch.__version__,
        "transformers": md.version("transformers"),
        "sampling": PARAMS,
        "max_tokens": MAX_TOKENS,
        "system_prompts": SYSTEM,
        "chat_template_kwargs": {},
        "n_jobs": len(jobs),
        "generate_seconds": round(elapsed, 1),
        "cot_full_up_to_d": COT_MAX_D,
        "cot_n_high": COT_N_HIGH,
        "vllm_env": {k: os.environ.get(k) for k in (
            "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_LOGGING_LEVEL")},
    }
    json.dump(env, open("env.json", "w"), indent=2)
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
