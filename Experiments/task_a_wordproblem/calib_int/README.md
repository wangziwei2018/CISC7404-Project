# calib_int — 整数加减版校准（模型还没跑）

- `calib_set.jsonl` / `calib_slim.jsonl`：seed=1234，串行 d∈{1,2,3,5,10,15,20}×100 = 700 题，`cond_addsub_dec0`。与 seed 7404 的样例 / 正式生成错开。答案已由独立求解器复算。
  - `sha256(calib_set.jsonl) = f8ccff2dd3bad27a1d2f2002e51f01ee7897827942a121c7548c131b8a135442`
- `run_calib.py`：在 GPU 上跑 vLLM，模型 `Qwen/Qwen3-4B-Instruct-2507`。T4 用 half 和 AGENTS.md 3.4 的显存参数；算力 ≥ 8 用 bfloat16。采样 T=0.6、top_p=0.95。DA `max_tokens` 64，CoT 2048。输出 `raw_outputs.jsonl` 和 `env.json`。
- `colab_calib_int.ipynb`：数据脚本都嵌在笔记本里。用 Colab GPU 打开即可，不必先上传仓库。
- `analyze.py`：在 `calib_int/` 下执行 `uv run python analyze.py raw_outputs.jsonl`。它从上一层目录导入 `generate.grade`，并写出 `metrics.json`、`accuracy_vs_d.png`、`failure_samples.jsonl`。这些结果文件要等真实推理后再提交。

用法和设计说明见 `../README.md`、`../design.md`。
