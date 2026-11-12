# Task A 替换候选：条件分支应用题

现有正式题是 40 行指针查表（`Experiments/data_generation/synthesis_pointer_numeric_dataset.jsonl`）。
那一版的直接回答（DA）从深度 d=5 起就落到 0，深度梯度主要是地板，不是一条可测的曲线。
这里是替换候选 `cond_wordproblem_v1`：GSM8K 风格的小学应用题，**只用加减**，小数可选。
第一轮整数校准已经跑完，DA 提示还要重测，见下面的 `calib_da_v2/`。不能把它当成已经定题的正式集。指针题和 Task B 的文件都没有改。

外部原型称这一候选为 Task A v4。仓库里 `AGENTS.md` 5.5 的 Task A v3 指的是已经交付的指针题。
两套题的 `version` 不同，不要混在同一张准确率表里。

## 机制

每一步比较当前数量和阈值 T：大于 T 就减去 a，小于 T 就加上 b。当前值永远不等于 T。
下一步用哪一个数、做加法还是减法，取决于上一步的结果，所以不能把题里所有数字一次加总。
串行 d=1..20，`W = d`。并行 w=1..10，链长 `[1, 2, ..., 2]`，`W = 2w−1`，用来和串行奇数深度对照。

只把字段 `q` 发给模型。`ans`、`answers`、`solution`、`chains` 里都有标准答案。

## 生成

在仓库根目录：

```bash
cd Experiments/task_a_wordproblem
uv run python generate.py --mode samples --show          # 整数样例 -> samples.jsonl
uv run python generate.py --mode samples --decimals 2 --out samples_dec2.jsonl --show
uv run python generate.py --mode check --n 500           # 每个 d 500 题的捷径统计，不写题库
uv run python generate.py --mode calib --seed 1234 --out calib_int/calib_set.jsonl
uv run python generate.py --mode full --out full.jsonl   # 串行 2000 + 并行 1000；先完成校准和人工审核
```

- 样例 seed 默认 7404。整数校准集 seed 是 1234，和正式 seed 错开。
- `--decimals 0|1|2`。0 是整数，1 和 2 是钱（内部按分计算，没有浮点误差）。
- `--ops` 只有 `addsub`。乘法 / `mixed` 已从生成路径删除；`config` 里留下的 `p_mul2` 不再被读取，只为和已生成样例的字段对齐。
- 同一 seed 会再生成一遍并比对 `qid`，用来确认随机序列没有漂。

设计细节和捷径比例见 `design.md`。`check_int.txt`、`check_dec2.txt` 是 `--mode check --n 500` 的输出。

## 校准（需要 GPU，本仓库的 CPU 环境不跑模型）

整数校准集在 `calib_int/`：

| 文件 | 内容 |
|---|---|
| `calib_set.jsonl` | 700 题全文，d ∈ {1,2,3,5,10,15,20} × 100，seed 1234 |
| `calib_slim.jsonl` | 推理用的瘦字段：`qid, d, i, q, ans, answer_decimals` |
| `run_calib.py` | vLLM 推理。模型 `Qwen/Qwen3-4B-Instruct-2507` |
| `colab_calib_int.ipynb` | 自包含 Colab 笔记本，内嵌瘦数据和 `run_calib.py` |
| `analyze.py` | 读 `raw_outputs.jsonl`，对照 task.md A.0 的门槛写 `metrics.json` |

Colab：打开 `colab_calib_int.ipynb`，选 GPU，依次运行。下载 `raw_outputs.jsonl` 和 `env.json` 放到 `calib_int/`。

命令行（在 GPU 机器上，工作目录为 `calib_int/`）：

```bash
python run_calib.py calib_slim.jsonl raw_outputs.jsonl
python analyze.py raw_outputs.jsonl
```

加载约定：

- `import vllm` 之前设置 `VLLM_USE_FLASHINFER_SAMPLER=0`、`VLLM_ENABLE_V1_MULTIPROCESSING=0`。
- 创建引擎前用 Hugging Face `model_info().sha` 解析 commit，并传给 `revision` 和 `tokenizer_revision`。`MODEL_REVISION` 可留空。
- 不传 `enable_thinking`。采样是 temperature 0.6、top_p 0.95；top_k=20 只记录。
- T4（算力 < 8）使用 `dtype="half"`、`gpu_memory_utilization=0.80`、`max_model_len=8192`、`max_num_seqs=1`、`enforce_eager=True`，并要求至少 14 GiB 可见显存。
- 更好的 GPU 使用 bfloat16、`gpu_memory_utilization=0.88`、`max_model_len=4096`。
- DA 的 `max_tokens` 是 64，CoT 是 2048。这是这道短应用题的校准预算，不是全矩阵的 6144。

`analyze.py` 会写出 `metrics.json`、`accuracy_vs_d.png`、`failure_samples.jsonl`。判分和 `generate.grade` 相同：去掉 `$` 和逗号，容差是最小单位的一半。第一轮的原文 `raw_outputs.jsonl` 还没放进仓库。

## DA 提示词修订（`calib_da_v2/`，结果未出）

第一轮 DA 用的是 “Solve silently...”。d=1,2,3,5,10,15,20 的准确率是 DA 0.10/0.03/0.01/0.00/0.00/0.01/0.01，CoT 0.90/0.95/0.96/0.91/0.77/0.77/0.75。DA(1)=0.10，多数输出只是把初始数量抄回去。

`calib_da_v2/` 只重跑 DA：同一批题里的 d∈{1,2,3,5,20}×100，系统提示改为 `Answer with only the final number, in the form #### N.`，并加 2 条只含 `#### N` 的 d=1 few-shot（seed 20261112）。L4 上回退准确率是 0.66/0.01/0.00/0.00/0.00；d=1 的 0.66 含截断 CoT 的回退解析，严格 `####` 正确是 36/100。细节在 `record.md`。

## DA v3（`calib_da_v3/`，结果未出）

`calib_da_v3/` 仍是这 500 题，只跑 DA。整段回复被收成一行：系统提示要求恰好一行 `#### N`，用户消息再提醒一次，assistant 预填 `#### `，`stop=['\n']`，`max_tokens=16`，4 条只含答案的 d=1 few-shot（seed 20261113）。主指标是严格正确率。用法在该目录 `README_cn.md`。结果还没有。

## 还没做

探针、斜率、人工审核和正式 3000 题都还没做。`full.jsonl` 不要在校准完成前当作正式数据提交。
