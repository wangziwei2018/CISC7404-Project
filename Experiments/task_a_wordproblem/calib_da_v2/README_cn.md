# calib_da_v2 — 只跑 DA 的提示词修订校准

上一轮整数校准里，旧 DA 提示词（“Solve silently. Do not show calculations…”）在 d=1 只有 0.10 正确率，约 70/100 题只是把题面里的初始数量原样写回去。这次**只改 DA 提示词并只跑 DA**，题目仍用原来的 `calib_set.jsonl`，方便和上一轮直接对比。不跑 CoT，不看 Task B，没有乘法。

## 改了什么

- 系统提示改为短正向指令：
  `Answer with only the final number, in the form #### N.`
  不再使用“不要写计算过程 / 不要给中间值”这类抑制性措辞。
- 在对话里加入 **2 条 d=1 few-shot**（先 user 题，再 assistant 只回 `#### N`，不带解题步骤）。
  - 生成种子是 **20261112**，与校准集种子 1234 不同。
  - 已核对：题目、qid、数字序列、`(初始量, T, a, b, 答案)` 与 `calib_set` 中任何一题都不完全重合；两条示例之间的数字也互不相交（一条是“增加”，一条是“减少”）。
- 评测范围：原校准集中 `d ∈ {1, 2, 3, 5, 20}` 各 100 题，共 **500** 条 DA。前四档用来看 DA(1) 是否回到合理区间，d=20 用来确认 DA(20) ≤ 0.10。
- 模型仍是 `Qwen/Qwen3-4B-Instruct-2507`，revision 钉在 `cdbee75f17c01a7cc42f958dc650907174af0554`。
- 采样与上一轮相同：temperature 0.6，top_p 0.95，DA `max_tokens=64`。
- 解析和判分与 `calib_int/run_calib.py` / `analyze.py` 相同（优先取最后一个 `#### N`，否则回退到文中最后一个数；整数容差 0.5）。

## 在 Colab 上怎么跑

1. 把 `colab_calib_da_v2.ipynb` 上传到 Colab（或从 Drive 打开）。
2. 运行时选 GPU：**优先 L4**（会用 bf16）。T4 也可以，脚本会按算力自动改用 fp16。
3. **依次运行全部单元格**。第一格会安装 vLLM。运行脚本的那一格在 Colab 里会带着 Jupyter 的 `-f <kernel.json>` 参数；脚本会忽略它，仍读取本目录的 `calib_da_v2_slim.jsonl`。
4. 预估时间（含模型下载与加载）：
   - L4：大约 **5–12 分钟**；若该会话里模型已缓存，生成本身大约 1–3 分钟。
   - T4：大约 **15–25 分钟**（只跑 500 条短 DA，会比上一轮 1400 条 DA+CoT 快很多）。
5. 运行过程中会打印每个 d 的正确率、解析失败率、以及输出等于题面初始量的比例。
6. 最后一格会通过 `files.download` 自动下载：
   - `raw_outputs_da_v2.jsonl`
   - `env.json`
7. 把这两个文件放到本地目录 `/workspace/new_task_a/calib_da_v2/`（不要覆盖 `calib_int/raw_outputs.jsonl`），然后在本目录运行：

```bash
../.venv/bin/python analyze_da_v2.py raw_outputs_da_v2.jsonl
```

分析脚本会对照 `../calib_int/raw_outputs.jsonl` 里的旧 DA 和已有 CoT，输出带 Wilson 95% 置信区间的对比表，并写出 `metrics_da_v2.json` 和 `da_v2_compare.png`。

## 本目录文件

| 文件 | 说明 |
|---|---|
| `colab_calib_da_v2.ipynb` | 自包含 notebook（数据 + few-shot + 运行脚本都嵌在里面） |
| `calib_da_v2_slim.jsonl` | 500 道评测题（来自原 calib_set 的同一批题目） |
| `few_shot.json` | 系统提示和两条 few-shot |
| `run_calib_da_v2.py` | 与 notebook 中运行单元相同的脚本 |
| `analyze_da_v2.py` | 本地对比分析 |
| `dry_run.py` | 无 GPU 的提示词拼装 / 解析自检 |

## 注意

- 不要改仓库，也不要动旧的 pointer 数据集。
- 这次不跑 CoT；和 CoT 的 gap 用上一轮已经跑完的 CoT 结果来算。
- few-shot 的 assistant 只有 `#### N`，不要改成带步骤的答案，否则就变成在教模型做 CoT。
- 若新 DA 仍然写出推理过程，`max_tokens=64` 可能截断，看打印表里的 `trunc` 和 `mean_tok`。
- notebook 是自包含的，在 Colab 里不必再上传 jsonl；本地这两个数据文件主要用于核对和复现。
