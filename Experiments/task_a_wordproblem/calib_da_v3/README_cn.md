# calib_da_v3 — 只跑 DA 的提示词与解码修订校准

上一轮 DA v2 把系统提示改成了短正向指令，并加了 2 条只含 `#### N` 的 d=1 few-shot。按「文中最后一个数」回退解析，d=1 正确率是 0.66，但这个数字被夸大了：100 题里有 54 题顶满了 `max_tokens=64`，模型在写思维链；严格按 `#### N` 只有 36 题正确，另外 29 题的「正确」来自截断 CoT 的回退解析。d≥2 几乎全是被截断的推理过程。

Task A 要求的 DA(1) ∈ [0.80, 0.95] 必须是**真正的直接作答**，不能靠泄漏出来的 CoT。本次 **v3 只改 DA 的提示词和解码，仍然只跑 DA**。题目与 v2 完全相同（同一批 500 题，文件 SHA256 不变），方便直接对比。不跑 CoT，不看 Task B，没有乘法。

**表头指标是严格正确率（strict）**：只认 `#### N` 标记，不用文中最后一个数做主结论。回退解析只作为和 v2 对照的参考列。

## 相对 v2 改了什么

1. 系统提示改为：
   `Your entire reply must be exactly one line in the form #### N.`
2. 每一条 user 消息（包括 few-shot 里的题，以及每一道测试题）在题目后面追加一行提醒：
   `Reply with only one line: #### N.`
3. **预填 assistant 前缀 `#### `（含空格）**。实现方式：先用官方 Qwen3 chat template 渲染，并设 `add_generation_prompt=True`（prompt 以 `<|im_start|>assistant\n` 结尾），然后再追加 `#### `。模型从空格后面开始生成。few-shot 的 assistant 仍然是完整的 `#### N`，前后一致。评分时如果生成文本不以 `####` 开头，会先补上这段预填再解析。因此模型只输出 `77` 时，严格解析看到的是 `#### 77`。
4. 采样温度和 top_p 不变（0.6 / 0.95），但加上 `stop=['\n']`，并把 `max_tokens` 从 64 降到 **16**，让泄漏的推理链写不完，也不能被当成正确答案。
5. few-shot 增加到 **4 条 d=1**（2 道增加、2 道减少），每条都带一句干扰句。生成种子是 **20261113**（与校准集种子 1234、v2 种子 20261112 都不同）。已核对：题目、qid、数字序列、`(初始量, T, a, b, 答案)` 与校准集 / 评测集都不完全重合；四条示例的数字互不相交；四个示例答案都不是这 500 道测试题的金标。
6. 评分同时报告：(a) **严格**：最后一个 / 唯一的 `#### N`，无回退（**表头用这个**）；(b) 回退到文中最后一个数，便于和 v2 对比。同时打印 `finish_reason=length` 比例、平均生成 token、输出等于题面初始量的比例，以及输出是否刚好是一行 `#### N`。

模型和 revision 与前两轮相同：`Qwen/Qwen3-4B-Instruct-2507`，钉在 `cdbee75f17c01a7cc42f958dc650907174af0554`。L4 用 bf16，T4 自动改用 fp16。共 500 条只跑 DA 的作业，`d ∈ {1, 2, 3, 5, 20}` 各 100 题。

## 在 Colab 上怎么跑

1. 把 `colab_calib_da_v3.ipynb` 上传到 Colab（或从 Drive 打开）。
2. 运行时选 GPU：**优先 L4**（会用 bf16）。T4 也可以，脚本会按算力自动改用 fp16。
3. **依次运行全部单元格**。第一格会安装 vLLM。
4. 预估时间（含模型下载与加载）：
   - L4：大约 **5–12 分钟**；若该会话里模型已缓存，生成本身大概 1 分钟以内（v2 在 L4 上 500 条、`max_tokens=64` 只用了约 43 秒；v3 更短）。
   - T4：大约 **10–20 分钟**。
5. 运行过程中会打印每个 d 的严格正确率（表头）、回退正确率、截断率、平均 token、回显初始量比例、以及是否单行 `#### N`。
6. 最后一格会通过 `files.download` 自动下载：
   - `raw_outputs_da_v3.jsonl`
   - `env.json`
7. 把这两个文件放到本地目录 `/workspace/new_task_a/calib_da_v3/`（不要覆盖 v2 或 `calib_int` 的结果），然后在本目录运行：

```bash
../.venv/bin/python analyze_da_v3.py raw_outputs_da_v3.jsonl
```

分析脚本会对照 `../calib_da_v2/raw_outputs_da_v2.jsonl` 里的 DA v2（同时给严格列和回退列），以及 `../calib_int/raw_outputs.jsonl` 里的旧 DA 和已有 CoT。缺哪个文件就会跳过那一列，不会报错中断。输出带 Wilson 95% 置信区间的对比表，并写出 `metrics_da_v3.json` 和 `da_v3_compare.png`。Task A 的达标判断用的是 **DA v3 严格正确率**。

## 本目录文件

| 文件 | 说明 |
|---|---|
| `colab_calib_da_v3.ipynb` | 自包含 notebook（数据 + few-shot + 运行脚本都嵌在里面） |
| `calib_da_v3_slim.jsonl` | 500 道评测题，内容与 `calib_da_v2_slim.jsonl` 完全相同（SHA256 `c3820a69e072fa6d4a6211c704dd64640af7674a85b32b2e0d06ac6b0aa57c51`） |
| `few_shot.json` | 系统提示、user 提醒、预填，以及 4 条 few-shot |
| `run_calib_da_v3.py` | 与 notebook 中运行单元相同的脚本 |
| `analyze_da_v3.py` | 本地对比分析（不覆盖 v2 输出） |
| `dry_run.py` | 无 GPU 的提示词拼装 / 预填 / 解析自检 |
| `qwen3_chat_template.jinja` | 官方 Qwen3 chat template，供本地 dry-run 使用 |

## 注意

- 不要改仓库，也不要动旧的 pointer 数据集。
- 这次不跑 CoT；和 CoT 的 gap 用第一轮已经跑完的 CoT 结果来算。
- few-shot 的 assistant 只有 `#### N`，不要改成带步骤的答案，否则就变成在教模型做 CoT。
- **DA(1) 的表头数字是严格正确率**，不要再用回退解析去宣称达标。
- 预填只能约束输出格式，不能阻止模型在内部默默推理；它要解决的是 v2 里「写出来的思维链被截断后，靠最后一个数误判为正确」的问题。
- `max_tokens=16` 加上遇换行就停，正常的直接作答（一个数字）足够写完；如果模型仍在 `#### ` 后面写句子，严格分会记为不正确（这是预期行为）。
- notebook 是自包含的，在 Colab 里不必再上传 jsonl；本地这两个数据文件主要用于核对和复现。
