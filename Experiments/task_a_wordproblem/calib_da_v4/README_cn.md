# calib_da_v4 — 角色提示 + continue_final_message 的 DA 校准

v3 在 L4 上的严格正确率是 d=1/2/3/5/20 = 0.70/0.08/0.03/0.00/0.02（提交 `3103148` 的笔记本输出；`raw_outputs_da_v3.jsonl` 尚未入库）。截断已经没有了，但 d=1 仍低于 0.80–0.95。本次只改 DA 的开头提示和预填的渲染方式，题目与 v3 完全相同（同一 500 题，slim SHA256 不变）。不跑 CoT，不看 Task B，没有乘法。结果还没有。

**表头指标仍是严格正确率**：把预填 `#### ` 接回生成文本后，只认 `#### N`。回退到文中最后一个数只作对照。

## 相对 v3 改了什么

1. 系统提示在最前面加上角色句，格式要求留在后面：
   `You are a math problem solver. Solve the following math problem. Your entire reply must be exactly one line in the form #### N.`
2. 每条 user 消息仍追加 `Reply with only one line: #### N.`
3. 预填不再是 `add_generation_prompt=True` 之后手工拼接。最后一条消息是 assistant，内容为 `#### `（含末尾空格）。渲染使用
   `apply_chat_template(..., add_generation_prompt=False, continue_final_message=True)`。
   已完成的 few-shot assistant 仍是完整的 `#### N`，并且以 `<|im_end|>` 结束。生成从预填的空格后面继续，不会再开一个 assistant 头。
4. 采样不变：temperature 0.6，top_p 0.95，`stop=['\n']`，`max_tokens=16`。
5. few-shot 直接沿用 v3 的 4 条 d=1（seed 20261113），没有重新抽题，也不与 500 道评测题的 qid / 题干重叠。
6. `env.json` 会记录 `continue_final_message`、`add_generation_prompt`、`prefill`、`role_prompt` 和完整系统提示。

模型仍是 `Qwen/Qwen3-4B-Instruct-2507`，钉在 `cdbee75f17c01a7cc42f958dc650907174af0554`。L4 用 bf16，T4 用 fp16。

## 在 Colab 上怎么跑

1. 打开本目录的 `colab_calib_da_v4.ipynb`。
2. 运行时选 GPU，优先 L4。
3. 依次运行全部单元格。脚本只接受以 `.jsonl` 结尾的命令行参数，Colab 的 `-f` 不会被当成输入文件。
4. 最后一格下载 `raw_outputs_da_v4.jsonl` 和 `env.json`。放到本目录，不要覆盖 `calib_da_v2/`、`calib_da_v3/` 或 `calib_int/` 里的结果。
5. 在本目录运行：

```bash
uv run python analyze_da_v4.py raw_outputs_da_v4.jsonl
```

缺少 v3、v2 或第一轮原文时，对应列会跳过，不会中断。达标判断用 **DA v4 严格正确率**。

## 本目录文件

| 文件 | 说明 |
|---|---|
| `colab_calib_da_v4.ipynb` | 自包含笔记本，输出已清空 |
| `calib_da_v4_slim.jsonl` | 与 v2/v3 slim 同一 500 题，SHA256 `c3820a69e072fa6d4a6211c704dd64640af7674a85b32b2e0d06ac6b0aa57c51` |
| `few_shot.json` | 角色提示、4 条沿用的 few-shot、预填 |
| `run_calib_da_v4.py` | 与笔记本运行单元相同的脚本 |
| `analyze_da_v4.py` | 本地对比，只写本目录的 `metrics_da_v4.json` 和 `da_v4_compare.png` |
| `dry_run.py` | 无 GPU。用官方 `apply_chat_template(..., continue_final_message=True)` 检查 prompt 以 `#### ` 结尾 |
| `qwen3_chat_template.jinja` | 钉住版本的 chat template，供对照 |
| `SHA256` | slim 与 few_shot 的校验值 |

本地自检：

```bash
uv run python dry_run.py
```
