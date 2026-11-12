# calib_da_v5 — 换成 Qwen3-4B，并关掉 thinking

v4 的提示、500 题、4 条 few-shot、预填和采样都不变。这一轮只换模型：使用 2025 年 4 月的混合模型 `Qwen/Qwen3-4B`，**不是** `Qwen/Qwen3-4B-Base`，也**不是** `Qwen/Qwen3-4B-Instruct-2507`。revision 钉在 `1cfa9a7208912126459214e8b04321603b3df60c`。结果还没有。

Qwen3-4B 会思考。这次 DA **必须关掉 thinking**：`apply_chat_template(..., enable_thinking=False)`。官方模板对最后一条 assistant 预填会先写一个空的 `<think>\n\n</think>\n\n`，再接 `#### `。生成从 `#### ` 后面继续，不会停在未闭合的 `<think>` 里。`env.json` 记录 `enable_thinking: false`、`continue_final_message`、`prefill` 和完整系统提示。

**表头指标仍是严格正确率**：把预填接回生成文本后只认 `#### N`。

## 和 v4 相同的部分

- 同一 500 题，slim SHA256 `c3820a69e072fa6d4a6211c704dd64640af7674a85b32b2e0d06ac6b0aa57c51`。
- 系统提示：`You are a math problem solver. Solve the following math problem. Your entire reply must be exactly one line in the form #### N.`
- 用户消息仍追加一行 `#### N` 提醒。
- 最后一条消息是 assistant，内容 `#### `（含末尾空格）。`add_generation_prompt=False`，`continue_final_message=True`。已完成的 few-shot 仍是完整的 `#### N`。
- temperature 0.6，top_p 0.95，`stop=['\n']`，`max_tokens=16`。
- 4 条 d=1 few-shot 沿用 v3/v4（seed 20261113），没有重抽。

## 在 Colab 上怎么跑

1. 打开本目录的 `colab_calib_da_v5.ipynb`。
2. 选 GPU，优先 L4（bf16）；T4 用 fp16。
3. 依次运行。脚本只接受以 `.jsonl` 结尾的参数，Colab 的 `-f` 不会被当成输入文件。
4. 下载 `raw_outputs_da_v5.jsonl` 和 `env.json` 放到本目录。不要覆盖 v2、v3、v4 或 `calib_int` 的结果。
5. 在本目录运行：

```bash
uv run python analyze_da_v5.py raw_outputs_da_v5.jsonl
```

缺少更早的原文时，对应列会跳过。达标判断用 **DA v5 严格正确率**。

## 本目录文件

| 文件 | 说明 |
|---|---|
| `colab_calib_da_v5.ipynb` | 自包含笔记本，输出已清空 |
| `calib_da_v5_slim.jsonl` | 与 v2/v3/v4 相同的 500 题 |
| `few_shot.json` | 与 v4 相同的提示和 4 条 few-shot |
| `run_calib_da_v5.py` | 与笔记本运行单元相同的脚本 |
| `analyze_da_v5.py` | 只写本目录的 `metrics_da_v5.json` 和 `da_v5_compare.png` |
| `dry_run.py` | 无 GPU。检查模型是 Qwen3-4B、thinking 关闭、prompt 以 `#### ` 结尾 |
| `qwen3_chat_template.jinja` | 该 revision 的 chat template |
| `SHA256` | slim 与 few_shot 的校验值 |

```bash
uv run python dry_run.py
```
