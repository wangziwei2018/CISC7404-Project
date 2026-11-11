# AGENTS.md

本文件是给 AI 编程助手（以及新加入的人类）的项目说明书。**修改代码前先读完本文件**，尤其是
第 3 节「vLLM 统一加载参数」——那是硬性要求，不可自行改动。

---

## 1. 项目是什么

CISC7404 课程项目。研究问题是：**思维链（CoT）带来的准确率提升，是否随推理深度 `d` 增长？**

核心做法是构造一个 `d × k` 的测试矩阵：

- `d` —— 推理深度 / 串行计算链长度（必须一步一步算的步数）
- `k` —— 输出约束（instruction-following constraint）的条数
- `W` —— 工作量（workload），即解题真正需要的原子运算次数；串行题 `W = d`，pointer 并行题
  `W = d × w`（`w` 个起点各走 `d` 步），Kaggle 模运算 pilot 的并行题 `W = 2w - 1`

实验自变量是「深度」，但深度天然伴随工作量变化，因此必须用同 `W`、不同 `d` 的对照
（串行 `d` 对齐并行 `w`，使 `d == 2w - 1`）把「深度效应」和「工作量效应」拆开。

数据来源（当前）：

1. **合成 pointer tracking**：`synthesis_dataset.ipynb` 生成 `synthesis_q_k_dataset.jsonl`（串行 `d × k`）和 `synthesis_d_w_dataset.jsonl`（并行 `w / d / W`）
2. **合成 arithmetic chain**：`synthesis_math_dataset.ipynb` 生成 `synthesis_math_dataset.jsonl`（小数加减链，求含起点的最小值）
3. **GSM8K**：真实数学推理数据，用 `<<expr=result>>` 注解图算真实 `d` / `W`
4. **IFEval**：指令遵循数据（目前只做到下载/预览）

对照条件固定为两种系统提示：`DA`（只输出最终答案）与 `CoT`（在输出中显式写出中间步骤）。

---

## 2. 环境与常用命令
### 2.1 命令

```bash
uv sync                                  # 按 pyproject.toml / uv.lock 同步环境
uv add <pkg>                             # 新增依赖（会改 pyproject.toml + uv.lock）
uv lock                                  # 更新锁文件

# 按机器分平台安装（依赖由三份 requirements 分别管理）
uv pip sync configs/cpu/requirements.txt            # CPU 机器
uv pip sync configs/windows-gpu/requirements.txt    # Windows GPU
# configs/linux-gpu/ 在 README 里有引用，但目录尚未创建

# PyTorch 必须手动装（CPU / GPU 平台不同）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

jupyter notebook
jupyter server --ServerApp.allow_remote_access True   # 需要远程访问时
```

### 2.4 目录

```
.
├── README.md                     安装 / 启动说明（英文）
├── AGENTS.md                     本文件
├── task.md                       当前两个任务的执行清单（问题集 / 约束集）
├── record.md                     研究计划与实验记录（中文，持续更新）
├── Proposal.pdf                  课程项目开题报告
├── pyproject.toml / uv.lock      uv 依赖定义与锁文件
├── uv.toml                       镜像源
├── prior_investigation_kaggle.ipynb   前期调研（Kaggle 模运算 pilot）
├── configs/
│   ├── cpu/requirements.txt
│   ├── linux-gpu/               空目录（README 有引用，尚未填依赖）
│   └── windows-gpu/requirements.txt
├── Experiments/                  实验代码与结果
│   ├── data_generation/          合成数据生成
│   │   ├── synthesis_dataset.ipynb          pointer tracking 生成（串行 d×k / 并行 d×w）
│   │   ├── synthesis_q_k_dataset.jsonl      串行 d×k，4000 行
│   │   ├── synthesis_d_w_dataset.jsonl      并行 d×w，4000 行
│   │   ├── synthesis_math_dataset.ipynb     小数加减链生成
│   │   └── synthesis_math_dataset.jsonl     小数加减链，4000 行
│   ├── argmin_math_experiment/   min-chain 实验
│   │   ├── experiment_math.ipynb            主实验 notebook
│   │   ├── pilot_math_results.jsonl         1.7B / 2048 token 的旧跑结果
│   │   ├── pilot_math_results_v2.jsonl      1.7B / 6144 token 的 v2 结果
│   │   ├── pilot_results.jsonl              pointer tracking pilot 结果
│   │   └── vllm_problems_solutions.md       vLLM 踩坑与解决方法（中文）
│   ├── vLLM_test.ipynb          以下 4 个为早期 notebook，历史参考
│   ├── gsm8k_test.ipynb
│   ├── Qwen3-4B-Test.ipynb
│   ├── experiment_pointer_track.ipynb
│   ├── experiment_calibration_v2(1).ipynb   4B 校准 v2 的实际代码（未提交）
│   ├── calib_v2_results_tierB.jsonl         校准 v2 原始结果（未提交，2420 条）
│   ├── calibration_v2_summary_tierB.csv     校准 v2 聚合表（未提交）
│   └── task_a_wordproblem/       加减应用题替换候选（校准未跑；见该目录 README）
├── gsm8k_dataset/                gitignore；由 notebook 生成
├── ifeval/                       gitignore；由 notebook 生成
├── models/                       gitignore；Qwen3-1.7B 已下载；4B 权重在 Colab，本地不下载
└── Thoughts/                     零散想法（Pointer_Track.md）
```

---

## 3. vLLM 统一加载参数（适用于RTX3070，如果是更好GPU可以自行调节）

**要求：所有 vLLM 推理代码都从本节模板出发，不要从零手写 `LLM(...)`。**
采样参数全项目统一；

### 3.1 两个档位

| 档位 | 模型 | 本地目录 | 用途 | 状态 |
|---|---|---|---|---|
| A | Qwen3-1.7B | `models/Qwen3-1.7B` | pilot、链路自测、回归 | 已下载，已跑通 |

- HF repo id 是 `Qwen/Qwen3-4B-Instruct-2507`。**不要**换成 `Qwen/Qwen3-4B`（base）或
  `Qwen/Qwen3-4B-Thinking-2507`（thinking-only）：base 版没有对齐好的对话能力，thinking 版会
  强制自带思考块，两者都破坏「CoT 由 prompt 控制」的设计。
- 档位 B 是**非 thinking 模型**：`tokenizer_config.json` 里的 chat template **没有**
  `enable_thinking` 分支，generation prompt 就是 `<|im_start|>assistant\n`。
  因此对档位 B **不要**传 `chat_template_kwargs={"enable_thinking": ...}`。

### 3.2 档位 A 模板（Qwen3-1.7B，RTX3070能够运行，其他GPU不必一定遵守）

```python
import os

# 必须在 import vllm 之前设置：关闭 FlashInfer 采样器，避免没有 nvcc 时的 JIT 采样扩展编译失败
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from vllm import LLM, SamplingParams

model_path = "models/Qwen3-1.7B"

# ---- 档位 A 的显存参数：8GB 卡上的稳定值，不要动 ----
vllm_model = LLM(
    model=model_path,
    gpu_memory_utilization=0.80,
    max_model_len=8192,
)

# ---- 全项目统一的采样参数 ----
PARAMS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,          # 见 3.5：目前只定义、不传
    "max_tokens": 6144,   # 按条件定，见第 4.3 节
}
sampling_params = SamplingParams(
    temperature=PARAMS["temperature"],
    top_p=PARAMS["top_p"],
    max_tokens=PARAMS["max_tokens"],
)

# ---- 生成（档位 A 必须显式关掉 thinking，否则输出被思考块占满；档位 B 不传）----
outputs = vllm_model.chat(
    messages,
    sampling_params,
    chat_template_kwargs={"enable_thinking": False},
)
```


---

### 3.3 Task A 的模型版本固定

Task A notebook 在创建 LLM 前通过 Hugging Face 公开接口 `HfApi.model_info().sha` 解析
远端模型的 main/tag/分支，将同一完整 commit 传入 `revision` 和 `tokenizer_revision`，
并记录用户请求的 revision 与解析后的 commit。不要依赖 `hf_config._commit_hash` 等私有字段；
vLLM 0.33.0 的本次 Colab 运行中该字段为空，但模型本身已经加载成功。
`MODEL_REVISION=None` 可自动解析，无需用户手找 commit；本地 snapshot 识别目录 commit，
普通本地目录须显式提供版本标签，并与已验证的远端 commit 区分记录。
显存、dtype、上下文长度和统一采样参数保持原值。

### 3.4 Task A 当前 Colab/T4 兼容配置（用户实测，2026-11-09）

提交 `c2cc018` 中用户改为以下配置后完成了 200 条真实探针，后续诊断继续沿用：

- B 档：`dtype="half"`、`gpu_memory_utilization=0.80`、`max_model_len=8192`、
  `max_num_seqs=1`、`enforce_eager=True`，仍要求至少 14 GiB 可见显存。
- import vllm 前设置 `VLLM_USE_FLASHINFER_SAMPLER=0`、
  `VLLM_ENABLE_V1_MULTIPROCESSING=0`、`VLLM_LOGGING_LEVEL=DEBUG`。
- 日志选择 `TRITON_ATTN`；不要因为日志提示 T4 不支持 FA2 就误判整个加载失败，实际有后端回退。
- 这是 T4 上针对 4B 的实际配置，不是将 1.7B 的 8GB 模板直接套用。采样参数未改。
  已验证能完成推理；随后 64 条诊断的基础控题、输入 token 与提交批量对照通过。
  这是有限控题的证据，不等于证明后端在所有题目上数值完全可靠。
- 多进程开关可能影响全局随机状态；题目和统计使用局部 `random.Random`，本轮 200 题已重建一致。
  不把一次同时修改多个选项后的成功归因于其中某个选项。

## 4. DA / CoT：只由 system prompt 决定

**最终实验没有 thinking 开关。** 档位 B 本身就不是 thinking 模型，条件只有两个，全部靠提示词区分。
这也是本项目「CoT 是否有用」的唯一操作化定义。

| 条件 | system prompt 里的指令 | 期望输出 |
|---|---|---|
| `DA`（direct answer） | 不写推理、不解释，直接给最终答案 | 只有最终答案（仍要满足 C1–Ck） |
| `CoT`（chain-of-thought） | 先显式写出每一步的中间结果，再给最终答案 | 推理 trace + 最终答案 |


## 5. 代码风格与协作约定

1. **一切实验代码写在 notebook 里**
2. 注释与讨论用**中文**，变量名/函数名/文件路径用**英文**；已有命名风格
   （`make_serial`、`q_matrix`、`ds_ifeval`、`recs_s3`）请沿用。
3. 数据流是「生成 → 落盘 JSONL → 用 `datasets` 回读」，不要长期只在内存里传递 DataFrame。
4. 结果记录：重要结论写进 `record.md`。
   新踩的坑按「问题 → 原因 → 解决方法」的格式追加。
5. 不要重跑并提交带大数据输出的 cell（例如全量生成、GSM8K 全量分析），会让 diff 无法阅读。
6. 用中文提交信息（现有 git log 都是中文，`main` 分支，直接提交）。

### 5.1 禁止事项

- ❌ 在 Windows 原生环境跑 vLLM（必须 WSL）
- ❌ 改动第 3 节的加载 / 采样参数而不更新本文件与 `record.md`
- ❌ 把档位 A 的 `gpu_memory_utilization=0.80 / max_model_len=8192` 直接套到 4B 上
  （档位 B 的当前 T4 实测配置与适用显存见第 3.4 节）
- ❌ 在 T4 上忘了 `dtype="half"`（Turing 不支持 bf16）
- ❌ 对 `Qwen3-4B-Instruct-2507` 传 `chat_template_kwargs={"enable_thinking": ...}`
- ❌ 用后处理硬截断制造 DA / CoT 的「预算公平」
- ❌ 把 min-chain 任务再当正式任务跑全曲线（已证明无可用深度区间，见第 1 节）
- ❌ 提交 `models/`、`gsm8k_dataset/`、`ifeval/`、`.venv/`（已在 `.gitignore`）
- ❌ 在没有 GPU 的机器上执行需要 CUDA 的 cell 后把报错当成代码 bug 修

### 5.5 Task A v3：lookup 已人工定题，正式数据已交付

- 入口：`Experiments/task_a/experiment_task_a.ipynb`。200 条真实寄存器探针为
  `register2_small=0/100`、`register2_easy=1/100`，两配置不进入斜率或正式生成。
  生成器仍保留用于重建历史结果：同时条件更新，无 mod/聚合，编码完整状态，`W_atomic=3W`。
- 提交 `6cea939` 及上传的 64 条诊断已经复核：预期/实际输入 token 全部相同，32 对单条/批量
  最终答案完全一致（3 对仅 CoT 文本不同）。基础抄写/读数/加减/比较/独立编码各模式均 2/2；
  寄存器组合编码题 DA 0/4、CoT 4/4，不编码读出 DA 1/2、CoT 2/2。
  重复提交不是独立样本；无解析失败或截断。有限控题不支持输入错配或提交批量故障的解释，
  也不能用来估计正式 p1。原文及审查见 `task_a_diagnostic*.json*`。
- 最终选定 `pointer_numeric_v1` / `pointer40_lookup`：固定 40 行编号表，每步 `current = next[current]`。
  历史 delta 候选每步 `current += delta[current]`，仅保留复核；
  每一步同构，答案为最终编号，无编码/mod/聚合。难度旋钮为编号位数与表行顺序；若偏难，
  试 `current = next[current]` 查表变体。不得混合不同变体统计。
  串行路径无重复，d=1..20。并行共享只读表，路径节点不重叠（w=10 最多 29 个节点）。
  链长 `[1,2,...,2]`，实际 d=1（w=1）或 2，`W=2w-1`、`matched_serial_d=W`；
  匹配串行奇数深度 1..19。delta 的 `W_atomic=2W`，lookup 为 `W`。
  输出每条链的答案，人工审核仍须评估多答案读出成本，不能声称无混杂。
- 上一轮 `RUN_STAGE="search"` 策略（保留复现）：从 `pointer40_delta`（0..39）开始，最多 3 个配置各 100 条 DA。
  太易（≥0.99）试三位编号；三位太易则打乱行，偏难则试全两位编号。起点偏难则试纯查表，
  查表太易再打乱行，仍偏难则停止。命中 [0.955,0.986] 后仅跑一次独立 5 深度×20×DA/CoT，
  然后停止供人工审核。最多 500 条新推理，未命中不选择“最接近”配置，斜率失败不无限重试。
  `probe` / `slope` 可指定配置手动运行；`diagnostic` 保留旧控题；CPU 验证用 `offline`。
  引擎 max_num_seqs=1，提交批量 16 不代表同时计算 16 个序列。
- 提交 `d2b57a4` 与 ZIP 的 200 条指针探针已复核：delta 11/100（Wilson 95% CI
  0.0625–0.1863），lookup 87/100（0.7902–0.9224）；均无解析失败、截断或输入 token 哈希/长度不匹配。
  搜索正常停止，未跑斜率。delta 不继续；lookup 在 A.0 的 DA(1) 描述范围内，但不在启发式窗口。
- 上一轮 `RUN_STAGE="slope"`、`SELECTED="pointer40_lookup"`；`PROBE_OVERRIDE_REASON`
  明确记录例外：DA 单步率并非 CoT 独立逐步成功率，因此不按 p1**20 判定 CoT(20) 地板，
  先做一次独立 5 深度×20×DA/CoT（200 条）验证唯一硬目标。不是将 0.87 宣称为探针窗口达标。
  不修改题目、prompt、seed、GPU 或采样，不重新运行已有探针；不自动批准正式生成。
- 新 Colab 会话加载 GPU 前会自动恢复 `archives/pointer_search_d2b57a4/calib_v3_task.jsonl`，
  校验固定 SHA256、逐题重建/判分/指纹后合并，保留原文、拒绝同 id 冲突。
  本地已有完整匹配探针则不联网。实际模型/环境指纹不匹配时仍拒绝复用，不篡改来源以绕过门槛。
  网络失败可手动把已导出 ZIP 的 JSONL 放入 RESULT_PATH。在线来源使用 GitHub raw，校验值固定。
- 本次先按用户要求独立归档并推送原文 `ae03c89`，再进行分析。lookup 斜率 200 条已完整复核：
  d=1,5,10,15,20 的 DA 正确数为 15/0/0/0/0，CoT 为 20/17/20/18/14（每格 20）。
  CoT(20)=0.70，DA(20)=0，gap 增长 0.45；CA 双侧 p=0.0130003 未达 0.01，不改用单侧检验。
  CoT(20) Wilson 95% CI [0.4810,0.8545]，gap 增长 bootstrap CI [0.15,0.70]；均非总体确定性结论。
  CoT 曲线和 gap 非单调，DA 从 d=5 即地板，不能宣称 gap 持续随 d 上升。
- 助手逐条读完全部 11 条 CoT 错误：9 条多走一步、2 条少走一步，未观察到这些错误中有查表转移错。
  用户已接受步数控制主导梯度的解释范围；人工批准另存 task_a_human_review.json，不把助手分析当批准。
  审查文件为 `task_a_lookup_review.md` / `.json`，曲线为 `task_a_lookup_calibration.png`。
  原上传 400 条（指针探针+斜率）及清单/摘要保存在 `archives/lookup_slope_20261109/`。
- 当前默认 `RUN_STAGE="offline"`，有完整本地 JSONL 时用 CPU 复核已完成校准；不再反复跑 GPU。
  `review_lookup_calibration()` 生成审核指标与错误端点线索，最终仍需读原文，不能按端点线索重判。
  `GENERATE_FINAL=False` 仅用于避免默认重复生成；REVIEW 从已批准的 task_a_human_review.json 读取。
  2026-11-09 用户回复“接受”，已按该审核生成并验证 2000 串行+1000 并行，重新生成仍校验结果指纹。
  不为跨过 p<0.01 而事后增样本/改方向/挑选题目，正式生成也不等于正式矩阵已测量。
- 必须先真实探针，再小样本斜率，再人工审核，正式生成默认关闭。
  **Task A 按人工接受的例外完成，正式3000题已交付；不能声称所有辅助指标数值达标。**
  使用第 3 节用户跑通的 T4 参数；A 档只自测，模拟结果不能冒充真实校准。
  `task.md` DA(1) 0.80–0.95 与探针 0.955–0.986 不相交，分别报告；`p1**20` 仅启发式预测。
  CA 检验对象为 CoT 准确率随深度下降，gap 端点差另报配对 bootstrap 区间。
- 真实探针/斜率写 `Experiments/task_a/calib_v3_task.jsonl`，诊断单独写
  `task_a_diagnostic.jsonl`，断点按题目/prompt/模型/采样指纹去重。题目写 `work/task_a/`。
  现有 canonical 600 条（寄存器探针 200 + 指针探针 200 + lookup 斜率 200），另有 64 条诊断，原始行字节保留；旧探针上传摘要
  只含 easy 组，完整结论应看 `task_a_probe_review.json`。诊断导出清单另存
  `task_a_diagnostic_export_manifest.json`，不覆盖原探针清单。
  `search` 报告为 `task_a_search_summary.json`；斜率写 `task_a_summary.json`，包含例外理由。
  指针原 ZIP 的三份文件完整存于 `archives/pointer_search_d2b57a4/`；
  当前导出清单重新记录合并后的 600 条总数与 SHA256，原寄存器清单另存 `archives/register_probe_c2cc018/`。
  最后一格下载 ZIP，保存 notebook 到 GitHub 不会保存 JSONL，须同时提交原文与摘要。
- 经真实校准与人工审核，已交付指针正式文件：
  `Experiments/data_generation/synthesis_pointer_numeric_dataset.jsonl` 与同名 `.manifest.json`；
  默认 2000 串行 + 1000 并行，与校准题不重叠。旧寄存器生成接口仍保留。
  指针查表字段：START / OPS / TABLE_SIZE / INDEX_MIN / INDEX_MAX / FIRST / LAST；
  FIRST/LAST 指印出的首/末表行。寄存器才有 MAXDELTA，不能机械迁移到指针；此处不做 Task B 筛选。
  正式数据使用说明在 `Experiments/data_generation/synthesis_pointer_numeric_dataset.README.md`。
  `task_a_final_validation.json` 记录3000题全量复算、datasets回读、确定性重建、去重、全部600条校准隔离及W配平。
  `q` 才是模型题干，不能把 `trace` / `answers` / `ans` 等gold元数据传给模型。
  正式全矩阵推理、Task B 与合并复核未执行。

### 5.6 加减应用题替换候选（校准未跑）

指针 lookup 的 DA 从 d=5 起为 0。替换候选在 `Experiments/task_a_wordproblem/`，
版本 `cond_wordproblem_v1`：每步比较阈值后加或减，没有乘法。
生成器和整数 700 题校准集已入库；Qwen3-4B 结果还没有，不能当作正式题替换 5.5 的 3000 题。
`calib_int/run_calib.py` 在 T4 上使用第 3.4 节的加载参数；算力 ≥ 8 的 GPU 用 bfloat16、
`gpu_memory_utilization=0.88`、`max_model_len=4096`。DA / CoT 的 `max_tokens` 分别是 64 和 2048。
采样仍是 0.6 / 0.95，不传 `enable_thinking`。说明见该目录 `README.md`。

## 6. 待办与未定项

**一周双轨（2026-11-07 起，详见 `record.md`「一周行动计划」）：**

- **轨道 1：问题集** —— 找一个 CoT 在 `d=1..20` 上有可测梯度的任务。
  指针 lookup 已按例外交付，但 DA 从 d=5 起为 0。下一候选是
  `Experiments/task_a_wordproblem/` 的加减应用题，校准尚未运行。
  - 第一天对 3 个候选任务各跑 `d=1 × 100 题 × DA`，量 p1；目标窗口 `0.955–0.986`。
  - 候选优先级：双寄存器条件更新（首选）> 编号数组指针 > 定长槽位覆盖写 > 栈操作。
  - 验收：`CoT(d=20) ∈ 0.40–0.75`、`DA(d=20) ≤ 0.10`（接受地板）、`gap(20) − gap(1) ≥ 0.30`、
    Cochran-Armitage `p < 0.01`。
- **轨道 2：约束集** —— 找 10 条能进 k 轴、且在 CoT 侧有成本的约束。
  - 只用指令遵循判分，不看问题正确率；测试床用 min-chain 的 `d=5..10`。
  - 三道门槛（第 4.2 节）：非空转、DA 单条 ≥ 0.95、CoT 比 DA 低 ≥ 0.15。
  - 主力是查表型：`START` / `OPS` / `FIRST` / `LAST` / `NEG` / `MAXDELTA`。
- **合并复核**：新任务 × 10 约束 × 2 模式 × 100 题 = 2000 条，不可假设轨道 2 的结论直接迁移。


