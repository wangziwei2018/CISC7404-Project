# Task A Stage 1.5 续接交付：评分版本 v3

当前自由文本评分入口为 `free_text_scoring_v3.ipynb` 中的 `score_free_text`；旧 v2 和 Stage 1 文件原样保留。
本轮只做 CPU 自由文本审查，不运行任何模型，不重写已有问题集，不修改数学规格。当前分支 work。

## 续接核验与完成状态

续接时已检查 AGENTS.md、task.md、record.md、分支、git status 和 git diff。
Stage 1 manifest 三个受保护文件哈希通过；旧 v2 报告中的实现、测试、68条样例及逐例评分哈希均通过。
上次执行产物确实包含19项 Stage 1 测试、8项评分测试、68条手工样例与500组随机检查通过的证据，并非仅凭聊天状态推断。

已完成：新旧版本隔离、重复声明策略更新、Markdown/跨行覆盖、分母/分量/歧义/泄漏统计、CPU回归、逐例结果及完整日志。
未完成但不属于本阶段：实际 Qwen3-4B 探针、自由文本逐步轨迹自动对齐、GPU runner 的计划请求计数与元数据适配。不得将本报告当作模型性能报告。

## 明确的协议差异与授权

旧 Stage 1/v2 对所有重复最终声明均判失败。用户本轮明确允许完全相同的重复声明，故 v3 将所有完整最终声明规范化为整数对：

- 全部相同则作为一个最终答案，重复次数另记。
- 至少两个完整声明冲突则 ambiguity=True、answer_parse_ok=False、answer_ok=False。
- 存在畸形/未完成的显式最终声明时不回退到较早答案。

原68条样例和旧评分没有覆盖。仅 `duplicate_markers` 的v3期望按本轮授权调整，变更依据记录在 v3 结果文件；其余原样例期望保持。
原文档 record.md 中的 da_trace 为“尚未应用”的历史建议（至少3行且含算术）；上次 Stage 1.5 已采用至少2个状态/步骤/算术证据的启发式。v3沿用该指标口径，不宣称它与历史模型结果可直接比较，也不把它并入 answer_ok。

## 新版本具体行为

- 支持带正负号和零的整数对、A/B明确标注、跨行答案、简单粗体/行内代码、Markdown最终标题（可省略冒号）。
- 有推理前文时必须有明确最终声明。带 Step 等上下文的中间 pair 不作为最终答案；缺失或畸形最终声明绝不回退。
- 正确的中间状态 + 错误的最终状态，应成功解析最终状态并判 answer_ok=False。
- 裸答案只在整个正常结束的响应就是答案时接受；没有最终声明的多步文本不猜测最后一对。
- 截断时末尾完整明确的最终答案允许判分；truncated/truncation=True。只有裸 pair 则保守失败。
- 完整最终答案后还有被截断的无数字、无否定/不确定措辞说明尾段，可保留答案；若出现新候选、撤回措辞或未完成的新最终声明，保守失败。
- raw_response 完全保留。解析用副本规范化，保存原行索引、parse_reason 和 scoring_version。
- DA泄漏只记录诊断，不改变内容正确性。相同重复声明不会被当作推理泄漏；冲突的最终声明本身也不作为推理证据。

本轮测试曾发现冲突最终声明被旧启发式视作两行状态追踪，导致泄漏率升高。v3为诊断单独排除最终声明候选行及其明确答案体，同时保留冲突解析失败。已有算术前文仍照常计入泄漏。

## 字段与统计口径

`answer_parse_ok` 为唯一完整最终整数对抽取成功；`answer_ok` 为抽取对与gold完全相同；`final_answer_ok` 为兼容别名；`parse_failure=not answer_parse_ok`。
`ambiguity` 只表示已识别的冲突完整声明、多个未标注候选或显式替代答案。其他不支持/畸形语法也会失败，但不全部计为歧义，故歧义率是可检测歧义率，不是语义不确定性的完整估计。

给定 N 条输出、P 条解析成功、C 条答案正确：

| 指标 | 分子 / 分母 |
|---|---|
| Main accuracy | C / N |
| Parse success rate | P / N |
| Conditional accuracy | C / P |
| Parse failure rate | (N-P) / N |
| Ambiguity rate | ambiguity=True 的记录数 / N |
| Truncation rate | finish_reason=length 的记录数 / 元数据已知记录数；另报未知数 |
| A、B分量正确率 | 完整成功解析且该分量正确的记录数 / N |
| A、B条件分量正确率 | 完整成功解析且该分量正确的记录数 / P |
| DA泄漏率 | DA中 da_arith 或 da_trace 为真的记录数 / 全部DA记录数 |

空分母返回 null。解析失败时不进行gold引导的局部补全，A/B均记未正确。失败和歧义从不移出主分母。
`truncated` 是 `truncation` 的兼容别名；finish_reason缺失时均为null。abort/error另报runtime_failure，不能当作达到长度上限。
`da_extra_text` 检测可识别最终答案之外的非空文本；解析失败的畸形回答也可能触发它，不能直接等同于推理泄漏。
`da_arith` 检测数字算式/算术词等；`da_trace` 检测至少两个轨迹证据单元；`da_leakage=da_arith or da_trace`。这三项对CoT为null。
推理证据检测不看gold。冲突最终声明的候选行从推理证据中排除，但不能据此让它通过答案解析。
不能混合v2和v3评分记录进行汇总，接口显式拒绝。

## 可复算的统计例子

所有样例 gold=(1,2)，均为DA：

| 输出 | 解析成功 | 答案正确 | A正确 | B正确 | 其他 |
|---|---|---|---|---|---|
| 算式前文，Final answer: (1,2) | 是 | 是 | 是 | 是 | DA泄漏 |
| (1,3) | 是 | 否 | 是 | 否 | 无 |
| (9,2) | 是 | 否 | 否 | 是 | 无 |
| Final answer: (1,2) 与 Final answer: (2,1) | 否 | 否 | 否 | 否 | 歧义 |
| 空输出，finish_reason=length | 否 | 否 | 否 | 否 | 截断 |

因此 Main=1/5=20%，Parse success=3/5=60%，Conditional=1/3≈33.33%。
解析失败率2/5=40%，歧义率1/5=20%，截断率1/5=20%，DA泄漏率1/5=20%。
A、B分量准确率各2/5=40%；条件分量准确率各2/3≈66.67%。
完整输入、评分与汇总保存在 free_text_audit_results_v3.json 的 calculation_example。

## 实际测试结果

| 测试组 | 实际结果 |
|---|---|
| Stage 1原回归 | 19/19通过；291600种单步组合、1000条长轨迹性质检查保留 |
| v2评分回归 | 8/8通过；原68条标注保持原结果 |
| v3评分与新增策略 | 15/15通过，0失败、0错误、0跳过 |
| v3手工文本样例 | 原68条加24条，共92/92符合v3标注期望 |
| 随机中间答案保护 | 500组，每组错误最终答案与不完整最终答案各1次，共1000次检查通过 |
| 版本与数据保护 | Stage 1、v2所有已有Task A文件与续接时SHA256一致 |

人工样例是模拟真实自由文本形态的对抗文本，不是真实模型生成。92/92是标注决策符合率，不是模型准确率或自然输出解析成功率。
每一项 unittest 的实际名称和结果见 free_text_test_results_v3.txt；每一个手工文本的原文、期望、实际结果见 parser_case_results_v3.jsonl。

v3的15项分别是：

| 测试 | 结果 |
|---|---|
| test_adversarial_cases | PASS |
| test_full_state_is_not_difference_only_convergence | PASS |
| test_gold_independent_extraction | PASS |
| test_invalid_api_inputs | PASS |
| test_no_score_filtering_by_da_leakage_or_truncation | PASS |
| test_primary_and_conditional_denominators | PASS |
| test_recovery_counts_consecutive_observed_errors | PASS |
| test_seeded_wrong_final_never_uses_intermediate | PASS |
| test_component_and_diagnostic_denominators | PASS |
| test_conflicting_final_claims_are_not_reasoning | PASS |
| test_duplicate_declarations_are_not_leakage | PASS |
| test_leakage_does_not_change_content_score | PASS |
| test_mixed_scoring_versions_rejected | PASS |
| test_new_case_ambiguity_labels | PASS |
| test_truncation_alias_unknown_and_nontruncated_malformed | PASS |

## 保留的轨迹指标分母

数学模块未修改。令X_t=(A_t,B_t)，b_t为第t步分支：

- 分支切换率：t=2..d中b_t!=b_(t-1)次数 / (d-1)。
- 最长同分支段：连续相同b的最大长度，不是比例，没有分母。
- 两步完整状态抵消率：t=2..d中X_t=X_(t-2)次数 / (d-1)。
- 固定邻居合流率：前一步完整状态不同、本步完整状态相同的事件数 / 前一步完整状态不同的比较机会数。
- 恢复率：两个相邻预测状态均有观测，前一步错误、下一步回到gold的次数 / 两步均有观测且前一步错误的次数。不是按错误片段计数，也不是只除以恢复过的片段。

固定邻居仍为(A+2,B-2)、(A-2,B+2)、(A+4,B-2)、(A-4,B+2)。合流后不再产生比较机会。
D相同不代表完整状态合流：u=2,v=1时，(45,43)与(42,46)分别变成(43,44)、(44,45)，D都为-1而完整状态不同。此反例测试通过。
分母为0时率为null；跨题应汇总事件数与机会数再相除。

## 局限与首次探针建议

- 仍是有界语法的保守解析器：代码围栏、LaTeX、JSON列表、反向标签、任意散文与复杂自我修正不全面支持，可能产生模式相关拒绝率。
- 截断后的说明尾段使用启发式判定，不能识别所有自然语言撤回或歧义；因此必须保留原文审查失败与异常成功样本。
- DA泄漏启发式可能漏检散文推理或误识别引用，不等同于可靠的思维过程测量。
- 自由文本到逐步状态/分支的自动对齐仍未实现；轨迹指标继续要求结构化输入，不伪造缺失指标。
- 没有真实模型输出覆盖率证据，没有验证GPU runner。仅可建议批准后进行小规模配对探针，先查看各模式解析率和失败类型。

评分接口已具备首次小规模Qwen3-4B探针的条件；模型运行仍需单独批准。冻结v3后若扩充语法，须另立版本并重评两模式所有原始输出，保留旧结果，不能只重评有利样本。

## 文件变更与复现

本轮新增：free_text_scoring_v3.ipynb、free_text_scoring_v3_tests.ipynb、parser_case_results_v3.jsonl、free_text_audit_results_v3.json、free_text_test_results_v3.txt、STAGE15_REVIEW_V3.md。
本轮仅向record.md追加完成记录。之前的Stage 1和v2文件均原样保留，未覆盖历史实验。

从仓库根目录，沿用现有CPU虚拟环境和Jupyter可写目录设置，运行：

```bash
.venv/bin/jupyter nbconvert --to notebook --execute \
  Experiments/task_a/free_text_scoring_v3_tests.ipynb \
  --output /tmp/task_a_free_text_v3_executed.ipynb \
  --ExecutePreprocessor.timeout=180
```

该notebook直接读取已有68条样例，仅执行旧测试单元，不执行Stage 1数据生成单元。派生v3评分结果会刷新；旧数据、旧结果不覆盖。
Git标准diff不包含未跟踪文件，不能用“只有record.md变更”描述全部工作。本轮新文件差异另保存到 `/workspace/scratch/task_a_stage15_review.patch` 供审查；Git状态与完整文件清单见本次交付消息。
未commit、push、merge、reset或切换分支。完成后停止，等待人工审查。
