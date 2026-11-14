# repair-v2 提示词修改与真实验证

旧版在 DP 5/30、π0.5 2/30 的反馈成功数下选择 stop，理由包括当前已经使用最佳验证 checkpoint、缺乏证明修改会有效的因果证据。新版区分已观察事实与待验证假设，要求检查失败轨迹、训练曲线、可用关键帧及历史，比较三类允许动作后提出预算内实验。低成功率不能单独证明某个修复有效，也不保证 100% 可达。

新提示词：[repair-v2.txt](../harness/prompts/repair-v2.txt)。新建输入默认使用 repair-v2；旧输入保留 diagnostic-v1，后续轮次继承各自首轮版本，调用读取保存的原始提示词。两种消融条件输入仍完全相同，区别仍只有 th.* 可用性。

本机 Codex CLI 实际使用 `gpt-6-astra`、`medium`，对 DP 与 π0.5 各调用一次。以下均为开发诊断，未调度候选训练或评测，也未改写旧门控。

| 组 | 旧决策 | 新决策 | 候选 | 原始引用验证 | 结构化事实错误 |
|---|---|---|---|---|---|
| univtac_hole_dp_rgbt | stop | select_ckpt | policy_periodic, policy_last | 10/10 | 0 |
| univtac_hole_pi05_rgb | stop | select_ckpt | baseline, last | 9/9 | 0 |

π0.5 的 baseline 已有反馈，会由现有编排器复用；若执行新动作，需要新增评测的是 last。DP 需要评测 periodic 和 last。模型预测是待验证假设，不是新测得的成功率。动作校验确认了参数、引用、结构化事实与文件存在性，不证明模型的因果解释或视觉判断一定正确。

33 项测试通过，包括 18 份旧首轮输入逐字节不变、新条件输入一致、保存提示词不随模板修改、未知版本拒绝及正式额度版本不匹配时禁止诊断。未加入 SHA 或其他校验和机制。

两次开发调用共用 468,990 tokens（输入含缓存，加输出）；总输入、缓存输入、输出分别保存在每次 receipt 及 [validation.json](../runs/prompt-revisions/repair-v2/validation.json)。旧正式实验和旧冻结额度未重算；正式 v2 实验需按既定规则独立进行 18 次首轮 harness 额度校准，并保持消融输入与首轮复用一致。

实际结果：[report.json](../runs/prompt-revisions/repair-v2/report.json)。
