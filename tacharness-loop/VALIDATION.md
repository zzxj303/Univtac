# 验收记录

此文件由 `python -m orchestrator.status` 从真实收据生成。未完成的评测不作为实验结论。

运行状态：`complete`。

## 本机 LLM

| 条件 | 首次调用 | 完整计量 | 实际模型 | 推理强度 |
|---|---:|---:|---|---|
| harness | 18/18 | 18 | gpt-6-astra | medium |
| raw | 18/18 | 18 | gpt-6-astra | medium |

冻结预算：每组 1156783 tokens；6 次 CLI；150 次模型请求。

## 训练与推理验收

| 策略 | 动作 | 训练更新 | 反馈回合 | 完成 |
|---|---|---:|---:|---|
| act | select_ckpt | — | 2/2 | True |
| act | finetune_more | 50 | 2/2 | True |
| act | finetune_data | 50 | 2/2 | True |
| dp | select_ckpt | — | 2/2 | True |
| dp | finetune_more | 50 | 2/2 | True |
| dp | finetune_data | 50 | 2/2 | True |
| pi05 | select_ckpt | — | 2/2 | True |
| pi05 | finetune_more | 50 | 2/2 | True |
| pi05 | finetune_data | 50 | 2/2 | True |

## 正式 test

| 目标组 | 已完成 | 完成整组 |
|---|---:|---|
| univtac_hole_act_rgbt | 30/30 | True |
| univtac_hole_dp_rgbt | 30/30 | True |
| univtac_hole_pi05_rgb | 30/30 | True |

## 门控

- univtac_hole_act_rgbt：第 0 轮；停止原因 `diagnosis_failed`。
- univtac_hole_dp_rgbt：第 0 轮；停止原因 `agent_stop`。
- univtac_hole_pi05_rgb：第 0 轮；停止原因 `agent_stop`。

## 边界与限制

- 原目录按 `.start_marker` 检查；不使用 SHA-256 或其他摘要。
- `/dev/shm` 仅用于用户许可的临时仿真 IPC。
- LIBERO 仅提供适配器接口；进化已移除。
- 事实错误或无效证据会阻止动作；模型输出不等于已通过验收。
- `runs/`、`runtime/` 内含私有运行状态，不应随开源代码发布。
