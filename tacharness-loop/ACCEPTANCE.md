# 实现与动作验收

通用 loop 已实现；28 项回归测试通过。已使用本机 Codex CLI，实际模型与推理强度均核对为 gpt-6-astra / medium。

## 原生动作验收

| 策略 | 动作 | 训练更新 | 原生策略回合 | 验收 |
|---|---|---:|---:|---|
| act | select_ckpt | — | 2 | 通过 |
| act | finetune_more | 50 | 2 | 通过 |
| act | finetune_data | 50 | 2 | 通过 |
| dp | select_ckpt | — | 2 | 通过 |
| dp | finetune_more | 50 | 2 | 通过 |
| dp | finetune_data | 50 | 2 | 通过（保留一次启动故障及独立恢复记录） |
| pi05 | select_ckpt | — | 2 | 通过 |
| pi05 | finetune_more | 50 | 2 | 通过 |
| pi05 | finetune_data | 50 | 2 | 通过 |

六项训练均确认了精确优化器更新数、权重严格加载、参数变化和最终检查点严格重载。保存于第 25、50 步的检查点用于验证中间保存；验收后删除了六个第 25 步检查点，释放约 5.06 GiB，最终检查点保留。

DP 换数据的一个验收实例曾在 Isaac 初始化期间挂起，未创建任务或执行策略。该失败记录原样保留；独立启动探针通过后，在新目录验证该 seed，并复用另一个已完成回合。恢复仅属于技术验收，不进入正式 benchmark 结果，也未改变其补试规则。

## 真实模型门控

| 目标组 | 选择 | 停止原因 | CLI 调用 | 记录模型请求 | token |
|---|---|---|---:|---:|---:|
| univtac_hole_act_rgbt | 基线（第 0 轮） | diagnosis_failed | 2 | 12 | 269770 |
| univtac_hole_dp_rgbt | 基线（第 0 轮） | agent_stop | 1 | 6 | 151135 |
| univtac_hole_pi05_rgb | 基线（第 0 轮） | agent_stop | 1 | 6 | 150792 |

ACT 在一次纠错后仍未通过最佳验证步数的事实核验，因此没有执行更新动作；DP 与 π0.5 的 stop requirement 通过了证据和事实核验。三组均保留基线，不能据此宣称性能提升。

## 正式评测状态

门控已经封存。三组各 30 个 test seed 正在继续运行，选中基线的结果会直接复用已登记评测；不再次运行同一批策略回合。最新进度见 [STATUS.json](STATUS.json)，最终测试收据写入各组的 test_receipt.json。

后台收尾进程为 orchestrator.finalize_campaign；它通过已登记 test 的锁等待结果，再保存最终状态。请不要额外启动一个 all 调度器。

## 文件与边界

- [使用说明](README.md)
- [跨 benchmark 接口](ARCHITECTURE.md)
- [首次诊断对照](reports/diagnostic-comparison.md)
- [真实动作验收收据](runs/acceptance/action_acceptance.json)
- [原始启动失败](runs/acceptance/dp/finetune_data/feedback/20002/result.json)
- [技术恢复收据](runs/acceptance/dp/finetune_data/acceptance_resolution.json)

原仓库、原环境和共享权重保持只读；不使用 SHA-256 防御机制。/dev/shm 仅用于用户授权的临时仿真 IPC。LIBERO 仅提供接口，没有访问另一台服务器；harness 进化已移除。
