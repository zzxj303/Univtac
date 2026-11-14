# π0.5 official50 六组评测

HDMI、斜孔、拔钥匙各自的 RGB／RGB＋触觉，共六份 best 权重。训练已完成并迁移核验。用户本次要求先停止旧评测、清理评测输出，重新评测；用户随后要求增加两组，当前并发上限为四组；训练 runs、base、stats 和训练验收记录不删除。

## 当前协议

每组 seed 20001–20030，共 180 回合。新一轮从 20001 开始，旧结果不计入。检查与 50 条示范的 seed/source_seed 无交集，不做专家筛选。

**超时回合记为异常，继续剩余回合和任务，不因超时停止队列。** 原生初始化超时保留为 error，整回合超时保留为 timeout；均计入固定分母 30，不重试或替换 seed。普通任务失败也继续。只有非超时的模型加载／环境创建等基础设施错误仍会停止队列，避免持续消耗种子。

保留原生 clean51、reset/pre_move、成功／提前停止规则和初始化时限；HDMI 最多 600 次动作，其余 300 次。整回合上限 1800 秒，包含模型和仿真启动。未完成时报告 pending，不将部分结果当作最终成功率。

## 启动和查看

从 UniVTAC 根目录执行：

```bash
.venv-pi05/bin/python scripts/pi05_official50/evaluation/validate_offline.py
.venv-pi05/bin/python -u scripts/pi05_official50/parallel_evaluation.py --gpu 0 --concurrency 4
.venv-pi05/bin/python scripts/pi05_official50/evaluation/status.py
```

并行调度器自动补位，新组尽量错峰初始化；单组完成后启动后续任务。每个新回合至少检查 24 GiB 可用显存、24 GiB 可用内存和 8 GiB 磁盘。不要重复启动调度器。日志、PID、当前组和排队组见 outputs/pi05_official50/evaluation_launch/。

正式启动器显式记录用户要求在完整仿真验收未通过时运行，不伪造 integration acceptance；仍要求六组离线验收通过且权重哈希匹配。start_formal.py 是串行备用入口，同样默认超时继续。runner.py 的 --stop-on-timeout 是保留的显式选项，当前调度器不传该选项。

## 模型与输入

每组绑定 best.pt、配置、训练集 quantile 统计和共享基础权重及代码哈希。head/wrist 加可选左右 rgb_marker；保持训练通道顺序，256 缩放后走 224 resize/pad。固定任务指令、50 步预测、10 步 Euler flow；前七维还原绝对关节目标，夹爪绝对值，每次只执行首动作再观测。

使用 .venv-pi05，worker 内补入现有仿真依赖；Taxim 固定 Torch 后端以匹配 ACT／DP，避免 OpenPI 的 JAX 改变 auto 选择。策略噪声独立按回合 seed 初始化，模型加载后恢复环境 RNG。

## 记录及检查

每组保留 manifest、runtime、summary，每个 seed 保留 request、started、worker.log、resolved_config、progress、动作／状态 trajectory、worker_result、最终 result 及 native 视频。初始化失败和超时的视频可能不完整，只记录实际文件。

已有结果不重跑；未结束的旧尝试记为 interrupted，只有未开始 seed 才续跑。组锁阻止重复实例，权重／统计量／协议变化拒绝续接。超时清理该 worker 的进程组，释放资源后继续。清理旧结果、从头重测仅在用户明确要求时执行。

离线检查覆盖十项调度测试，包括初始化超时和整回合超时后的持续执行、固定分母和不重试；另检查六组真实模型加载、训练预处理一致、动作归一化往返、有限 8 维动作、RGB 不依赖触觉字段、独立 RNG。离线通过不代表仿真成功率。


## 四组并行试运行

已按用户要求将并发上限从 2 提高到 4，保持正在运行的两个 runner 及其 seed 不变，追加斜孔 RGB／RGB＋触觉。调度器等待活动组进入策略阶段后逐组加入，以减少初始化竞争；拔钥匙两组待空位自动补入。显存容量允许四组试运行，但实际吞吐仍应以完成回合数和耗时衡量。新调度日志为 evaluation_launch/parallel_four.log，接管记录为 four_slot_adoption.json。超时计异常并继续的规则不变。
