# official50：十二组固定评测入口

已完成脚本准备与离线验收；随后按用户要求启动十二组正式评测队列，最多并发六组，实际状态以 evaluation_launch/status.json 为准。离线检查使用已经录制的观测，不运行任务环境。当前不可将未完成的部分结果当作最终成功率。

## 固定实验协议

- 三任务：insert_HDMI、insert_hole、pull_out_key；每任务 ACT/DP × RGB/RGB＋触觉，共 12 组。
- 全部使用 `outputs/official50/<act|dp>/<任务>/<模态>/policy_best.ckpt` 和同目录训练配置及归一化统计量，禁止静默回退到 last 权重。
- 每组固定 seed 20001–20030，共 30 次；正式执行全部脚本后是 360 回合。训练和验证所用全部 50 条示范的 metadata.seed 与 metadata.source_seed 都被排除。每次启动会重新核验无交集。
- 不进行专家预筛选，不读取历史 seeds.json，不重试，不替换失败种子。同一任务的四种模型共享 seed 清单，但浮点物理仿真不保证逐位相同的初态。
- 沿用 `clean51` 的原生物理、控制频率、reset/pre_move、成功条件及提前终止。策略从原生准备动作后接管，不代表从桌面抓取开始的全流程操作。
- HDMI 上限 600 次动作，另两项 300 次。每个 seed 独立子进程，墙钟上限 1800 秒，包含应用启动、模型加载、初始化、策略运行和清理。原生 reset_time_limit 不作修改。

## 模型推理约定

相机均为 head＋wrist，触觉组增加左、右 rgb_marker。使用与数据准备一致的 OpenCV INTER_LINEAR 256×256 缩放及通道顺序；RGB 使用 ImageNet 归一化，触觉只除以 255。RGB 模型不会访问触觉字段，但物理环境仍保留原生触觉传感器。

ACT 复用官方 `ACT` 类、50 步预测、每步重新预测及原有时序聚合，并使用训练时的 mean/std。配置显式绑定 best 权重路径。

DP 复用此次训练的独立 ResNet18 编码器与条件 U-Net，min/max 来自训练保存值；输出预测长度 20。采用上游推理的 **10 次 DDPM 去噪**（训练 diffusion_timesteps=100），**每次只执行第一个动作，再重新观测预测**。这是本次 UniVTAC 部署适配的闭环执行设置，不声称是原真实机器人入口无需修改的行为。DP 的初始噪声和每次调度器噪声均使用按 seed 初始化的独立 GPU 随机数生成器。

两种模型都输出 8 维绝对关节目标，通过原生 `take_action(action_type='qpos')` 执行。

## 十二条正式启动命令

从 UniVTAC 项目根目录运行：

```bash
cd /root/autodl-tmp/gujie/UniVTAC
bash scripts/official50/evaluation/act_insert_HDMI_rgb.sh --gpu 0
bash scripts/official50/evaluation/act_insert_HDMI_rgb_tactile.sh --gpu 0
bash scripts/official50/evaluation/dp_insert_HDMI_rgb.sh --gpu 0
bash scripts/official50/evaluation/dp_insert_HDMI_rgb_tactile.sh --gpu 0
bash scripts/official50/evaluation/act_insert_hole_rgb.sh --gpu 0
bash scripts/official50/evaluation/act_insert_hole_rgb_tactile.sh --gpu 0
bash scripts/official50/evaluation/dp_insert_hole_rgb.sh --gpu 0
bash scripts/official50/evaluation/dp_insert_hole_rgb_tactile.sh --gpu 0
bash scripts/official50/evaluation/act_pull_out_key_rgb.sh --gpu 0
bash scripts/official50/evaluation/act_pull_out_key_rgb_tactile.sh --gpu 0
bash scripts/official50/evaluation/dp_pull_out_key_rgb.sh --gpu 0
bash scripts/official50/evaluation/dp_pull_out_key_rgb_tactile.sh --gpu 0
```

上述命令单独使用时各自执行一组。当前已通过持久调度器启动十二组队列；不要重复手工启动。首实例实测占用约 11 GiB，当前上限六组，并在启动新组前检查剩余显存与磁盘。

每个 shell 入口只支持 `--gpu GPU_ID`、`--output-root DIR` 和 `--help`。默认结果根目录为 `outputs/official50/evaluation/`。例如指定结果目录：

```bash
bash scripts/official50/evaluation/act_insert_HDMI_rgb.sh \
  --gpu 0 --output-root /root/autodl-tmp/gujie/UniVTAC/outputs/official50/evaluation
```

脚本调用项目 `.venv/bin/python`，不要求激活环境。GPU 通过 CUDA_VISIBLE_DEVICES 映射，进程内部使用 cuda:0。入口允许从其他工作目录启动。首次仿真启动仍依赖已安装的 Isaac Sim/TacEx/UIPC 环境；离线验收不能替代真实仿真启动验证。

## 结果、异常和续接

输出结构：

```text
<结果根目录>/<策略>_<任务>_<模态>/
  manifest.json       # 权重/统计量/代码哈希、配置、种子排除记录
  runtime.json        # 调度进程、GPU、启动时间
  summary.json        # 固定分母 30 的汇总，以及逐回合记录
  20001/
    started.json     # 子进程启动前创建的一次性记录
    request.json
    worker.log
    resolved_config.json
    trajectory.jsonl # 每次动作执行前后分别落盘，异常时保留部分轨迹
    progress.json
    worker_result.json
    result.json      # 调度器最终结果，超时/崩溃也会生成
    native/          # 官方视频、原生元数据等
  ...20030/
```

`success` 为原生成功；`failed` 为步数用尽或原生提前停止；`error`、`timeout`、`interrupted` 单列异常。所有结果都占用一个 seed，不补测。最终成功率为 success/30；未完成时明确标记 pending 和 success_rate_is_final=false，不能把部分结果作为最终成功率。

有启动记录但没有最终结果的 seed，在重启同一脚本时标为 interrupted，不再运行。已完成结果直接保留；尚未开始的 seed 才继续执行。每组有独占文件锁，避免两个入口同时操作同一组目录；若权重、配置或代码哈希变化，则拒绝混合结果。

**续接必须使用同一输出根目录，不能删除已有目录。** 去重范围是该结果目录；改用空目录代表另一轮实验，会再次执行那些 seed。

超时会清理该 worker 的整个进程组；初始化失败或强制终止时，视频可能不存在或不完整，记录中会显示实际存在的文件，绝不伪造视频。原生视频关闭等清理异常写入 worker_result.cleanup_errors；保留策略判定结果并供审计。

## 已完成的离线验收

```bash
.venv/bin/python scripts/official50/evaluation/validate_offline.py
```

该命令只测试历史观测与模拟子进程，不导入评测 worker、不启动 Isaac、不消耗任何正式仿真 seed。覆盖：十二份 best 权重严格加载、训练帧预处理逐元素一致、有限值 8 维动作、RGB 无触觉字段可推理、ACT 时序聚合、DP 10 次去噪及独立 RNG、固定分母、失败/超时/中断处理、进程清理、重复启动和文件锁。

验收报告：`outputs/official50/evaluation_preflight/report.json`；日志：同目录 `validation.log`。这些是接口检查结果，不是任务成功率。

## 当前队列进度

运行 `.venv/bin/python scripts/official50/evaluation_status.py` 查看十二组进展。调度器状态、PID、日志及并发控制文件位于 `outputs/official50/evaluation_launch/`。单组启动器会在 startup/load_policy/create_environment 阶段的错误发生后停组，已尝试 seed 保留且不重试；总调度器检测到异常退出会停止启动其余组，避免系统配置错误连续消耗测试 seed。运行环境补充 OMNI_KIT_ACCEPT_EULA=YES，以匹配已有 Isaac 安装的无交互启动。
