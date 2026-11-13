# UniVTAC π0.5 official50 混合微调

> 最新状态：按用户要求已停止并清理上一轮 π0.5 训练输出，从基础权重重新启动 4,000 步实验。warmup=200，cosine 在第 4,000 步降至峰值 10%；micro-batch=8、累积=4、seed=42 不变。当前先并行 HDMI／insert_hole 的 RGB、RGB＋触觉及 pull_out_key_rgb_tactile 五组，pull_out_key_rgb 等待补位。当前五组预计约 30–36 小时，六组整体保守预留 1.5–2 天；尚未启动仿真评测。








本目录实现三个任务 × RGB／RGB+触觉，共六组独立微调。正式训练须通过 `outputs/pi05_official50/acceptance.json` 验收门禁；本实验不自动启动仿真评测，训练／验证 loss 不代表仿真成功率。

## 固定实验

- 每组从同一份官方 `gs://openpi-assets/checkpoints/pi05_base` 初始化，优化器更新恰好 4,000 次。
- 训练 seed 固定 42；已有 `data/official50/<task>/split.json` 的 seed=1、40／10 轨迹划分保持不变。episode 0–49 全部使用。
- Python、NumPy、Torch CPU/CUDA、LoRA 初始化统一 seed=42；启动脚本设置 `PYTHONHASHSEED=42`，关闭 cuDNN benchmark。数据排列、DataLoader、训练 flow noise/time 各用独立 seed-42 generator；验证每次使用独立且重置为 42 的流。
- DataLoader `num_workers=0`，避免预取使恢复游标漂移；已提供符合 PyTorch worker seed 约定的初始化函数。恢复保存 Python/NumPy/CPU/CUDA RNG、排列、游标、shuffle generator、loader generator、flow generator、FP32 LoRA 主副本、AdamW 状态、优化步数和 LR。
- rank=16、alpha=32、dropout=0。SigLIP、Gemma 和视觉投影原参数冻结，参与动作预测的 attention／MLP／视觉投影 Linear 接入 LoRA。VLM 浮点参数全部 BF16；LoRA 使用模型外 FP32 优化器主副本，未量化。
- 最后一个语言层的 prefix query/output/MLP 无法影响动作 suffix，因此不接入无梯度 LoRA；语言输出头不训练，未使用的动作专家 LM head 移除。
- 完整动作专家（含 AdaRMS）、action in/out projection、time MLP 全参 FP32，BF16 autocast；不使用专家 LoRA。
- 有效 batch=32，当前固定 micro=8／accumulation=4；四组并行。新配置经对应 batch 的数值及恢复验收后启动，实际参数记录于验收和运行配置。
- AdamW betas=(0.9,0.95)、weight decay=0.01；LoRA／专家峰值 LR=1e-4／5e-5，200 步 warmup 后 cosine 至峰值 10%；全局梯度裁剪 1.0；梯度检查点开启，无 EMA。
- 每 500 步及末步对完整 10 条验证轨迹的所有窗口计算 loss，按有效动作元素加权汇总；最小验证 loss 选 best。训练累积同样按整个有效 batch 的有效元素总量加权。

## 数据与推理

固定图像顺序为 `cam_high, cam_wrist`，触觉组追加 `tac_left, tac_right`（原始左右 `rgb_marker`）。四路共享 SigLIP／LoRA。适配器直接传递显式图像列表及全真 mask，不经过默认三图像过滤。沿用 official50 保存的通道顺序（原 cv2 解码顺序，不额外 RGB/BGR 交换），OpenPI PIL bilinear resize/pad 到 224×224，uint8 映射 [-1,1]，无随机增强。

state 为 8 维 qpos，先对归一化后的 8 维状态使用官方 π0.5 token 化，再将数值状态补零至 32，与官方 transform 顺序一致。固定英文指令见 `common.py`，同任务两模态相同。action[t] 是已有数据中的 qpos(t+1)[:8]；50 步 chunk 的前 7 维均减去 chunk 起始观测 qpos，夹爪维绝对值。q01/q99 仅从 40 条训练轨迹统计；动作统计包含所有窗口的有效未来动作（同一未来动作可在不同起点窗口出现）。按 OpenPI quantile 规则归一化、不额外裁剪；8 维外和尾部填零并屏蔽 loss。两模态共用同一份统计。此前 ACT／DP 使用全部 50 条的统计，本实验统计口径不同。

`inference.Pi05Controller` 重建基础模型+微调参数，默认 10 步 Euler flow integration，逆归一化并还原关节绝对目标。`act()` 每次只执行第一个动作，调用 `env.take_action(action, action_type='qpos')`；调用方随后读取新观测。RGB 模型无需触觉字段。离线验收不创建仿真环境，不使用正式测试 seed。

## 启动命令

在 UniVTAC 根目录执行（验收通过后才允许启动）：

```bash
scripts/pi05_official50/train_insert_HDMI_rgb.sh --gpu 0
scripts/pi05_official50/train_insert_HDMI_rgb_tactile.sh --gpu 0
scripts/pi05_official50/train_insert_hole_rgb.sh --gpu 0
scripts/pi05_official50/train_insert_hole_rgb_tactile.sh --gpu 0
scripts/pi05_official50/train_pull_out_key_rgb.sh --gpu 0
scripts/pi05_official50/train_pull_out_key_rgb_tactile.sh --gpu 0
```

每个入口只接受 `--gpu`、`--output` 和 `--resume`；不接受 seed 或训练配方覆盖。默认输出 `outputs/pi05_official50/runs/<task>_<mode>/`。已有产物禁止意外覆盖，恢复需明确 `--resume`。完成组不自动重新训练。

五槽持久队列：

```bash
nohup .venv-pi05/bin/python -u scripts/pi05_official50/campaign.py --gpu 0 > outputs/pi05_official50/queue.log 2>&1 &
.venv-pi05/bin/python scripts/pi05_official50/progress.py
```

队列遇到数值异常、配置错误、空间不足或子进程失败即停止，不自动重试或越过失败组。当前并发为用户确认的 5 组；任一任务失败后不再启动等待组，已运行组可继续完成。

## 输出与恢复

共享基础权重只保留一份：`outputs/pi05_official50/base/model.safetensors`。`base/manifest.json` 记录 SHA-256，`setup/base_source.json` 锁定官方每个对象的 generation 和 CRC32C（普通对象另校验 MD5，组合对象不提供 MD5）；`setup/base_download_checksums.json` 另记录各文件 SHA-256；转换使用官方转换器，先 FP32 载入，再将 VLM 转 BF16，保留动作专家原始 FP32 精度。

各组保存 `best.pt`、`last.pt`（仅 LoRA＋动作专家＋投影／时间层）、`stats.json`、`config.json`、`parameter_audit.json`、`versions.json`、`metrics.jsonl`、`status.json`。每 100 步及验证时保存完整 `recovery.pt`；它是恢复的权威状态，不依赖可能因中断未写完的 last。完成 4,000 步且最终检查点保存后只移除该组的 recovery，保留 best/last。写入使用临时文件和原子替换，保存前核算额外空间并保留至少 8 GiB；不删除旧实验。

独立入口环境 `.venv-pi05` 安装固定 Transformers 4.53.2 和 tokenizers 0.21.4，应用 OpenPI Transformers 补丁；只读复用本机 cu128 PyTorch 运行库和 OpenPI 依赖，不覆盖现有仿真环境。完整依赖记录在 `setup/requirements.lock.txt`，运行时记录硬件、驱动、代码提交和脚本哈希。固定随机 seed 不承诺跨硬件／驱动逐位一致。

## 验收记录

- `setup/data_audit.json`：六组划分／数据／掩码／动作可逆检查。
- `setup/benchmark_*.json`：真实显存和吞吐测量。
- `acceptance/<mode>/parameters.json`、`gradients_*.json`、`progress.json`、`report.json`：参数精度、梯度、至少 20 次有效 batch=32 的短程更新、冻结权重哈希、左右触觉特征影响、10 步离线动作、seed42 可重复、重载预测和完整恢复检查。
- `acceptance.json`：全部通过后才生成成功门禁。

当前训练运行／排队状态以进度工具及 `queue.json` 为准。这里没有正式仿真成功率。

## 本轮验收及启动记录

两种模态均已通过完整离线验收，六组持久训练队列已启动。当前选定 micro-batch=8、accumulation=4，并发=4；以下 4.09／7.49 秒为先前 micro=4 的历史测量。RGB／RGB+触觉完整验收 allocated 峰值约 15.82 GiB，单组独占 GPU 测试中，每个有效 batch=32 的优化平均分别为 4.09／7.49 秒。完整验收包含额外的推理和恢复分配。

基础权重 SHA-256：`5e35f71ee55e407da16b6e4d1f836d5419b5ece60748fbca4412edafd1fbed58`。详见 `outputs/pi05_official50/acceptance.json`、`memory_report.json` 和 `queue.json`。首次 last／恢复文件在第 100 步生成，首次 best 在第 500 步完整验证后生成。

本轮重启只改变调度并发及检查点写入互斥，模型与优化配方不变。共享 `checkpoint.lock` 将四组的大文件保存串行化；重新核算四份活动恢复状态和临时原子写入的最坏新增空间约 46.65 GiB，启动时约 61.42 GiB 可用，满足 8 GiB 余量。调度已通过六任务四槽补位及失败阻止新任务启动测试。

最新 micro-batch 实测完整表格：[测试报告](../../outputs/pi05_official50/micro_batch_benchmark/README.md)。上述耗时为单进程测试，并非四组并行速度。

## 当前正式配方：8×4，四组并行

两种模态均已在 micro=8 下完成 20 步优化，以及重载预测、下一步恢复更新的逐位一致性检查；新验收报告已绑定脚本哈希。正式入口统一使用 micro-batch=8、accumulation=4、effective batch=32、seed=42、4,000 步。队列先运行 HDMI／insert_hole 四组，再补入拔钥匙两组。

仅用于独占 GPU 基准的每次更新平均耗时为 RGB 3.21 秒、RGB+触觉 5.99 秒；实际四组并行耗时应查看各组 metrics.jsonl，不将此数值作为并行速度。

## 五组并行追加记录

当前调度由 `campaign_five.py` 管理：接管原四个训练 PID，额外启动 `pull_out_key_rgb_tactile`，只保留 `pull_out_key_rgb` 等待空位。原四组未中断或重启。详见 `outputs/pi05_official50/setup/five_slot_plan.json`，其中记录用户授权、接管 PID、控制器哈希和包含五份恢复文件的磁盘预算。按保守总预算计算剩余约 9.60 GiB，高于 8 GiB 下限；检查点写入仍使用全局互斥锁。

## 4,000 步重启记录

本次正式训练均为全新初始化，不从此前短跑检查点继续。模型、数据和 micro=8 的验收结果复用；新增 `check_schedule.py` 验证第 1／200／4,000 步学习率、warmup 后单调衰减、越界拒绝及优化器恢复一致性。`train.py` 的终点、最终验证、状态显示和学习率调度均读取同一份 RECIPE，避免只改循环步数。现由 `campaign.py` 的五槽队列管理，不再接管已停止的旧 PID。详见 `setup/restart_4000_audit.json`、`schedule_4000_audit.json`、`eta_4000.json`。
