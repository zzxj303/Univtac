> 仓库说明：本文同步自本地 `gujie/README-six-task-rsi.md`。UniVTAC 内部链接已改为仓库相对路径；`lab/`、LIBERO 及 `outputs/` 等链接引用本地工作区或未提交的实验产物。训练数据、基础权重、微调权重、运行日志和虚拟环境不随 Git 上传。运行状态是同步时的快照。

# LIBERO／UniVTAC 六任务实验与后续 RSI 思路

> 最新状态：按用户要求已停止并清理上一轮 π0.5 训练输出，从基础权重重新启动 4,000 步实验。warmup=200，cosine 在第 4,000 步降至峰值 10%；micro-batch=8、累积=4、seed=42 不变。当前先并行 HDMI／insert_hole 的 RGB、RGB＋触觉及 pull_out_key_rgb_tactile 五组，pull_out_key_rgb 等待补位。当前五组预计约 30–36 小时，六组整体保守预留 1.5–2 天；尚未启动仿真评测。








本文记录当前实验思路。先完成固定任务、固定数据和固定评测条件下的策略对照，再实现由 agent 根据实验结果诊断问题、提出改进并验证的 RSI 流程。

2026-11-13 结果更新：UniVTAC 的 official50 实验已完成三任务 × ACT／DP × RGB／RGB＋触觉共 12 组训练与评测，全部 360 回合结束。合计 47 次成功、292 次任务失败、21 次初始化异常；异常保留在分母中，未补测或重试。最终逐组结果见下节；下方 30 条示范 ACT 实验为历史记录。

2026-11-11 历史结果记录：LIBERO 三任务已各完成 30 条合格物理回放；ACT／DP 的 12 组训练及评测、官方 π0.5 权重的 6 组基线评测已有结果，合计 540 个评测回合，成功率见下文。π0.5 本地微调尚无正式评测结果；UniVTAC 官方示范已下载，ACT 六组训练及单回合流程验收已完成，当时的 30-seed 正式评测尚在进行；当前 official50 最终结果见下节。数据采集历史见 [实施报告](lab/TacHarness/docs/research/2026-11-09-six-task-replay.md)。

2026-11-12 方案更新：第二个 benchmark 确定为 [UniVTAC](https://github.com/univtac/UniVTAC)。已克隆至 [UniVTAC/]()，使用 `isaac51` 分支，固定 commit `d541e5568227ca3b66104d294f63c80acad7c52c`。已完成仓库阅读、任务选型及下述官方示范下载；已安装仿真环境，官方 ACT 六组训练已全部完成，每组 4,000 步；六组完整回合验收已完成，当时的 30-seed 正式评测尚在进行；当前 official50 最终结果见下节。

## UniVTAC π0.5 official50 混合微调（seed=42，六组队列已启动）

已完成本地 OpenPI π0.5 接入及离线验收，启动 `insert_HDMI`、`insert_hole`、`pull_out_key` × RGB／RGB＋触觉六组持久训练队列。沿用 episode 0–49 的既有 40／10 划分（划分 seed=1），每组训练 seed 固定 **42**、优化器更新 **4,000 步**。六组均独立从官方 `pi05_base` 开始，基础权重 SHA-256 为 `5e35f71ee55e407da16b6e4d1f836d5419b5ece60748fbca4412edafd1fbed58`；不是 UniVTAC 官方发布的已微调策略权重。

VLM 原始权重冻结，参与动作前向的 SigLIP／Gemma attention、MLP 和视觉投影使用 LoRA（r=16、alpha=32、dropout=0），VLM 参数全部 BF16；动作专家、AdaRMS、动作投影和时间 MLP 使用 FP32 主参数全参微调、BF16 autocast。有效 batch=32；当前采用 micro-batch=8、梯度累积=4。AdamW 的 LoRA／专家峰值 LR=1e-4／5e-5，200 步 warmup 后 cosine 至 10%；开启梯度检查点，不使用 EMA。

两路／四路图像顺序固定为 head、wrist、左 rgb_marker、右 rgb_marker，共享视觉编码器。保留既有通道顺序，224×224 resize/pad，无随机增强。动作 horizon=50，前 7 维相对当前 qpos、夹爪绝对值；32 维接口仅前 8 维及有效时间位置计入 loss。归一化只统计 40 条训练轨迹，两模态共享，与此前 ACT／DP 的全 50 条统计口径不同。推理使用 10 步 flow integration，恢复绝对关节目标，每次只执行首动作后重新观测。

两种模态分别完成 20 次有效 batch=32 的短程优化，并通过全部参数精度／梯度／更新检查、冻结权重哈希检查、三任务有限动作检查、左右触觉特征影响、seed-42 重复初始化、权重重载和下一步断点恢复逐位一致性检查。验证随机流独立；不同长度的独立示例确认有效元素加权及 padding 屏蔽正确。验收用验证子集做随机性检查，正式训练每 500 步及结束时使用完整 10 条验证轨迹选 best。

| 输入 | 单组短程每次优化平均耗时（batch=32） | 完整验收 allocated 峰值 | 完整验收 reserved 峰值 |
| --- | ---: | ---: | ---: |
| rgb | 3.21 秒 | 16.08 GiB | 18.01 GiB |
| rgb_tactile | 5.99 秒 | 16.01 GiB | 17.72 GiB |

上表耗时来自 micro=8 的单进程独占 GPU 基准；显存峰值来自对应完整验收，含离线推理／重载／恢复测试，不含全部驱动进程开销。当前按用户追加请求采用五组并发；显存与吞吐以实时记录为准。启动时可用磁盘 61.42 GiB，六组 best／last、活动恢复及原子写入的保守新增预算 46.65 GiB，满足至少 8 GiB 余量。基础权重共享一份，已清理本轮临时 JAX 下载，未删除旧实验。

队列先并行运行 `insert_HDMI_rgb`、`insert_HDMI_rgb_tactile`、`insert_hole_rgb`、`insert_hole_rgb_tactile`，拔钥匙两组等待空位；最新状态以 [queue.json](outputs/pi05_official50/queue.json) 及各组 `status.json` 为准。每组保存 best／last、统计量、配置、参数与版本审计；活动组保存完整随机状态和优化器恢复文件，完成后移除该组滚动恢复文件。错误或磁盘余量不足会停止后续启动。

- 六个固定入口、详细配方、恢复约定：[训练 README](scripts/pi05_official50/README.md)。
- 验收与资源报告：[acceptance.json](outputs/pi05_official50/acceptance.json)、[memory_report.json](outputs/pi05_official50/memory_report.json)。
- 微调产物：[runs/](outputs/pi05_official50/runs/)。
- 进度命令（在 UniVTAC 根目录）：`.venv-pi05/bin/python scripts/pi05_official50/progress.py`。

**本轮只启动微调，没有启动正式仿真评测，尚无这六组模型的仿真成功率。训练／验证 loss 不作为成功率。**

## 50 条示范 ACT／DP 实验（2026-11-13，训练与评测全部完成）

用户新增实验：`insert_HDMI`、`insert_hole`、`pull_out_key`，每任务选官方 episode 0–49，固定 40 条训练／10 条验证；ACT／DP × RGB／RGB＋触觉共 12 组，共享轨迹与划分。12 组训练已全部完成，训练期间最高并发 12 组，完成记录见 [status.json](outputs/official50/status.json)。本节不替代下方历史 30 条示范实验结果。

ACT 调用官方 `imitate_episodes.py`，采用官方双相机配置，batch 64、学习率 1e-5、chunk 50、4000 步预算（原循环实际执行 4001 次更新）。DP 经用户确认采用官方支持的 ResNet18 从头训练，保留 batch 8、学习率 1e-4、预测长度 20、3500 epochs（40 条训练轨迹对应 17,500 次更新）；使用仓库内 ViTAL diffusion 的网络和训练循环，适配双 RGB／双触觉 RGB、8 维状态动作，并修复验证后未切回训练模式的问题。因此 DP 应标为“上游实现经 UniVTAC 接口适配”，不能声称原始脚本无需修改即可运行。归一化统计量按官方行为使用全部 50 条轨迹，包含验证集。

显存短程实测峰值（PyTorch reserved）：ACT RGB 7.84 GiB、ACT RGB＋触觉 9.82 GiB、DP RGB 2.18 GiB、DP RGB＋触觉 3.03 GiB；不包含全部进程开销。已完成真实数据训练／验证试跑、DP 权重重载与完整 100 步去噪检查；正式评测采用下述 10 次去噪设置，现已全部完成。

入口与完整适配说明：[scripts/official50/README.md](scripts/official50/README.md)。运行 `.venv/bin/python scripts/official50/status.py` 查看训练进展。结果保存至 [outputs/official50/](outputs/official50/)，与历史实验隔离。

### official50 正式评测结果（2026-11-13 核对）

已为三任务 × ACT／DP × RGB／RGB＋触觉准备 12 个独立入口，固定使用各组 `policy_best.ckpt`，每组 seed `20001–20030` 各一次；现已完成全部 360 回合。已核验这批 seed 与训练及验证共 50 条示范的 `seed`、`source_seed` 均无交集。不进行专家筛选、不重试或替换失败 seed；异常和中断保留并计入固定分母 30。

ACT 沿用官方时序聚合；DP 使用本轮模型、10 次 DDPM 去噪、预测长度 20，每次执行首个动作后重新观测。保留原生 `clean51` 和任务步数上限。离线权重加载、观测预处理一致性、动作输出及调度逻辑验收已通过。正式评测最多并发六组，十二组均已完成，每组 30 条唯一 seed 结果与 summary.json 的计数一致，pending 均为 0。HDMI 保留 600 次动作上限，斜孔和拔钥匙保留 300 次；策略从原生 reset/pre_move 准备完成后接管，不是从桌面抓取开始的全流程评测。

单元格为「成功率（成功回合数／30）」；点击结果可查看对应组的原始汇总。所有运行异常均计入固定分母。

| 任务 | ACT RGB | ACT RGB＋触觉 | DP RGB | DP RGB＋触觉 |
| --- | ---: | ---: | ---: | ---: |
| HDMI 插接 | [3.3%（1/30）](outputs/official50/evaluation/act_insert_HDMI_rgb/summary.json) | [3.3%（1/30）](outputs/official50/evaluation/act_insert_HDMI_rgb_tactile/summary.json) | [6.7%（2/30）](outputs/official50/evaluation/dp_insert_HDMI_rgb/summary.json) | [6.7%（2/30）](outputs/official50/evaluation/dp_insert_HDMI_rgb_tactile/summary.json) |
| 斜孔插入 | [16.7%（5/30）](outputs/official50/evaluation/act_insert_hole_rgb/summary.json) | [30.0%（9/30）](outputs/official50/evaluation/act_insert_hole_rgb_tactile/summary.json) | [0.0%（0/30）](outputs/official50/evaluation/dp_insert_hole_rgb/summary.json) | [16.7%（5/30）](outputs/official50/evaluation/dp_insert_hole_rgb_tactile/summary.json) |
| 旋转并拔出钥匙 | [26.7%（8/30）](outputs/official50/evaluation/act_pull_out_key_rgb/summary.json) | [33.3%（10/30）](outputs/official50/evaluation/act_pull_out_key_rgb_tactile/summary.json) | [10.0%（3/30）](outputs/official50/evaluation/dp_pull_out_key_rgb/summary.json) | [3.3%（1/30）](outputs/official50/evaluation/dp_pull_out_key_rgb_tactile/summary.json) |

**总计：47 次成功、292 次任务失败、21 次运行异常，共 360 回合。** 未出现 1800 秒整回合超时或中断记录；初始化超时归类为 error，不是调度器的 timeout。

异常次数单独列出，避免将环境初始化失败误认为策略已执行但未完成任务：

| 任务 | ACT RGB 异常 | ACT RGB＋触觉异常 | DP RGB 异常 | DP RGB＋触觉异常 |
| --- | ---: | ---: | ---: | ---: |
| HDMI 插接 | 3 | 4 | 3 | 2 |
| 斜孔插入 | 1 | 1 | 1 | 1 |
| 旋转并拔出钥匙 | 1 | 3 | 0 | 1 |

21 次异常全部发生在初始化阶段：19 次触发原生 reset 的 120 秒墙钟上限，2 次在等待触觉 marker 附着稳定时触发初始化超时。未移除这些回合、修改成功条件或重跑 seed。因此上表反映本轮含初始化异常的完整评测结果，不能仅归因于策略能力；并行资源竞争对初始化耗时的影响尚未单独隔离验证。

原始证据：[逐回合结果、动作／状态轨迹与视频](outputs/official50/evaluation/)、[评测调度完成状态](outputs/official50/evaluation_launch/status.json)。各组 manifest.json 保存权重与配置哈希及 seed 排除记录；每个 seed 目录的 result.json／worker_result.json 保留结果和异常详情。

十二条命令及记录规则见 [评测 README](scripts/official50/evaluation/README.md)。离线报告见 [report.json](outputs/official50/evaluation_preflight/report.json)。

## 第一阶段：六任务对照实验

### 任务与示范数据

LIBERO 三任务沿用原名单；UniVTAC 三任务于 2026-11-12 根据任务源码选定：

| Benchmark | 任务 | 准确标识 | 数据现状 |
| --- | --- | --- | --- |
| LIBERO | 碗放到炉子上 | `libero_goal:1` / `put_the_bowl_on_the_stove` | 源 50 条；本轮已验收 30 条双 RGB／双触觉回放 |
| LIBERO | 打开柜子中间抽屉 | `libero_goal:0` / `open_the_middle_drawer_of_the_cabinet` | 源 50 条；本轮已验收 30 条双 RGB／双触觉回放 |
| LIBERO | 将盘子推到炉子前面 | `libero_goal:5` / `push_the_plate_to_the_front_of_the_stove` | 源 50 条；本轮已验收 30 条双 RGB／双触觉回放 |
| UniVTAC | HDMI 插接 | `insert_HDMI` | 已下载 `isaac51` 官方示范 100 条；哈希与 HDF5 结构检查通过，物理回放待验收 |
| UniVTAC | 斜孔插入 | `insert_hole` | 已下载 `isaac51` 官方示范 100 条；哈希与 HDF5 结构检查通过，物理回放待验收 |
| UniVTAC | 旋转并拔出钥匙 | `pull_out_key` | 已下载 `isaac51` 官方示范 100 条；哈希与 HDF5 结构检查通过，物理回放待验收 |

这六个任务按单臂配置开展实验。左右触觉指同一个夹爪两侧的传感器。UniVTAC 使用原生 Franka 与双 GelSight Mini（`gsmini`），保留 TacEx／UIPC 接触物理；不把已有 LIBERO 的机器人或触觉配置直接套到 UniVTAC 上。

每个任务收集 30 条独立、完整、验收合格的示范，共计 180 条。这是目标数量，不表示数据已经齐备。LIBERO 沿用源动作物理 replay；UniVTAC 优先核验对应版本的官方示范，必要时使用原生专家脚本和 cuRobo 补采。完整性按原生策略执行阶段判断，具体边界见下文。

#### LIBERO 源示范

源文件位于 `lab/TacHarness/inputs/datasets/vendor/tabero/Isaaclab_Libero/assembled_hdf5/`（相对于本文目录）：

- Bowl：`libero_goal_task1_put_the_bowl_on_the_stove_demo.hdf5`
- 抽屉：`libero_goal_task0_open_the_middle_drawer_of_the_cabinet_demo.hdf5`
- 推盘子：`libero_goal_task5_push_the_plate_to_the_front_of_the_stove_demo.hdf5`

已读取三个文件，各有 50 条轨迹；抽查的 episode 包含 `initial_state` 和 `actions`。这些是源动作回放数据，不能直接算作已经采齐双相机和真实仿真触觉的训练示范。需从各任务中筛选并成功重录 30 条。

#### UniVTAC 任务选择与原生成功条件

选择依据是接触精度、姿态调整和动作阶段衔接，而非已测得的成功率排名。三个任务分别覆盖精细插接、斜孔接触调整、旋转后拔出；相较单纯提起罐子等任务，更适合研究触觉能否补充视觉。这是基于源码的难度判断，尚未进行本地难度排序实验。

| 任务与源码 | 难点和初始化变化 | 原生 `check_success()` 条件摘要 |
| --- | --- | --- |
| [HDMI 插接](envs/insert_HDMI.py) | 插座 XY 位置扰动为 ±5 mm；抓取姿态有 ±10° 变化，预插接位置另有 ±5 mm 扰动。需要在抓取偏差和近距离遮挡下完成对齐与插入 | 相对 `target_pose` 的 X、Y 绝对误差均 <5 mm，Z <5 mm；末端世界高度 >0.145 m；连接器相对 Z 轴点积 >0.965 |
| [斜孔插入](envs/insert_hole.py) | 孔座 XY 位置扰动为 ±2／±3 cm，朝向在 0／π 中切换；目标倾角随之为 ∓30°。需接触后调整试管姿态并沿斜孔深入，同时避免夹持滑移 | 相对 `target_pose` 的 X、Y 绝对误差均 <1 cm，Z <−4 cm；相对 Z 轴点积 >0.99；相对初始夹持位置的轴向滑移 <4 cm |
| [旋转并拔出钥匙](envs/pull_out_key.py) | 锁座 XY 位置扰动为 ±5 mm、绕 Z 轴旋转 ±45°；钥匙初始相对角在 −90° 至 −45°。专家先旋转、回调，再拔出，要求维持夹持并避免卡住 | 钥匙世界高度 >0.09 m，Z 轴与世界竖直方向点积 >0.965；锁座相对初始 X 轴点积 >0.99；末端与钥匙高度差 <0.14 m |

上述为阅读代码后的摘要，判定以固定 commit 的 `check_success()`、`check_early_stop()` 和基础任务执行逻辑为准，不额外放宽阈值。HDMI 的 `step_lim=600`，另外两项继承基础配置的 `step_lim=300`；这是 `take_action` 调用计数上限，不能直接套用 LIBERO 的 400 步设置。

**评测起点：** [基础任务](envs/_base_task.py) 的 `reset()` 会执行 `pre_move()`。三项任务的抓取和预定位由脚本完成，策略随后接管接触操作阶段；采集默认不保存 `pre_move` 阶段。报告须明确这是原生准备状态后的策略评测，不能称为从桌面抓取开始的全流程自主操作。若未来扩展为全流程任务，应另行记录为修改后的任务设置。

#### UniVTAC 环境与示范数据

本轮固定 `isaac51`，对应 Isaac Sim 5.1.0、Isaac Lab 2.3.0、Python 3.11、PyTorch 2.7.0；上游脚本使用 CUDA 12.6，本机 RTX PRO 6000 Blackwell 改用 CUDA 12.8 并以 `sm_120` 编译 cuRobo／UIPC。安装依据 [Installation.md](docs/Installation.md)，使用仓库内修改版 TacEx／libuipc 和 cuRobo。`main` 对应 Isaac Sim 4.5，两个版本的数据不能混用。仓库根 README 的 Installation 段误写了 `main`，这里以分支说明和安装文档为准。

[官方仓库 README](README.md) 声明在 [Hugging Face](https://huggingface.co/datasets/byml/UniVTAC) 和 [ModelScope](https://modelscope.cn/datasets/byml2024/UniVTAC) 提供每个任务 100 条 HDF5 示范及 `metadata.json`。本轮已下载三个任务的全部 300 条 HDF5 示范及 3 份元数据，并完成官方哈希和 HDF5 结构核验。文件完整性检查不等于物理回放及训练数据验收，ACT 已固定选择每任务 episode 0–29，按 24／6 划分训练和验证；本地逐条物理回放验收仍未完成。数据应位于 `UniVTAC/data/isaac51/<task>/hdf5/`。已发布 checkpoint 对应 Isaac Sim 4.5，不直接作为本轮 5.1 的匹配权重。

**2026-11-12 全量下载记录：** 固定 Hugging Face 数据 revision 为 `e1aee7b0c95543b535e0146b2de3ee1bc6ddaabd`，从官方 Hugging Face 与 ModelScope 来源下载，所有文件均按该 revision 的官方哈希验证。共 303 个文件、69,421,824,062 字节（约 69.42 GB／64.65 GiB），无未完成分片。

| 任务 | 本地目录 | 轨迹数 | 总帧数 | 单条帧数范围 |
| --- | --- | ---: | ---: | ---: |
| `insert_HDMI` | [data/isaac51/insert_HDMI/](data/isaac51/insert_HDMI/) | 100 | 11,803 | 118–121 |
| `insert_hole` | [data/isaac51/insert_hole/](data/isaac51/insert_hole/) | 100 | 19,151 | 157–245 |
| `pull_out_key` | [data/isaac51/pull_out_key/](data/isaac51/pull_out_key/) | 100 | 20,129 | 175–223 |

三份 `metadata.json` 各含 100 条 `success` 记录和 100 个不同的 `source_seed`；`seed` 与 `source_seed` 均保留，不能混用。全部 HDF5 可打开，外部／腕部 RGB、左右触觉及机器人状态字段齐全，各 dataset 的首维帧数一致，已读取每个 dataset 的首尾记录。这是文件与结构检查，尚未在本地物理仿真中复现成功。

核验记录：[官方文件清单](data/download_audit/manifest.json)、[逐文件哈希结果](data/download_audit/summary.json)、[HDF5 与元数据检查](data/download_audit/hdf5_validation.json)。

上游安装、下载命令（本机环境已安装；Blackwell 环境使用下文的本地入口，不应直接覆盖重装）：

```bash
cd /root/autodl-tmp/gujie/UniVTAC
bash scripts/install.sh
conda activate UniVTAC
bash data/download.sh --task insert_HDMI --task insert_hole --task pull_out_key --version 51
```

下载脚本默认使用 ModelScope，需要 `modelscope` 命令。已记录数据 revision、文件哈希、episode／seed 身份、官方成功标记及传感器字段；ACT 的每任务 30 条示范和 24／6 划分已固定，详细设置见下文。先确认官方数据的控制频率、物理配置、初始条件和动作语义与本轮配置匹配；不能只因为目录名是 `isaac51` 就认定可以无损物理回放。

不足时采用 [原生采集流程](docs/Collection.md) 补采。例如，在独立输出目录为一项任务生成 30 条成功示范：

```bash
python scripts/collect_data.py insert_HDMI clean51 --headless \
  --config-overrides collect_settings.episode_num=30 \
                     collect_settings.save_root_dir=data/six_task_native
```

另外两项使用同样入口并替换任务名。`clean51.yml` 使用 `gsmini`／`taxim`，物理和采集控制频率 120 Hz、保存和评测频率 60 Hz；实际实验须保存配置快照。原生采集器按 seed 运行专家，只将成功回合保存为示范，并用 `suc_map.txt` 跟踪进度；失败诊断信息须另行保留。重新采集和回放要分别标记，不重复回放同一源 episode 凑独立示范数。

[数据格式说明](docs/Collection.md) 包含 `observation/head/rgb`、`observation/wrist/rgb`、左右 `tactile` 和 `embodiment/joint`、`embodiment/ee`。`press_depth` 表示正向压入深度，旧字段 `depth` 是相机距离，两者不可混用。HDF5 中的机器人状态不直接等于策略控制命令；转换时须核对动作目标、状态与图像的时间对齐，以及关节／末端坐标约定。物体真值只用于诊断和成功判定，不作为主实验策略输入。

这里的 replay 是恢复示范初始条件、执行源动作并推进物理仿真，同时重新记录观测与动作。逐帧强制回填状态的播放不能替代物理执行，也不能据此生成有效触觉示范。

每条轨迹保留外部相机、腕部相机、左右真实仿真触觉、机器人状态、实际发送的控制命令、执行后状态、时间戳、任务指令和原生成功记录。记录源轨迹身份、初始条件、控制器配置和传感器配置，便于复现与排查。

每个任务的目标是获得 30 条验收合格的示范，失败或不完整的尝试另行保留，不混入成功示范。完整记录作为共同数据源，再按 ACT、DP 和 π0.5 的输入及动作约定分别导出。保存成不同格式时必须核对动作语义、时序与 episode 边界，不能只改字段名。

### 策略与模态

每个任务分别训练或微调以下策略：

| 策略 | RGB | RGB＋触觉 |
| --- | --- | --- |
| ACT | 训练对应任务模型 | 训练对应任务模型 |
| DP（Diffusion Policy） | 训练对应任务模型 | 训练对应任务模型 |
| π0.5 | 基于选定预训练权重微调 | 基于选定预训练权重扩展触觉输入并微调 |

RGB 固定包含外部相机和腕部相机。RGB＋触觉在同样的 RGB 输入上增加左右触觉。状态和指令等其他输入在同一策略的两种模态之间保持一致。

π0.5：LIBERO 沿用官方 `pi05_libero`；UniVTAC 仓库未提供可直接沿用的 π0.5 任务权重，本轮已选定官方 `pi05_base`，完成 Franka 状态、动作与双相机／双触觉的本地适配及离线验收，并启动上述六组微调。不能把其他 benchmark 的权重配置写成 UniVTAC 官方基线。

UniVTAC 提供 [ACT](policy/ACT/) 等策略实现与[策略部署接口](docs/Deploy.md)，本轮 DP 接入与评测已经完成，π0.5 本地接入及离线验收已经完成，六组微调队列已启动。原生 ACT 数据处理会按任务选择相机；本实验要求显式核对导出的输入包含 `head` 和 `wrist` 两路，不能直接沿用单视角默认值。三种策略均须对接 `task.take_action()`，核对动作维度、夹爪语义、坐标系与控制时序。

UniVTAC ACT 使用官方视觉／触觉编码器及 Transformer 融合；本轮 UniVTAC π0.5 使用四路共享 SigLIP／LoRA 的图像融合。此前 LIBERO 的 π0.5 接口支持额外图像视角 `image_concat` 和专用编码器 `encoder`；本计划暂将 RGB＋触觉作为一组实验，不自动把两种融合方式都加入主实验矩阵。

同一任务尽量共用同一批示范和训练／验证划分，训练与评测使用匹配的物理配置。RGB 对照仍使用相同机器人及传感器安装配置，只关闭策略的触觉输入，避免把机器人变化混入模态对比。

### 脚本训练与评测

通过脚本完成数据检查、格式转换、训练或微调、checkpoint 保存以及评测。模型必须绑定所用数据、统计量、模态和动作定义。

每个任务、每种策略、每种模态均评测 30 个 seed。同一任务的各实验组使用相同的 30 个测试 seed 和初始化规则，保存视频、逐帧轨迹、原生成功判定与异常记录。训练／验证数据与最终测试开局隔离，checkpoint 选择依据验证结果。

UniVTAC 评测沿用原生 `pre_move`、成功／提前终止逻辑和任务上限。上游 [eval_policy.py](scripts/eval_policy.py) 可以通过专家检查筛除不可执行 seed，并在异常时减少已测试计数；正式对照须提前固定三种策略共享的 30 个有效测试 seed、专家筛选记录和异常处理规则，避免不同模型实际评测不同开局。异常必须完整报告，不可默默丢弃后只报成功率。

按每组只训练一版模型计算，首轮规模为：

| 项目 | 数量 |
| --- | ---: |
| 任务 | 2 个 benchmark × 3 个任务 = 6 |
| 合格示范 | 6 × 30 = 180 条 |
| 策略／模态组合 | 6 × 3 种策略 × 2 种模态 = 36 组 |
| 测试回合 | 36 × 30 个 seed = 1080 回合 |

上述数量不包含失败采集重试、接口诊断、多训练随机种子或额外融合方式消融。已完成的 LIBERO 实验采用下节所列训练预算和评测设置；其余实验配置仍待确定。此前单独 Bowl 实验的预算不自动沿用到六任务实验。

### UniVTAC 官方 ACT 本地运行（历史 30 条示范实验，2026-11-12 记录）

使用仓库官方 `ACTPolicy`、`TacArenaDataset` 与 ACT 时序聚合。每任务选官方 episode `0–29`，随机种子 42 固定划分为 24 条训练、6 条验证；这批官方成功示范已通过文件检查，但尚未逐条完成本地物理回放验收。两种模态共享划分，均输入 head／wrist RGB 和 8 维关节状态，触觉组增加左右 `rgb_marker`。图像缩放到 256×256，动作沿用官方转换约定 `qpos(t+1)[:8]`，统计量仅由训练集计算。

每组训练 4,000 个优化步，batch size 64，每 500 步完整验证并保存验证 loss 最低的 checkpoint。触觉编码器使用官方共享 `encoder.pth`；没有加载 Isaac Sim 4.5 的任务策略权重。此预算用于先跑通官方 ACT 基线，不代表已做收敛或最优超参研究。

本机入口为 [scripts/act_six/](scripts/act_six/)。`.venv` 复用已有 Isaac Sim 5.1／PyTorch 2.7 cu128 的二进制包，并在 UniVTAC 环境内安装 Isaac Lab 2.3、TacEx、cuRobo 和 UIPC。运行依赖该共享运行时路径，尚不是可移动的独立环境。`pip check`、GPU scatter、cuRobo 导入和带相机的 Isaac 启动检查已通过。

```bash
cd /root/autodl-tmp/gujie/UniVTAC
source scripts/act_six/env.sh
python scripts/act_six/status.py
# 新实验可使用独立 --name，避免覆盖当前结果
python scripts/act_six/train.py insert_HDMI rgb --steps 4000 --name another_run
```

正式评测计划：每任务从 seed 10001 起用原生专家预筛选 30 个成功开局，两种模态共用固定名单；保留全部筛选尝试。策略评测沿用原生任务上限和成功条件，异常计入失败，保存视频及动作／执行后状态轨迹。仅将初始化墙钟超时上限从 120 秒提高至 600 秒，以容纳本机并行任务；仿真时间步及策略任务步数不变。

六组训练已全部完成，本轮最佳 checkpoint 均选于第 4,000 步。以下为验证 loss，不是任务成功率：

| 任务 | ACT RGB 验证 loss | ACT RGB＋触觉验证 loss |
| --- | ---: | ---: |
| `insert_HDMI` | 0.097379 | 0.098451 |
| `insert_hole` | 0.088305 | 0.085214 |
| `pull_out_key` | 0.091280 | 0.093539 |

[验证曲线](outputs/act_six/training_curves.png)。三任务原生专家均已成功运行至少一个回合；正式权重的完整单回合验收记录另存于 [integration](outputs/act_six/evaluation/integration/)，不与 30-seed 正式统计混算。斜孔一次并行初始化异常慢的记录保存在 [hole_slow_initialization](outputs/act_six/setup/hole_slow_initialization/)。

**完整单回合验收：** 六组均用各自正式训练的最佳权重，执行完整原生策略阶段，未设置诊断步数截断；同任务两种模态使用相同 seed。六组均完成且没有运行异常。以下仅是接口与流程验收，不是 30-seed 成功率估计。

| 任务（验收 seed） | RGB | RGB＋触觉 | 原生结果说明 |
| --- | --- | --- | --- |
| `insert_HDMI`（10002） | 成功（1/1） | 成功（1/1） | 均通过原生成功判定 |
| `insert_hole`（10001） | 失败（0/1） | 失败（0/1） | 分别在 190／223 个策略步因夹持轴向滑移超过 4 cm 提前终止 |
| `pull_out_key`（10001） | 失败（0/1） | 失败（0/1） | 均在 300 个策略步上限内未达到成功条件 |

[逐组核验汇总](outputs/act_six/verified_integration.json)已检查 checkpoint 哈希、同任务统计量一致性、seed、轨迹长度、有限数值和成功标记。原始目录包含视频、指令、策略接管时状态、动作／执行后状态及原生诊断。原生物理预定位存在小幅数值差异，配对实验固定的是 seed 和初始化规则，而非逐浮点数相同的接管状态。

历史 30 条示范实验曾计划由 `run_task_eval.sh` 先筛选每任务 30 个专家成功开局，再评测两种模态；本节保留当时尚未完成的记录，不将上述单回合验收当作正式成功率。当前已完成的是上方 official50 实验，使用 50 条示范、固定 seed 20001–20030 且不进行专家筛选，两轮结果不可混算。

运行记录：[实验清单](outputs/act_six/experiment.json)、[训练目录](outputs/act_six/formal/)、[评测目录](outputs/act_six/evaluation/)、[环境安装与诊断日志](outputs/act_six/setup/)。未完成的评测不填成功率。

### LIBERO 脚本评测成功率（2026-11-11 核对）

已完成三任务 × 三种策略 × 两种模态，共 18 组、540 回合的评测。下表按任务横向比较各策略与输入模态，单元格为「成功率（成功回合数／总回合数）」。已逐回合核对成功数、完成数与汇总成功率；原始结果见表后链接。

| 任务 | ACT RGB | ACT RGB＋触觉 | DP RGB | DP RGB＋触觉 | π0.5 RGB（官方基线） | π0.5 RGB＋触觉（官方基线） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 碗放到炉子上 | 100.0%（30/30） | 100.0%（30/30） | 90.0%（27/30） | 100.0%（30/30） | 0.0%（0/30） | 0.0%（0/30） |
| 打开柜子中间抽屉 | 100.0%（30/30） | 100.0%（30/30） | 100.0%（30/30） | 100.0%（30/30） | 3.3%（1/30） | 0.0%（0/30） |
| 将盘子推到炉子前面 | 83.3%（25/30） | 26.7%（8/30） | 43.3%（13/30） | 23.3%（7/30） | 3.3%（1/30） | 0.0%（0/30） |

评测环境为 `Isaac-Libero-Franka-Replay-Camera-Tactile-v0`，不是官方 MuJoCo LIBERO benchmark。每组使用 seed `10001–10030`，每回合最多执行 400 个策略步，原生成功条件连续满足 10 步才计为成功；开局另有 10 步稳定阶段。测试 seed 排除了本地训练／采集 seed `42`、`9173`，与官方 π0.5 上游训练 seed 是否重叠未知。

ACT 和 DP 均完成每组 20,000 步训练，评测使用验证集选出的 `best.ckpt`，不一定是第 20,000 步权重。π0.5 表中使用官方 `pi05_libero` 权重：上游已经过 LIBERO 微调，但未做本地适配；其 RGB＋触觉组加入的是未经训练的额外触觉图像视角。因此，这六组是官方权重在当前 Isaac 环境中的基线，不代表本地触觉微调效果。

原始结果：[ACT／DP 评测目录](lab/TacHarness/outputs/research/six_task_evaluation/formal-20261110T085253Z/)（各组 `attempt-1/summary.json`）；[π0.5 官方基线评测目录](lab/TacHarness/outputs/research/six_task_evaluation/pi05_official/formal/20261110T145214Z/)（各组 `summary.json`）。

训练与评测来源：

- ACT：[训练目录](lab/TacHarness/outputs/research/six_task_training/campaign-20261110-003248/)。
- DP：[训练目录](lab/TacHarness/outputs/research/six_task_training/dp-campaign-v1/)。
- ACT／DP：[正式评测计划](lab/TacHarness/outputs/research/six_task_evaluation/formal-20261110T085253Z/plan.json)，含各组脚本路径与 checkpoint 绑定。
- π0.5：[官方权重评测状态](lab/TacHarness/outputs/research/six_task_evaluation/pi05_official/launch-20261110T145214Z/status.json)及[评测配置示例](lab/TacHarness/outputs/research/six_task_evaluation/pi05_official/formal/20261110T145214Z/bowl_rgb_tactile/evaluation_config.json)。

本地 π0.5 每组 10,000 步的微调另见[训练状态](lab/TacHarness/outputs/research/six_task_training/pi05-finetune10k-after-baseline-20261110T154622Z/status.json)。本次读取时，该文件记录完成 0/6 组，Bowl RGB 仍标为训练中；尚无这批微调模型的正式成功率，不填为 0%。UniVTAC 的 ACT 本地训练与评测另见上节，不纳入上述 LIBERO 结果。

## 第二阶段：由 agent 实现 RSI（后续实现）

这里的 RSI 具体指：agent 能根据实验结果自主诊断问题，提出改进措施，并通过受控实验检查改进是否有效。

预期流程：

1. 读取训练曲线、验证指标、评测轨迹、视频和失败日志。
2. 区分数据、观测／动作接口、控制执行、训练和模型能力方面的问题，给出证据及待验证的解释。
3. 提出具体改进，例如数据筛选与补采、预处理或接口修复、训练参数调整、触觉融合方式调整。
4. 在独立版本和实验目录中实施改进，运行必要的接口检查、训练及诊断评测。
5. 对比改进前后的结果，保留有效修改并记录失败尝试，进入下一轮。

首阶段先建立可靠基线和完整实验记录，不同时开启 RSI 自动迭代。后续还需明确 agent 可修改的范围、算力预算和停止条件。

RSI 使用的诊断／开发评测与最终保留测试分开；若原先的 30 个测试 seed 被 agent 反复用于调整方案，后续就应视为开发集，不能继续当作独立最终测试。不得通过放宽成功条件或丢弃失败回合制造改进。

## 当前实现基础与缺项

- 已有双相机 OpenPI 适配及三种触觉模式入口，见 [运行说明](lab/TacHarness/scripts/vtla/pi05_openpi/README.md)。
- 本轮 LIBERO 三任务均已验收 30 条双 RGB／双触觉回放，共 90 条；此前单独 Bowl 的 27 条数据属于历史实验。
- UniVTAC 已克隆并固定版本，确定 `insert_HDMI`、`insert_hole`、`pull_out_key` 三任务；已阅读任务、采集、数据和评测代码。三个任务各 100 条官方示范已下载，官方哈希和 HDF5 结构核验通过。环境已安装；official50 的 ACT／DP 共 12 组训练与 360 回合正式评测均已完成，DP 使用上游实现的 UniVTAC 接口适配版本。完整示范物理回放验收仍待完成；UniVTAC π0.5 本地适配和离线验收已完成，六组 seed-42 混合微调队列已启动，尚未正式评测。
- 官方 π0.5 基线已使用标定后的增量控制几何与 Isaac 绝对 IK 完成评测；本地微调后的任务成功率仍待验证。
- 六任务名单已确定；六任务数据验收、完整训练配置及 RSI 自动迭代尚未完成。

当前阶段结果与阻断项见 [OpenPI 适配实施报告](lab/TacHarness/docs/research/2026-11-09-pi05-openpi-adaptation.md)。
