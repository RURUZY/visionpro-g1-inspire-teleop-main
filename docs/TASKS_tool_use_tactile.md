# G1 + Inspire 工具使用与触觉：任务梳理（2026-10-17）

本文档整理当前课题的目标、已完成内容、未完成内容和待确认事项。项目总体计划见上级目录 `README.md`（8–12 周实施计划）。

## 1. 目标

两个课题方向，共用同一套平台和数据：

| 方向 | 内容 | 对应文件夹 |
|---|---|---|
| **A. VLA 工具使用（主课题）** | 人形机器人用海绵擦拭指定区域；收集示范数据，训练语言条件模仿学习策略（ACT → LC-ACT，之后可试预训练 VLA），闭环评估 | `act_tools/wipe_teacher`、`act_tools/wipe_block`、`act_tools/eval` |
| **B. 触觉感知（题目 1）** | 用指尖/指腹触觉检测工具在手中的滑移，并调整握力；比较有/无触觉反馈 | `act_tools/tactile_slip` |

B 是 A 的"执行监控 / 触觉反馈"模块，两者共用场景、工具和数据格式。

本科与硕士范围的区别：

| | 本科 | 硕士 |
|---|---|---|
| A | 一个工具：数据集 + 基准 + 现有策略的训练与评估 | 接触感知或分层的手–臂策略；跨工具/任务变体迁移 |
| B | 集成并标定传感器，在一个任务上演示触觉反馈 | 多模态（视觉+触觉+关节）估计或学习型触觉策略；跨物体/接触条件迁移 |

## 2. 已完成（全部为仿真，未经真机验证）

### 2.1 海绵遥操作场景（Vision Pro + SONIC + MuJoCo）

- 海绵流程与抓方块流程统一：终端 1 `bash bin/sonic-sim-sponge.sh`，终端 2 `bash run_sonic_deploy.sh sim`，终端 3 `bash solo_teleop.sh <VP_IP> --print-debug`。说明见 `海绵测试命令.txt`。
- 问题定位：
  - "机器人乱跳"：按 9 落地时从高处掉下并向前冲撞到桌子，跌倒后仿真不断复位；
  - 按 `]` 时手臂甩动会把海绵扫落；
  - Codex 与 Claude 曾同时修改同一批文件。
- `tools/sponge_sim_entry.py`（已接入 `bin/sonic-sim-sponge.sh`）：
  - 悬挂高度降低 0.10 m，实测原地落地，不再前冲；
  - 桌子启动时停在远处，避开 `]` 的手臂甩动和落地；机器人站稳 2 秒后自动推到桌沿 0.30 m，实测落地后离桌子最近 5.9 cm；
  - G 键把海绵放入右手（辅助握持，非接触握持）；
  - R 键把海绵放回桌面。

### 2.2 脚本教师：自动生成 ACT 训练数据（`act_tools/`）

| 版本 | 工具与握法 | 状态 |
|---|---|---|
| `wipe_teacher` | 竖直手柄海绵，手指接触握持（无固定） | 左/右区域、语言指令、LeRobot v3 数据；右侧姿势别扭（手臂伸直），抬起阶段工具偶尔掉落 |
| `wipe_block` | 手心朝下按住海绵块，全程贴桌推动 | 姿势自然；海绵为**辅助固定**（weld），不能研究滑移 |

两个版本共同具备：

- 字段：头部相机图像，13 维状态与动作（右臂 7 + 右手 6），任务指令，20 Hz。
- 成功判定：目标区域覆盖 ≥90%、另一区域 ≤5%、工具不掉；`wipe_block` 额外检查动作平滑度。

### 2.3 触觉与滑移（`act_tools/tactile_slip/`）

- `tactile.py`：仿真手上 17 块接触垫，对应 Inspire FTP 手的 17 组触觉；另记录 6 个手指驱动出力。换手时主要改这个文件。
- `slip_teacher.py`：
  - 随机化握力、工具质量、手–柄摩擦、擦拭压深，并加入扭转或推力扰动；
  - 每帧按"工具相对手掌速度 >3 cm/s 或 >45°/s，持续 ≥100 ms"标注滑移；
  - 保留失败示范。约 40% 的示范出现真实滑移。
- `slip_detector.py`：只用触觉和手指出力（不用仿真真值）的逻辑回归滑移检测器。在未见过的 30 条示范上：

  | 方法 | 召回率 | 延迟（中位数） | 误报 |
  |---|---|---|---|
  | 学习型检测器 | 40% | 75 ms | 0.6 次/分钟 |
  | 总触觉力突变阈值 | 20% | 150 ms | 0.6 次/分钟 |

- `feedback_eval.py`：同一批 27 组随机条件下的三组对比：

  | 条件 | 成功率 (95% CI) | 掉落 | 有滑移的组数 |
  |---|---|---|---|
  | 低握力、无反馈 | 85% (68–94%) | 2 | 7 |
  | 低握力 + 触觉反馈（检测到滑移即升至最大握力） | 82% (63–92%) | 3 | 7 |
  | 始终最大握力 | 93% (77–98%) | 0 | 5 |

  **结论：第一版反馈没有带来提升。** 检测器只在 5 组中触发，且偏晚（中位数 0.5 s）；差异在置信区间内。下一步是改进早期检测与反馈动作。

### 2.4 闭环评估（`act_tools/eval/eval_policy.py`）

- 支持三种策略：
  - 脚本教师（上限）；
  - 数据回放（检查录制与执行一致）；
  - 训练好的 LeRobot 模型。
- 指标：
  - 目标/另一区域覆盖率；
  - 工具掉落、手中漂移、滑移帧数；
  - 指尖最大受力、推理耗时；
  - 成功率附 Wilson 95% 区间。
- 已验证：CPU 上训练 20 步 ACT，评估器可加载并闭环控制（仅流程验证，模型未学会）。CPU 训练约 4.5 秒/步，正式训练需要 GPU。
- `view_policy.py`：在 MuJoCo 窗口里实时观看模型（或脚本教师）擦拭，需要接显示器。

### 2.5 第一个 ACT 结果（2026-10-19，擦左边区域，仿真）

- 数据：脚本教师 200 条示范（43306 帧）。训练：实验室 RTX 4080 电脑，约 27 步/秒，6 万步约 38 分钟，loss 0.035。
- 挑选用场景（种子 1000，20 次）：2 万步 12/20，4 万步 13/20，5 万步 15/20，6 万步 17/20；脚本教师 19/20。
- **测试集（种子 2000，100 次，未用于挑选）**：

  | 模型 | 成功 (95% CI) | 平均覆盖率 | 掉落 | 擦到另一区域 |
  |---|---|---|---|---|
  | 6 万步 | **91/100 (84–95%)** | 95% | 0 | 0 |
  | 5 万步 | 69/100 (59–77%) | 78% | 0 | 0 |

  6 万步的 9 次失败都是覆盖不足（7 次在 78–89%，2 次 26%）。Jetson CPU 推理约 32–41 ms/次。
- 模型：Jetson `g1-act-tools/runs/act_left200_60k/ckpt_060000/pretrained_model`。仅仿真结果，未在真机验证。
- 训练电脑用法：`act_tools/train/train_act_labpc.ps1`（复制到 `D:\g1act\train_act.ps1`）；新训练写 `D:\g1act\next_run.json` 后运行计划任务 G1ACT_train。

### 2.6 3D 打印海绵擦（`act_tools/hardware/sponge_tool/`）

与仿真尺寸一致的手柄 + 底板 STL（手柄 Ø24 mm、高 124 mm；底板 120×70×12 mm），下面粘 120×70×24 mm 海绵。打印说明见该目录 README。

## 3. 下次要做（2026-10-19 记录）

1. Jetson 接显示器后，用 `view_policy.py` 观看 6 万步模型擦拭的效果。
2. 3D 打印海绵擦，粘海绵，称重（目标 70–100 g）。
3. 加右边区域，训练听指令擦左/右的 LC-ACT（VLA 主线下一步）。
4. 仿真加随机化（光照、颜色、桌面纹理、相机位置），为上真机做准备。
5. （可选）把评估也搬到 Windows 4080 电脑，加快 100 次评估。
6. 真机（没装手也能做）：SONIC 站立 → Vision Pro 只控手臂 → 录头部相机画面与仿真对比 → 手腕绑海绵试擦；先检查真机流程在没有手时会不会报错。
7. 装手后：触觉话题 `rt/inspire_hand/touch/r` 读取与标定，录真机示范做仿真 + 真机混合训练。

## 4. 未完成

| 任务 | 依赖 |
|---|---|
| 正式训练 ACT（单区域）→ LC-ACT（左/右指令）→ 闭环评估与泛化测试（同义指令、左右交换、工具参数变化） | **GPU** |
| 试用预训练 VLA（如 π0、GR00T N1.x）微调 | 大显存 GPU |
| 改进滑移检测（更多数据、时序模型、拇指/指腹特征）和反馈策略（更早触发、渐进增力、暂停/减速） | 无，Jetson 可做 |
| 去掉辅助固定，研究手按海绵块的纯接触握持 | 无 |
| 真机：基础控制验证 → 触觉话题读取与标定 → Vision Pro 遥操作采集真实示范 | **真机开放、确认灵巧手型号** |
| 仿真触觉与真实触觉阵列的对照标定 | 真机触觉 |

## 5. 待确认

已确认（2026-10-18）：

- **灵巧手：Inspire RH56DFTP**（带力和触觉），与仿真模型一致，17 组触觉对应关系可直接使用。真机开放后先确认 `rt/inspire_hand/touch/r` 有数据。
- **GPU：组里的工作站**，用于 ACT / LC-ACT 训练。

仍待确认：

1. 组里工作站暂时坏了，目前用实验室 RTX 4080 电脑（192.168.2.6，Windows）训练。
2. 本科还是硕士范围。
3. 真机每周可用时间。

## 6. 复现命令（在 `act_tools/` 下，使用已安装的 `.venv`）

```bash
# 生成示范
python wipe_teacher/wipe_teacher.py --episodes 10 --output runs/handle10
python wipe_block/wipe_block.py --episodes 10 --output runs/block10
python tactile_slip/slip_teacher.py --episodes 30 --seed 2 --output runs/slip_a
python tactile_slip/slip_teacher.py --episodes 30 --seed 3 --output runs/slip_b

# 滑移检测与反馈对比
python tactile_slip/slip_detector.py --train runs/slip_a/dataset --test runs/slip_b/dataset --output runs/det/detector.json
python tactile_slip/feedback_eval.py --detector runs/det/detector.json --episodes 40 --output runs/feedback

# 闭环评估
python eval/eval_policy.py --policy teacher --trials 10 --output runs/eval_teacher
python eval/eval_policy.py --policy checkpoint --checkpoint <pretrained_model> --trials 20 --output runs/eval_act
```

运行环境：`act_tools/` 是 `/home/epcap/projects/g1-act-tools` 的源代码副本，该目录下有安装好的 `.venv`（LeRobot 0.4.4、MuJoCo 3.14）。`sponge_demo.py` 中的机器人模型路径是本机绝对路径。
