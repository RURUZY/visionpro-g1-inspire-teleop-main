# Vision Pro–G1 Teleoperation: Operations Runbook

This is the canonical copy-paste guide for daily operation. For installation, architecture, code ownership, and known limitations, see [`HANDOVER.md`](HANDOVER.md).

All commands are run from the repository root on the Linux lab workstation. Replace `<VP>` with the Vision Pro IP address or Tracking Streamer room code.

## Jetson AGX Orin checkpoint — 2026-10-08

### 最新保存状态与 2026-10-09 复测

四指仿真映射已经修正，29 项自动测试通过。回放同一组手指弯曲数据
`[65.4, 73.8, 75.0, 69.8]`，原指令为 `[664, 602, 593, 631]`，
修复后为 `[13, 0, 0, 0]`（0 为闭合，1000 为张开）；该指令也已通过
独立 MuJoCo 四指闭合验证。尚未确认操作者重启桥接后的现场效果。

明天按下面三个终端的顺序启动，完成自动校准，再分别测试左右手张开和
握拳，各保持 3 秒。确认四指既能闭合也能重新张开。检查桥接启动日志中的
`using sim finger flexion range 10-70 degrees`（已有个人校准文件时使用该文件）。
本次修复只改变无个人校准文件时的仿真四指默认范围，拇指仍待单独校准；
跟踪值间歇归零的原因以及头显画面是否稳定，继续现场检查。无需重新编译
SONIC，也无需重新下载模型。

本次进度已保存，回来后按以下顺序启动。当前网络记录：Jetson
`192.168.2.22`，Vision Pro `192.168.2.5`；地址变化时替换命令中的 IP。
三个终端均先进入 `~/projects/visionpro-g1-inspire-teleop-main`。

1. 终端一：`SONIC_HAND_MODEL=inspire SONIC_ENABLE_FPV=1 ./run_sonic_sim_loop.sh`。
   等待五指仿真窗口和 `Camera publish: ON`；报错时先保留终端输出。
2. 终端二：`./run_sonic_deploy.sh`，等待 `Init Done`。
3. 头显打开 Tracking Streamer，终端三运行：

   ```bash
   bash solo_teleop.sh 192.168.2.5 --loco-velocity-gain 0 --loco-yaw-gain 0 --no-head-height-squat
   ```

脚本自动执行 `F → ]`。提示落地时点击 Jetson 的 MuJoCo 窗口按 `9`，
回到终端三按回车，保持校准姿势，脚本继续执行 `S → T`。这套命令先关闭
头部行走、转向和下蹲输入，方便检查手臂与手指。三个进程需要持续运行。
桥接在 tmux 中，关闭终端窗口不一定会停止它；重新运行脚本会替换 `tele`
会话。停止桥接：`tmux send-keys -t tele o`。

已完成：修复 SONIC 在 ENGAGE 切换 motion 后丢失 VR encoder mode 的问题，
已在 AGX 编译并替换部署程序，操作者确认手臂可以跟随；仿真入口改为
Inspire FTP 五指手，接入六通道手指命令。五指仿真的 10 项协议、动力学和
ZMQ 测试通过，也完成相机渲染检查。仍需操作者重新启动后验证完整头显链路。

待复测：最后一次现场检查中仿真和相机进程已退出，5555 端口在 4 秒内
收到 0 帧；只有视频服务 ready 不能证明头显已经收到图像，退出原因尚不明确。
更早的 20 秒采样虽持续收到桥接消息，但姿态和手指值完全不变，不能证明
头显产生了新追踪帧。启动后分别检查张手、握拳、手臂移动和头显画面。
左右手校准 JSON 尚未生成，拇指旋转命令曾固定为 800；需要单独做手指
张开/握拳校准，`F/S` 不会校准手指范围。实机 Inspire 驱动仍是独立链路，
本次新增适配器仅控制仿真。

### 保存的 SONIC 补丁与编译脚本

上游 GR00T 仓库不是个人仓库，因此其源码修复以
[`sonic-vr-encoder-mode.patch`](../tools/jetson/sonic-vr-encoder-mode.patch)
保存在这里。补丁基于 GR00T commit
`b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`。当前机器已经应用并编译，
正常重启无需重新应用或编译。另一份兼容的干净源码可按以下方式恢复：

```bash
cd ~/projects/visionpro-g1-inspire-teleop-main
git -C ../GR00T-WholeBodyControl apply --check "$PWD/tools/jetson/sonic-vr-encoder-mode.patch"
git -C ../GR00T-WholeBodyControl apply "$PWD/tools/jetson/sonic-vr-encoder-mode.patch"
cp tools/jetson/build_sonic_deploy.sh tools/jetson/copy_sonic_libs.sh ../
cd ..
bash build_sonic_deploy.sh
```

脚本须放在 `~/projects`，与 GR00T、`sonic_libs`、`jetson-teleop` 同级。
它依赖现有 ONNX Runtime 1.16.3 和 TensorRT 10.7/CUDA 12.6 运行库；若未
复制运行库，先运行 `bash copy_sonic_libs.sh <旧Jetson用户@地址>`。
构建依赖解压到本地目录，不需要 sudo。实际编译、动态库检查和 usage
冒烟检查已通过；上游 FK 测试因缺少 `reference/bones_072925_test` 数据而跳过。
仓库保存源码、补丁与脚本；运行库、模型下载缓存、构建产物、`.env` 和运行日志
保留在本机。五指模型可由 `tools/setup_inspire_sim.py` 按固定上游版本重新生成。

## 1. Every-session calibration

In the AVP bridge terminal:

```text
F → ] → S → T
```

- `F` — hold a forearms-forward L-shape for about two seconds; records head and wrist references.
- `]` — engage the SONIC policy; wait until the robot is stable.
- `S` — match the displayed robot arm pose and hold for about two seconds.
- `T` — begin live teleoperation.
- `H` — re-zero head facing and squat height.
- `P` — pause or resume pose mapping.
- `o` — stop and exit from the bridge terminal.
- `O` — emergency stop from the SONIC deploy terminal.

In MuJoCo, press `9` in the simulator window after `]` if the robot remains suspended by the elastic band.

## 2. MuJoCo whole-body teleoperation

Start three terminals in order.

Terminal 1:

```bash
./run_sonic_sim_loop.sh
```

Terminal 2:

```bash
./run_sonic_deploy.sh
```

Terminal 3:

```bash
./run_sonic_avp_teleop.sh <VP>
```

Complete `F → ] → S → T`.

The experimental MuJoCo first-person stream starts by default but is not reliably visible inside Vision Pro. Disable it when unnecessary:

```bash
./run_sonic_avp_teleop.sh <VP> --no-mujoco-fpv
```

This disables video only; simulated Inspire fingers remain enabled. Use
`--no-inspire-hand-sim` to disable simulated fingers independently.

## 3. MuJoCo pick-and-place

### 抓取诊断窗口与同步记录

第一版提供手部近景、左右手切换、碰撞外形、接触点和方块受力箭头，以及
目标/实际关节角度、关节力矩、接触力、离桌高度和相对手腕位移曲线。
窗口是独立的只读进程，不发送机器人命令。左右手、头部控制、摩擦、力矩
上限和联动约束均沿用原配置。

如果旧仿真正在运行，先放下方块，结束桥接（`tmux send-keys -t tele o`），
再在原控制和仿真终端按 Ctrl+C。不要同时启动两套仿真或桥接。所有终端
先进入 `~/projects/visionpro-g1-inspire-teleop-main`，按顺序启动：

```bash
# 终端一：带诊断记录的桌面场景
bash bin/sonic-sim-pick-diagnostics.sh

# 终端二：等待 Init Done
./run_sonic_deploy.sh sim

# 终端三：头显启动 Tracking Streamer 后，自动校准
bash solo_teleop.sh 192.168.2.5 --print-debug

# 终端四：电脑桌面上的诊断窗口
bash bin/grasp-diagnostics.sh
```

仍按自动校准提示操作。需要落地时，机器人悬挂才按仿真窗口的 `9`，已经
站在地上则直接回终端三按回车。诊断窗口打开不代表头显视频显示已修复。

诊断窗口中 `left/right` 选择观察手；`front/side/top/whole` 选择正面、
侧面、俯视或全身。拖动近景画面可旋转视角，滚轮缩放。碰撞外形开关显示
实际碰撞几何；模型中的网格显示不等于精确凸包边界。接触点以观察器副本
重算，橙色受力箭头来自原仿真该采样步的求解结果。箭头 1 N 对应显示长度
2 cm，超过 15 cm 截断显示；实际数值仍完整记录。

界面含义：

- 原始目标：由 0–1000 指令换算的关节角度；限速后目标：控制器本步实际
  追踪的目标；实际角度：关节状态。比较三者区分映射不足与关节跟不上。
- 灰柱是 PD 力矩请求，蓝柱是 MuJoCo 执行器产生的关节力矩；接近当前
  1.4 N·m 上限时变红。力矩不是指尖夹持力。
- 法向接触力之和是手对方块各接触法向力的标量之和，不是净托举力。
- 离桌高度使用旋转后方块最低点，不是中心高度；桌面承载同时检查接触力。
- 相对手腕位移以当前曲线窗口的首帧为参考，也会包含主动摆放动作，不能
  单独用作自动滑落判定。
- 指令接收状态只反映仿真通信超时；头显骨架有效性仍查看桥接日志。
- 数据超过 1 秒未更新会提示停更。开始/抬起/掉落/结束按钮只写事件标记，
  不触发机器人动作，也不自动认定抓取成功。

仿真开启后自动以约 50 Hz 仿真时间采样；诊断窗口约 5 FPS 更新，GUI 不
参与控制循环。记录包含 `model.mjb`（实际模型）、`metadata.json`、
`samples.jsonl`、人工标记 `markers.jsonl` 和退出时的 `recording_status.json`。
目录由终端一打印，为 `recordings/grasp_diagnostics_<UTC时间>/`。写盘使用
有界队列，丢帧/错误显式记录。查看记录状态确认保存完成；强制杀进程可能
丢失最后一段缓冲。曲线只显示最近 300 个已显示样本，原始文件保留全部
已记录样本。短于采样间隔的接触事件可能漏掉。

回放和导出（将路径替换成实际记录目录）：

```bash
bash bin/grasp-diagnostics.sh --replay recordings/grasp_diagnostics_实际目录
bash bin/grasp-diagnostics.sh --replay recordings/grasp_diagnostics_实际目录 --snapshot /tmp/grasp.png
bash bin/grasp-diagnostics.sh --replay recordings/grasp_diagnostics_实际目录 --video /tmp/grasp.mp4
```

回放使用保存的状态重绘，不是重新运行动力学；导出为 5 FPS 诊断视频。
截图/视频导出使用 EGL，可无桌面运行。当前 Euler 场景中，控制角度与接触
力对应同一步求解，记录 `sim_time`（步前）与 `post_step_time`（步后）；
切换积分器后需重新检查力与姿态的时间对应。MuJoCo 二进制模型建议用记录
时的相同版本回放。诊断窗口和渲染会消耗 GPU/CPU，仍需现场检查仿真速度。

第一版不包含自动桌面抓取基准，不增加手指力度，也不把方块绑定到手上。
下一次先录 3 次空手拇指动作和 3 次桌面抓放，依据记录再决定改哪些参数。

### 握紧时仿真手意外张开

仅仿真输出现在会检查指骨数据：缺失、非有限值、重合关节点或退化掌面
不会直接变成张手指令。左右手各自短暂保持上一条有效指令；持续失效达到
`--hand-tracking-timeout`（默认 1 秒）后松开。桥接持续发送保持指令，因此
不会在短暂丢失期间先触发仿真接收端 0.5 秒的通信超时。
正常张手仍立即进入原有平滑映射，通信中断时接收端仍按原规则松开。
同时开启实机 DDS 的模式保留原行为。

日志中的 `[hand_tracking] left/right: holding` 表示数据失效后暂时保持，
`open_after_loss` 表示无有效历史手势或持续失效后松开，`tracking` 表示恢复。
这些检查不能检测 SDK 重复返回的、几何上仍有效的旧帧，也不能识别所有
几何合理但估计错误的手势。已有掉落日志未记录原始骨架或接触力，不能据此
确认每次掉落原因，需重新抓取并对照状态日志。

修改在重启桥接后生效。使用 `solo_teleop.sh` 自动校准时，如果机器人已
解除悬挂并站在地上，中途落地提示处直接按回车，不要再次按 `9` 切换悬挂。

### 仅校准拇指，保留当前四指参数

先放下物体、停止桥接（`tmux send-keys -t tele o`，等待退出），头显打开
Tracking Streamer，然后运行：

```bash
bash bin/calibrate-thumb.sh 192.168.2.5 --side right
```

按四次提示分别采集拇指伸直、弯曲、向外展开、向掌心对向的姿势。每次按
回车后有 5 秒准备时间，再采集 3 秒。程序只读取追踪，不发送机器人指令。
无效骨架和完全重复的骨架不计入有效样本；每姿势至少需要 30 个不同样本。
端点角度相差不足 5 度或变化无法与采样波动区分时拒绝保存，需重试。

保存时只更新拇指两通道，四指保留当前有效范围（包括无个人文件时仿真
使用的 10–70 度范围）。原文件、有效参数及采样记录保存到
`recordings/thumb_calibration_*`。输入校准不改变输出端点、力度或关节限位。
重启桥接后生效；先空手检查方向和范围，再抓物体。此流程针对当前仿真
配置，生成的个人文件不要未经验证用于实机。左手可另用 `--side left`。

### 已校准拇指的全行程对照实验

2026-10-11 的抓取记录已保存 8,911 条样本，无记录器丢帧或写入错误。
记录中存在右手承载、方块离桌且悬挂关闭的片段。空手时右拇指弯曲目标
约 26.6°、实际约 26.5°，根部目标约 53.4°、实际约 53.3°；旧输出端点
200–800 只覆盖模型约 20–80% 行程。左右手已有各自校准文件，无需重新
采集这次基线。拇指抓住方块时也可能因物体接触而无法到达空手闭合角度。

保留现有默认配置，另用 `--sim-thumb-full-range` 测试完整输出范围。
该参数把双手两个拇指通道统一为 1000 张开、0 闭合，覆盖原拇指输出端点
与命令反转选项。左右手仍读取各自的个人输入校准，四指参数不变；关节
限位、PD 增益、力矩上限不变。只允许仿真手输出，开启实机手 DDS 时拒绝
启动。去掉这个参数并重启桥接即可恢复原抓取配置。

启动带诊断记录的场景与 deploy 后，在项目根目录运行：

```bash
mkdir -p recordings
LOG="$PWD/recordings/bridge_thumb_full_$(date +%Y%m%d_%H%M%S).log" \
bash solo_teleop.sh 192.168.2.5 \
  --loco-velocity-gain 0 --loco-yaw-gain 0 --no-head-height-squat \
  --sim-thumb-full-range
```

自动校准后，分别用左右手空手伸直、弯曲、外展、对掌，各保持 3 秒；
对照修改后的目标/实际角度，再检查原先的方块抓放是否仍稳定。这是修改后
的验证，不需要重新做输入校准。已落地时跳过 `9`，只确认自动流程提示。

左手配置已存在，但上一轮抓取记录主要验证右手，仍需独立验收左手抓取。
刷墙任务下一阶段按“抓住滚筒柄 → 提出杯托保持 → 空中直线移动 → 轻触墙面 →
短距离刷动”推进。独立干滚筒、墙板和接触记录入口已增加，使用
`bash run_sonic_sim_loop_paint.sh`，并配合 deploy 的 `sim` 模式及
第三终端的 `bash bin/sonic-teleop-paint.sh --auto`（沿用 IP `192.168.2.5`，
开启头显回传，T 后直接开启头显行走，关闭键盘移动）。无需按 B 解锁，
不再使用空格锁定；指令上限为 0.15 m/s。自动流程结束后可执行
`tmux attach -t tele` 查看桥接状态。场景不再有桌子：滚筒刷竖直插在右前方
立柱的杯托里（滚筒朝上），右手原地即可握住竖直刷柄；自立式墙板前表面在 x=0.60 m，
目标中心高 1.22 m、在右手前方。刷子掉落后移开双手，
点击 MuJoCo 窗口按 R 可只复位刷子，保留当前校准、连接和机器人站位。
刷墙专用头部仿真视角同时覆盖墙面目标与杯托里的刷柄。
双手身体映射与已录制抓盒子工作流对齐，校准时两只手都需可追踪。
完整命令见 [刷墙先导实验](PAINT_EXPERIMENT.md)；`--no-mujoco-fpv` 会关闭头显
画面，不应加入正常头显操作命令。新记录器采集滚筒—墙面、手—工具
接触和工具相对手腕姿态；没有模拟颜料转移，也没有自动执行或证明刷墙成功。
以上仅指仿真任务。

重复启动终端 3 时，保持 Tracking Streamer 打开并继续追踪；启动脚本先正常结束旧桥接，
电脑端会自动重试无首帧／断流连接。头显端的重复连接效果仍需实测。

Terminal 1:

```bash
./run_sonic_sim_loop_pnp.sh
```

Terminal 2:

```bash
./run_sonic_deploy.sh
```

Terminal 3:

```bash
./run_sonic_avp_teleop_pick.sh <VP>
```

Complete `F → ] → S → T`. Pinching in Vision Pro closes the simulated Inspire fingers.

Keyboard hybrid locomotion is not enabled automatically. To use it:

```bash
./run_sonic_avp_teleop_pick.sh <VP> --hybrid-locomotion
```

## 4. Real G1 preflight

> Keep the robot hoisted for initial engagement. Clear people and equipment from its reachable area, and keep the Unitree remote ready.

Before running SONIC:

1. Connect the workstation Ethernet interface to the `192.168.123.x` robot network.
2. Confirm `.env` contains the correct `SONIC_NET_IF`, such as `enp3s0`.
3. Hoist the G1 with its feet initially off the ground.
4. Power on and wait for zero-torque mode; joints should move freely by hand.
5. Press `L2+R2` on the Unitree remote until the robot enters debug mode with the yellow LED and damping behavior.
6. Confirm no person is within arm or leg reach.

If deploy repeatedly reports `Failed to switch to Release Mode`, the sport controller is still active. Re-enter debug mode or reboot the robot and repeat the preflight.

## 5. Real G1 without physical finger control

Terminal 1:

```bash
./run_sonic_deploy.sh real
```

Terminal 2:

```bash
./run_sonic_avp_teleop.sh <VP> --no-mujoco-fpv --no-inspire-hand-sim --disable-inspire-hand-dds
```

`--disable-inspire-hand-dds` (alias `--no-inspire-hand-dds`) explicitly disables
physical finger commands. This mode does not require `inspire_sdkpy` or the hand
driver processes. Arm tracking remains available. Replace any existing
`--enable-inspire-hand-dds` flag; the enable and disable flags are mutually exclusive.

For keyboard-assisted positioning:

```bash
./run_sonic_avp_teleop.sh <VP> \
  --no-mujoco-fpv \
  --no-inspire-hand-sim \
  --disable-inspire-hand-dds \
  --hybrid-locomotion
```

Complete `F → ] → S → T` slowly and verify stability after every stage.

## 6. Real G1 with both Inspire Hands

Check both hands before starting:

```bash
ping -c 2 192.168.123.210
ping -c 2 192.168.123.211
```

Terminal 1:

```bash
./run_sonic_deploy.sh real
```

Terminal 2:

```bash
./run_both_hand_driver.sh --dds-network enp3s0
```

Terminal 3:

```bash
./run_sonic_avp_teleop.sh <VP> \
  --no-mujoco-fpv \
  --no-inspire-hand-sim \
  --enable-inspire-hand-dds \
  --hand-dds-sides both \
  --hand-dds-network enp3s0
```

Add `--hybrid-locomotion` to the Terminal 3 command when keyboard positioning is needed.

For one physical hand only:

```bash
# Left hand driver
./run_both_hand_driver.sh --sides l --dds-network enp3s0

# Add to the teleop command
--enable-inspire-hand-dds --hand-dds-sides l --hand-dds-network enp3s0
```

Hand-driver frequency output confirms Modbus communication with the hand. It does not prove that DDS commands are arriving from the bridge.

## 7. Hybrid keyboard locomotion

Enable:

```text
--hybrid-locomotion
```

After reaching `T`, focus the bridge terminal and hold:

- `W` — walk forward;
- `S` — walk backward while preserving facing;
- `,` / `.` — strafe left/right;
- `A` / `D` or `j` / `l` — turn in place;
- `space` or `r` — stop while preserving facing.

Come to a complete stop before reversing or turning in place. Before `T`, `S` means arm synchronization rather than backward walking.

## 8. Common bridge options

Append options after `<VP>`:

```bash
./run_sonic_avp_teleop.sh <VP> [OPTIONS]
```

- `--hybrid-locomotion` — enable keyboard/head locomotion together.
- `--no-mujoco-fpv` — disable the experimental first-person video path.
- `--loco-max-speed 0.4` — reduce maximum walking speed.
- `--no-loco-imu-correction` — disable base-IMU yaw correction for diagnosis.
- `--active-hands right` — ignore left-arm input.
- `--left-wrist-orientation-mode neutral` — diagnostic fallback for left-arm hunching.
- `--enable-inspire-hand-dds` — publish physical Inspire Hand commands.
- `--hand-dds-sides both` — select both hand DDS topics.
- `--hand-dds-network enp3s0` — select the hand DDS interface.
- `--print-debug` — print command and tracking state; standard wrappers already enable it.

Defaults in the standard wrapper:

- head locomotion and squat control: on;
- staged calibration: on;
- arm tracking-loss hold: on;
- IMU yaw correction: on;
- keyboard hybrid mode: off;
- command publish rate: 50 Hz.

## 9. Normal shutdown and emergency stop

Normal shutdown:

1. Press `o` in the bridge terminal and confirm it exits.
2. Stop the hand drivers with `Ctrl+C`.
3. Stop SONIC deploy.
4. Stop MuJoCo if running.

Unexpected real-robot movement:

1. Press `O` in the SONIC deploy terminal immediately.
2. Use the Unitree remote emergency control if required.
3. Do not restart until the cause is understood.

Do not rely on closing a terminal window as the primary emergency-stop method.

## 10. Troubleshooting

### CALIB_SYNC fails

- Confirm SONIC deploy is running.
- Confirm SONIC feedback is available on ZMQ port `5557`.
- Restart the bridge and repeat `F → ] → S → T`.

### Robot does not walk

- Confirm live teleoperation has reached `T`.
- Press `H` to re-zero facing.
- Check bridge debug output for nonzero movement and speed.
- For keyboard control, confirm `--hybrid-locomotion` was passed.

### Walking direction or facing drifts

- Face the intended neutral direction and press `H`.
- Repeat `F` if needed.
- Confirm base-IMU feedback is reaching the bridge.

### Arms move suddenly when the policy starts

- Start the bridge before pressing `]`.
- Complete `F` so the initialized arm targets are buffered.
- Do not skip `S`.

### A wrist disappears and the arm moves incorrectly

- Confirm arm tracking hold has not been disabled.
- Pause with `P` if tracking does not recover.

### Left arm pulls the torso or hunches

Try one diagnostic change at a time:

```text
--active-hands right
--left-wrist-orientation-mode neutral
```

Also recheck the synchronized pose at `S`.

### Simulated Inspire fingers do not move

The standard simulator launcher now loads Inspire FTP five-finger hands and
subscribes directly to `tcp://127.0.0.1:5556/inspire_hand`. The default upstream
`scene_43dof.xml` contains Dex3 three-finger hands and cannot consume that topic.
`SONIC_HAND_MODEL=dex3` explicitly selects the original upstream simulator.

After changing the hand model, stop the bridge, deploy, and simulator; restart
the simulator first, then deploy, then the existing `solo_teleop.sh` workflow.
No C++ rebuild is needed for the hand simulation adapter. The simulator prints:

```text
[inspire_sim] Five-finger FTP hands listening on .../inspire_hand
[inspire_sim] Hand commands received; finger tracking active.
```

The six commands are little, ring, middle, index, thumb bend, and thumb rotation.
Each hand has six driven joints and six coupled joints. Commands stop updating
for 0.5 seconds → the simulated fingers open. Body control stays on the existing
29-motor DDS interface. For a bridge on another host/port, set
`SONIC_HAND_ENDPOINT=tcp://HOST:PORT` on the simulator.

Assets are generated by `tools/setup_inspire_sim.py` from the official
[Unitree G1 Inspire FTP description](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf),
pinned to commit `5994d4faef0a9cadd3287f8de0199a67eeb2a259`. Downloaded URDFs,
meshes, and their BSD license are cached under `assets/mujoco/inspire_ftp/`;
the assembled model and scenes are under `assets/mujoco/g1_runtime/`.
The first standard simulator launch generates them if absent. The original
GR00T body model and its camera are retained; the Dex3 palms/fingers are replaced.

If fingers move but cannot fully close, run a separate open/fist calibration;
`F → ] → S → T` calibrates head/wrists, not finger ranges. The per-hand calibration
JSON files are ignored by Git and must be copied or regenerated on a new machine.

2026-10-08 follow-up: when only simulated hands are enabled and no personal
calibration file exists, the four fingers now use 10° open / 70° closed. The
measured flexion is a weighted mean of joint angles; recorded closed-fist values
were around 65–75°, so the legacy 165° endpoint left commands near half closure.
This is a fallback based on the observed session, not a completed personal
calibration. Personal files override it, and physical DDS output (including mixed
sim/DDS mode) retains the previous ranges. Thumb ranges are unchanged and still
require calibration. Restart the AVP bridge to load this change; the simulator
and deploy do not need rebuilding. Repeat the automatic calibration workflow;
if the robot is already standing on the ground, acknowledge the drop prompt with
Enter without toggling the suspension again with `9`.

### Physical Inspire fingers do not move

- Ping both hand IPs.
- Confirm the hand drivers are running.
- Confirm all Inspire DDS flags are present in the bridge command.
- Use the same network interface for the bridge and drivers.

### Python import or ZMQ error

- Confirm `SONIC_PYTHON` points to the intended environment.
- Run `"$SONIC_PYTHON" -m pip install -e .`.
- Verify `"$SONIC_PYTHON" -c "import zmq"` succeeds.

### SONIC cannot find the real robot

- Confirm the workstation is on `192.168.123.x`.
- Check `SONIC_NET_IF`.
- Confirm the G1 is in debug mode.

### MuJoCo waits for camera frames

Start the standard MuJoCo loop or disable video with:

```text
--no-mujoco-fpv
```
