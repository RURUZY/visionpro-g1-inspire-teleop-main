# 宇树 SDK 接口的 MuJoCo 替身（仿真桥）与模型执行

目的：程序只通过宇树 SDK 话题和机器人交互，先在仿真里验证，再原样用于真机（README 第 9 节"仿真与真机使用相同 SDK 消息"）。

| 程序 | 作用 |
|---|---|
| `sdk_sim/g1_sdk_sim.py` | 用擦拭场景（G1 + Inspire、桌子、海绵，骨盆固定）扮演真机：订阅 `rt/lowcmd`、`rt/arm_sdk`（按权重混合）、`rt/inspire_hand/ctrl/{l,r}`；发布 `rt/lowstate`、`rt/inspire_hand/state/{l,r}`、`rt/inspire_hand/touch/r`（17 组触觉）；ZMQ 5560 发头部相机画面，5561 控制（重置回合、读成绩） |
| `sdk_deploy/run_policy_sdk.py` | 让训练好的 ACT 通过 SDK 控制机器人：13 维动作（右臂 7 关节 + Inspire 6 通道）；异步分块推理保持 20 Hz；保护层（arm_sdk 权重渐变接管/交还、每关节限速、关节限位、看门狗） |

默认 DDS 域 1（仿真）；真机是域 0，执行程序目前禁止域 0，等安全检查完成后再开放。

## 运行（Jetson，`g1-act-tools/.venv`，需已安装 unitree_sdk2py、cyclonedds、pyzmq）

```
.venv/bin/python sdk_sim/g1_sdk_sim.py --task pickup            # 终端 1；加 --viewer 打开窗口
.venv/bin/python sdk_deploy/run_policy_sdk.py --checkpoint runs/act_pick_rand1000/ckpt_200000/pretrained_model \
    --trials 20 --output runs/sdk_eval                            # 终端 2
```

## 近似与限制（仅仿真）

- 骨盆固定、连杆重力补偿：kp/kd/力矩不起作用，指令关节直接作为位置执行器目标。
- 触觉值是每块接触垫的法向力（0.01 N）平均分到该组的每个点，不是标定后的传感器模型。
- arm_sdk 权重为 0 时，手臂回到重置时的姿态（代替宇树控制器的保持姿态）。
