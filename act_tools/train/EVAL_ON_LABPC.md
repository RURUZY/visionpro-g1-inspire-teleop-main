# 在实验室 RTX 4080 电脑上做闭环评估（Windows）

每次评估约 3 秒（Jetson 约 15–20 秒），画面与 Jetson 逐像素一致。

## 一次性准备（已在 192.168.2.6 上完成）

1. 训练环境里加装 MuJoCo：`D:\g1act\venv\Scripts\python.exe -m pip install mujoco==3.14.0`
2. 模型网格拷到 `D:\g1act\assets\g1_meshes`、`D:\g1act\assets\inspire_meshes`。
3. 机器人 XML：把 `assets/mujoco/g1_runtime/g1_29dof_with_inspire_hand.xml` 里的网格路径换成上面两个目录，
   存为 `D:\g1act\assets\g1_29dof_with_inspire_hand.xml`。
4. 代码：`act_tools` 下的 `wipe_teacher wipe_block tactile_slip pickup_teacher eval sponge_demo` 拷到 `D:\g1act\code\`。
   代码有改动时重新拷一次。

## 运行（cmd，或从 Jetson 用 ssh labpc "..."）

```
set HF_HUB_OFFLINE=1&& set MUJOCO_GL=glfw&& set G1_ROBOT_XML=D:/g1act/assets/g1_29dof_with_inspire_hand.xml&& D:\g1act\venv\Scripts\python.exe D:\g1act\code\eval\eval_policy.py --policy checkpoint --device cuda --checkpoint D:\g1runs\<run>\checkpoints\120000\pretrained_model --variant pickup --side left --seed 2000 --trials 50 --output D:\g1runs\<eval_name>
```

- `MUJOCO_GL=glfw` 必须设置（Jetson 上默认用的 egl 在 Windows 上不可用）。SSH 会话里也能渲染。
- 长时间评估从 Jetson 上用 `setsid nohup` 启动的脚本里调用 ssh，断开 VS Code 也不影响；
  失败时可退回 Jetson 评估（见 `auto_pickup_jetson.sh`）。
