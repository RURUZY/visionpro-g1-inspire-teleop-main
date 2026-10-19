# 在 RTX 4080 电脑上训练第一个 ACT（擦左边区域）

数据：`g1_wipe_left200_dataset.zip`（200 条仿真示范，51 MB），在 Jetson 的
`/home/epcap/projects/g1-act-tools/exports/`。用 U 盘或网盘拷到训练电脑。

版本与 Jetson 一致：Python 3.10、PyTorch 2.6.0（CUDA 12.4）、LeRobot 0.4.4、PyAV 15.1.0。
RTX 4080 显存 16 GB（笔记本版 12 GB），这个配置大约用 4–6 GB。

## 1. 先确认显卡驱动

在终端（Windows 用 PowerShell）运行：

```
nvidia-smi
```

能看到 RTX 4080 和 "CUDA Version: 12.x" 即可。看不到就先装 NVIDIA 显卡驱动。

## 2. 装环境（只做一次）

需要 Python 3.10。Windows 可从 python.org 安装 3.10，勾选 "Add to PATH"。

**Windows（PowerShell）：**
```
py -3.10 -m venv $HOME\g1act
$HOME\g1act\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install lerobot==0.4.4 av==15.1.0
```

**Linux：**
```
python3.10 -m venv ~/g1act
source ~/g1act/bin/activate
python -m pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install lerobot==0.4.4 av==15.1.0
```

检查 GPU 能用（应打印 True 和显卡名）：
```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. 解压数据

把 zip 解压到一个文件夹，例如 Windows 的 `D:\g1data`，Linux 的 `~/g1data`。
解压后应有 `left200/dataset/meta`、`left200/dataset/data`、`left200/dataset/videos`。

## 4. 先跑 50 步试一下（约 1 分钟）

**Windows（PowerShell，先激活环境）：**
```
$env:HF_HUB_OFFLINE="1"; $env:HF_DATASETS_OFFLINE="1"
lerobot-train --dataset.repo_id=local/g1_inspire_wipe_sim --dataset.root=D:\g1data\left200\dataset --dataset.video_backend=pyav --policy.type=act --policy.device=cuda --policy.chunk_size=20 --policy.n_action_steps=5 --policy.push_to_hub=false --batch_size=8 --steps=50 --save_freq=50 --log_freq=10 --num_workers=2 --wandb.enable=false --output_dir=D:\g1runs\act_smoke
```

**Linux：**
```
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 lerobot-train --dataset.repo_id=local/g1_inspire_wipe_sim --dataset.root=$HOME/g1data/left200/dataset --dataset.video_backend=pyav --policy.type=act --policy.device=cuda --policy.chunk_size=20 --policy.n_action_steps=5 --policy.push_to_hub=false --batch_size=8 --steps=50 --save_freq=50 --log_freq=10 --num_workers=4 --wandb.enable=false --output_dir=$HOME/g1runs/act_smoke
```

看输出里的 `updt_s`（每步耗时）和 `loss`（应在下降）。把这几行发回来。

## 5. 正式训练（2 万步）

把上面命令里的 `--steps=50 --save_freq=50 --log_freq=10` 换成
`--steps=20000 --save_freq=5000 --log_freq=200`，`--output_dir` 换成 `...\act_left200`（Linux 为 `.../act_left200`）。
中途每 5000 步保存一次。

## 6. 拷回 Jetson 评估

把 `act_left200/checkpoints/020000/pretrained_model` 整个文件夹拷回 Jetson，例如放到
`/home/epcap/projects/g1-act-tools/runs/act_left200/pretrained_model`，然后在 Jetson 上：

```
cd /home/epcap/projects/g1-act-tools
.venv/bin/python eval/eval_policy.py --policy checkpoint --checkpoint runs/act_left200/pretrained_model --side left --trials 20 --output runs/eval_act_left200
```

结果在 `runs/eval_act_left200/summary.json`：成功率（附 95% 置信区间）、覆盖率、工具掉落等。

## 说明

- 这是只擦一边、不读语言指令的原版 ACT，用来验证"仿真数据 → 训练 → 闭环评估"整条流程。
- `--dataset.repo_id` 必须是 `local/g1_inspire_wipe_sim`（与数据集生成时一致）。
- 失败或报错时，把完整的报错文字发回来。
