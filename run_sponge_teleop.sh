#!/usr/bin/env bash
# Dedicated sponge teleoperation entry; does not edit existing launchers/.env.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
die() { printf '错误：%s\n' "$*" >&2; exit 1; }
usage() {
    cat <<EOF
海绵 + Vision Pro 遥操作（使用三个终端）

终端 1：bash "$ROOT/run_sponge_teleop.sh" scene
终端 2：bash "$ROOT/run_sponge_teleop.sh" control
终端 3：bash "$ROOT/run_sponge_teleop.sh" vr 192.168.2.5 --auto

控制器出现 Init Done 后再启动终端 3；头显打开 Tracking Streamer 并启动追踪。
自动校准时按提示站稳；实际策略反馈、仿真速度和姿态检查通过后，才允许按 9 落地。
默认双手、头显画面开启；关闭头部行走、下蹲及实物手指 DDS。
省略 --auto 可手动执行 F → ] → 落地稳定 → S → T；P 暂停，O 退出。
先空手开合，再握住桌面海绵的竖直手柄、抬起、擦拭绿色区域。
掉落后先移开手，在仿真窗口按 R，只复位海绵。
SONIC 保持全身平衡；异常时冻结并保存记录，需结束并重新启动三终端。
本场景不固定基座，也不自动握持或擦拭。

环境检查：bash "$ROOT/run_sponge_teleop.sh" check
EOF
}
[[ $# -gt 0 ]] || { usage; exit 0; }
MODE="$1"; shift
case "$MODE" in
    -h|--help|help) usage; exit 0 ;;
    control) [[ $# -eq 0 ]] || die 'control 不接受额外参数。' ;;
    vr)
        if [[ $# -eq 2 && "$2" == --auto ]]; then
            export G1_TELEOP_ROOT="$ROOT"
            source "$ROOT/bin/_common.sh"
            g1_teleop_load_env
            export ROS_DOMAIN_ID=0 SONIC_HAND_MODEL=inspire
            export SONIC_ENABLE_FPV=1 G1_TELEOP_ENV_PRELOADED=1
            [[ -n "$1" && "$1" != -* && "$1" != *[[:space:]]* ]] || die '请输入有效 Vision Pro 地址。'
            [[ "${SONIC_SIM_NET_IF:-lo}" == lo && -z "${CYCLONEDDS_URI:-}" ]] || die '海绵入口仅允许本机 lo。'
            export SONIC_SIM_NET_IF=lo
            unset RMW_IMPLEMENTATION ROS_LOCALHOST_ONLY FASTRTPS_DEFAULT_PROFILES_FILE
            exec bash "$ROOT/tools/sponge_solo_teleop.sh" "$1" \
                --host 127.0.0.1 --active-hands both --no-auto-start \
                --enable-inspire-hand-sim --disable-inspire-hand-dds \
                --enable-mujoco-fpv --no-head-locomotion --no-head-height-squat \
                --no-hybrid-locomotion --keyboard-loco-speed 0 --loco-max-speed 0
        fi
        exec bash "$ROOT/run_project.sh" sim vr "$@" ;;
    check) [[ $# -eq 0 ]] || die 'check 不接受额外参数。'
        exec bash "$ROOT/run_project.sh" check ;;
    scene) [[ $# -eq 0 ]] || die 'scene 不接受额外参数。' ;;
    *) die '支持 scene、control、vr、check；不带参数查看操作说明。' ;;
esac

export G1_TELEOP_ROOT="$ROOT"
source "$ROOT/bin/_common.sh"
g1_teleop_load_env
export GR00T_ROOT="${GR00T_ROOT:-$ROOT/../GR00T-WholeBodyControl}"
[[ "${SONIC_SIM_NET_IF:-lo}" == lo && -z "${CYCLONEDDS_URI:-}" ]] || die '请检查 SONIC_SIM_NET_IF / CYCLONEDDS_URI；此入口仅使用本机 lo。'
DDS_LIB="$GR00T_ROOT/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/$(uname -m)"
export LD_LIBRARY_PATH="$DDS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset RMW_IMPLEMENTATION ROS_LOCALHOST_ONLY FASTRTPS_DEFAULT_PROFILES_FILE
if [[ "$MODE" == control ]]; then
    if pgrep -f '[/]g1_deploy_onnx_ref([[:space:]]|$)' >/dev/null; then
        die '已有 SONIC 控制器运行，请使用原终端或先正常结束它。'
    fi
    # Use the already-built controller directly. The legacy environment setup
    # invokes sudo jetson_clocks and can edit ~/.bashrc; neither is needed here.
    DEPLOY="$GR00T_ROOT/gear_sonic_deploy"
    BINARY="$DEPLOY/target/release/g1_deploy_onnx_ref"
    [[ -x "$BINARY" ]] || die "缺少已编译控制器：$BINARY"
    TRT_LIB="${TensorRT_ROOT:-$ROOT/../sonic_libs}/lib"
    export LD_LIBRARY_PATH="$DDS_LIB:$TRT_LIB:/usr/lib/$(uname -m)-linux-gnu/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    cd "$DEPLOY"
    printf '海绵仿真控制器：lo；使用现有模型；等待 Init Done。\n'
    mkdir -p "$ROOT/recordings"
    CONTROL_LOG="$ROOT/recordings/sponge_control_$(date +%Y%m%d_%H%M%S)_$$.log"
    printf '控制器日志：%s\n' "$CONTROL_LOG"
    "$BINARY" lo policy/release/model_decoder.onnx reference/example/ \
        --obs-config policy/release/observation_config.yaml \
        --encoder-file policy/release/model_encoder.onnx \
        --planner-file planner/target_vel/V2/planner_sonic.onnx \
        --input-type zmq_manager --output-type all --zmq-host localhost \
        --init-arm-pose "${SONIC_INIT_ARM_POSE:-forearms-forward}" \
        --policy-start-ramp-sec "${SONIC_POLICY_START_RAMP_SEC:-0.8}" --disable-crc-check 2>&1 | tee "$CONTROL_LOG"
    exit "${PIPESTATUS[0]}"
fi
# Do not start a second scene on the shared DDS/image channel.
if pgrep -f '[p]ython[^ ]* .*tools/(sonic_run_sim_loop|sponge_teleop_sim)\.py' >/dev/null; then
    die '已有 SONIC 仿真场景运行；请在旧场景终端按 Ctrl+C 后再启动。'
fi
PYTHON="${SONIC_VENV_SIM:-$HOME/.config/visionpro-g1-inspire-teleop/venv_sonic_sim}/bin/python"
[[ -x "$PYTHON" ]] || die "未找到仿真 Python：$PYTHON"
export PYTHONPATH="$ROOT:$GR00T_ROOT:$GR00T_ROOT/external_dependencies/unitree_sdk2_python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export SONIC_HAND_MODEL=inspire SONIC_HAND_ENDPOINT=tcp://127.0.0.1:5556
export SONIC_GRASP_DIAGNOSTICS=0 SONIC_PAINT_DIAGNOSTICS=0 ROS_DOMAIN_ID=0
export SONIC_SPONGE_STATUS="$ROOT/recordings/sponge_teleop_latest.json"
export SONIC_ROBOT_SCENE
SONIC_ROBOT_SCENE="$("$PYTHON" "$ROOT/tools/generate_sponge_scene.py")"

export MUJOCO_GL=glfw
[[ -n "${XAUTHORITY:-}" || ! -f "$HOME/.Xauthority" ]] || export XAUTHORITY="$HOME/.Xauthority"
command -v xdpyinfo >/dev/null || die '需要 xdpyinfo 检查图形桌面。'
if [[ -z "${DISPLAY:-}" ]] || ! timeout 2 xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    available=()
    for socket in /tmp/.X11-unix/X*; do
        [[ -S "$socket" ]] || continue
        candidate=":${socket##*/X}"
        if timeout 2 xdpyinfo -display "$candidate" >/dev/null 2>&1; then available+=("$candidate"); fi
    done
    [[ ${#available[@]} -eq 1 ]] || die '请在 Jetson 桌面终端运行，或设置可访问的 DISPLAY。'
    export DISPLAY="${available[0]}"
fi
printf '海绵遥操作场景：%s\n显示器：%s\nDDS：lo；头显图像：5555；手指指令：5556。\n' "$SONIC_ROBOT_SCENE" "$DISPLAY"
printf '桌面海绵由手指接触抓取；R 复位海绵；9 请求落地；异常冻结。\n'
cd "$GR00T_ROOT"
exec "$PYTHON" "$ROOT/tools/sponge_teleop_sim.py" --interface lo --enable-offscreen --enable-image-publish
