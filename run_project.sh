#!/usr/bin/env bash
# A separate task selector. Existing launchers and .env remain unchanged.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF="${PROJECT_ROOT}/run_project.sh"
DRY_RUN=0
[[ "${1:-}" != --dry-run ]] || { DRY_RUN=1; shift; }

die() { printf '错误：%s\n' "$*" >&2; exit 1; }
show_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }

usage() {
    cat <<EOF
G1 项目分类入口（新增脚本）

用法：bash "$SELF" [--dry-run] <分类> [操作]

  check                         检查依赖、模型、进程；不启动控制
  sponge [view|render|check]     独立海绵演示：窗口 / 视频 / 抓持检查
  sim                           显示仿真的三个终端启动顺序
  sim scene cube                桌面方块场景 + 抓持诊断
  sim scene paint               原有滚刷墙面场景 + 接触诊断
  sim control                   SONIC 仿真控制器（lo）
  sim vr <VisionPro地址>         手动校准遥操作（F → ] → S → T）
  sim vr <VisionPro地址> --auto  沿用 solo_teleop.sh 自动校准
  real                          查看原有实物脚本入口

无参数时显示菜单；--dry-run 只打印计划执行的命令。
海绵目前是脚本驱动的独立演示，尚未接入 Vision Pro / ACT。
sim vr 默认关闭头部行走、下蹲和移动速度，保留双手与头显画面；
SONIC 仍负责全身平衡，这不等于固定基座仿真。
EOF
}

sim_guide() {
    printf '依次打开三个终端：\n'
    show_command bash "$SELF" sim scene cube
    show_command bash "$SELF" sim control
    printf '控制器显示 Init Done 后，打开头显 Tracking Streamer 并启动追踪：\n'
    show_command bash "$SELF" sim vr '<VisionPro当前IP>' --auto
    printf '校准时先按终端提示操作；仅当机器人仍悬挂时，在仿真窗口按 9 落地。\n'
    printf '结束顺序：遥操作按 O，再在控制器、场景终端分别按 Ctrl+C。\n'
    printf '自动校准模式：tmux attach -t tele 查看；tmux send-keys -t tele o 退出桥接。\n'
}

real_guide() {
    printf '实物分类保留原有入口；此菜单只显示路径，不启动硬件：\n'
    printf '  双手驱动：%s/run_both_hand_driver.sh\n' "$PROJECT_ROOT"
    printf '  SONIC 控制：%s/run_sonic_deploy.sh real\n' "$PROJECT_ROOT"
    printf '  Vision Pro 桥接：%s/run_sonic_avp_teleop.sh\n' "$PROJECT_ROOT"
    printf '  操作说明：%s/docs/OPERATIONS.md\n' "$PROJECT_ROOT"
    printf '实物手指 DDS 参数按原有操作说明配置；上述列表不是完整启动指令。\n'
}

menu() {
    cat <<'EOF'
请选择本终端要运行的部分：
  1  海绵预览窗口（当前演示）
  2  仿真：方块场景（终端 1）
  3  仿真：滚刷墙面场景（终端 1，替代方块）
  4  仿真：SONIC 控制器（终端 2）
  5  仿真：Vision Pro 自动校准（终端 3）
  6  环境检查
  7  查看实物入口
  8  查看仿真启动顺序
  0  退出
EOF
    local choice endpoint
    read -r -p '编号：' choice || return 0
    case "$choice" in
        1) main sponge view ;;
        2) main sim scene cube ;;
        3) main sim scene paint ;;
        4) main sim control ;;
        5) read -r -p 'Vision Pro 当前 IP 或房间码：' endpoint || return 0
           main sim vr "$endpoint" --auto ;;
        6) main check ;;
        7) real_guide ;;
        8) sim_guide ;;
        0) return 0 ;;
        *) die '无效编号。' ;;
    esac
}

load_config() {
    export G1_TELEOP_ROOT="$PROJECT_ROOT"
    [[ -f "$PROJECT_ROOT/bin/_common.sh" ]] || die '请将此脚本放在遥操作仓库根目录。'
    source "$PROJECT_ROOT/bin/_common.sh"
    g1_teleop_load_env
    export GR00T_ROOT="${GR00T_ROOT:-${PROJECT_ROOT}/../GR00T-WholeBodyControl}"
}

prepare_sim() {
    # Existing deploy.sh reloads .env, so reject a conflicting effective value.
    [[ "${SONIC_SIM_NET_IF:-lo}" == lo ]] || die 'SONIC_SIM_NET_IF 不是 lo；请检查当前环境及 .env。'
    [[ -z "${CYCLONEDDS_URI:-}" ]] || die '当前设置了自定义 CYCLONEDDS_URI；请使用未加载该配置的终端运行仿真。'
    export SONIC_SIM_NET_IF=lo ROS_DOMAIN_ID=0 SONIC_HAND_MODEL=inspire
    export SONIC_ENABLE_FPV=1 G1_TELEOP_ENV_PRELOADED=1
    export SONIC_GRASP_DIAGNOSTICS=0 SONIC_PAINT_DIAGNOSTICS=0
    unset RMW_IMPLEMENTATION ROS_LOCALHOST_ONLY FASTRTPS_DEFAULT_PROFILES_FILE
}

prepare_display() {
    export MUJOCO_GL=glfw
    if [[ -z "${XAUTHORITY:-}" && -f "$HOME/.Xauthority" ]]; then
        export XAUTHORITY="$HOME/.Xauthority"
    fi
    if ! command -v xdpyinfo >/dev/null; then
        [[ -n "${DISPLAY:-}" ]] || die '请从 Jetson 桌面终端启动场景，或设置 DISPLAY。'
        return 0
    fi
    if [[ -n "${DISPLAY:-}" ]] && timeout 2 xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        return 0
    fi
    local socket candidate
    local -a available=()
    for socket in /tmp/.X11-unix/X*; do
        [[ -S "$socket" ]] || continue
        candidate=":${socket##*/X}"
        if timeout 2 xdpyinfo -display "$candidate" >/dev/null 2>&1; then
            available+=("$candidate")
        fi
    done
    [[ ${#available[@]} -eq 1 ]] || die '无法唯一确定桌面显示器；请在 Jetson 桌面终端运行，或指定 DISPLAY。'
    export DISPLAY="${available[0]}"
    printf '仿真窗口显示器：%s\n' "$DISPLAY"
}

guard_controller() {
    # Local ZMQ is shared by sim/real deploy; do not attach a sim bridge to real.
    local action="$1" pid count=0
    local -a argv=()
    command -v pgrep >/dev/null || die '缺少 pgrep，无法检查控制器进程。'
    while read -r pid; do
        [[ -r "/proc/$pid/cmdline" ]] || continue
        mapfile -d '' -t argv < "/proc/$pid/cmdline" || continue
        [[ ${#argv[@]} -gt 0 ]] || continue
        [[ "${argv[0]##*/}" == g1_deploy_onnx_ref ]] || continue
        case "${argv[1]:-}" in
            lo|127.0.0.1) count=$((count + 1)) ;;
            *) die "发现非本机回环控制器 PID=$pid；请先结束原控制会话，再启动仿真。" ;;
        esac
    done < <(pgrep -f '[g]1_deploy_onnx_ref' || true)
    if [[ "$action" == control ]]; then
        [[ "$count" -eq 0 ]] || die '仿真控制器已在运行，请使用已有终端。'
    elif [[ "$action" == vr ]]; then
        [[ "$count" -eq 1 ]] || die '请先启动一个 sim control，等待 Init Done 后再接入 Vision Pro。'
    fi
}

launch() {
    show_command "$@"
    [[ "$DRY_RUN" -eq 0 ]] || return 0
    exec "$@"
}

check_environment() {
    local failures=0 item py
    printf '仓库：%s\n实物网卡（原配置）：%s\n仿真网卡：lo\n' "$PROJECT_ROOT" "${SONIC_NET_IF:-未指定}"
    for item in \
        "$GR00T_ROOT/gear_sonic_deploy/target/release/g1_deploy_onnx_ref" \
        "$GR00T_ROOT/gear_sonic_deploy/policy/release/model_decoder.onnx" \
        "$GR00T_ROOT/gear_sonic_deploy/policy/release/model_encoder.onnx" \
        "$GR00T_ROOT/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"; do
        if [[ -s "$item" ]]; then printf '[OK] %s\n' "${item##*/}"
        else printf '[缺少] %s\n' "$item"; failures=$((failures + 1)); fi
    done
    py="$(g1_teleop_python)"
    printf '遥操作 Python：%s\n' "$py"
    if ! PYTHONDONTWRITEBYTECODE=1 "$py" -c 'import mujoco, zmq, avp_stream, tyro; print("[OK] MuJoCo / ZMQ / Vision Pro SDK / tyro 导入")'; then
        failures=$((failures + 1))
    fi
    if command -v tmux >/dev/null; then printf '[OK] tmux 自动校准工具\n'
    else printf '[提示] 无 tmux，请使用手动校准 sim vr。\n'; fi
    printf '相关进程（若为空，说明当前未发现）：\n'
    pgrep -af '[g]1_deploy_onnx_ref|[s]onic_run_sim_loop.py|[g]1_avp_sonic_teleop|[a]vp_to_sonic_zmq.py|[s]ponge_demo.py' || true
    printf '检查不连接头显、不启动控制；实际图形与遥操作仍需运行验证。\n'
    [[ "$failures" -eq 0 ]]
}

main() {
    if [[ $# -eq 0 ]]; then
        if [[ -t 0 ]]; then menu; else usage; fi
        return
    fi
    local group="$1" action endpoint
    shift
    case "$group" in
        -h|--help|help) usage; return ;;
        real) [[ $# -eq 0 ]] || die 'real 只显示原有入口。'; real_guide; return ;;
        sim) [[ $# -ne 0 ]] || { sim_guide; return; } ;;
        check|sponge) ;;
        *) die "未知分类：$group；使用 --help 查看用法。" ;;
    esac
    load_config
    case "$group" in
        check)
            [[ $# -eq 0 ]] || die 'check 不接受额外参数。'
            check_environment ;;
        sponge)
            [[ $# -le 1 ]] || die 'sponge 只接受 view、render 或 check。'
            local demo="${PROJECT_ROOT}/../g1-act-tools/sponge_demo/run.sh"
            [[ -f "$demo" ]] || die "未找到海绵演示：$demo"
            printf '独立海绵演示：预抓持 + 脚本动作；尚未接入 Vision Pro / ACT。\n'
            case "${1:-view}" in
                view) launch bash "$demo" --close-up ;;
                render) launch bash "$demo" --render ;;
                check) launch bash "$demo" --grasp-check ;;
                *) die 'sponge 只接受 view、render 或 check。' ;;
            esac ;;
        sim)
            prepare_sim
            action="$1"; shift
            case "$action" in
                scene)
                    [[ $# -eq 1 ]] || die '用法：sim scene cube|paint'
                    local scene_launcher
                    case "$1" in
                        cube) scene_launcher=sonic-sim-pick-diagnostics.sh ;;
                        paint) scene_launcher=sonic-sim-paint.sh ;;
                        *) die '当前支持 cube、paint；海绵请用 sponge 分类。' ;;
                    esac
                    if [[ "$DRY_RUN" -eq 0 ]]; then guard_controller scene; prepare_display; fi
                    printf '场景：%s；DDS 接口 lo；头显画面开启。\n' "$1"
                    launch bash "$PROJECT_ROOT/bin/$scene_launcher" --interface lo ;;
                control)
                    [[ $# -eq 0 ]] || die 'sim control 不接受额外参数。'
                    [[ "$DRY_RUN" -eq 1 ]] || guard_controller control
                    printf '仿真控制器：SONIC_SIM_NET_IF=lo。\n'
                    launch bash "$PROJECT_ROOT/run_sonic_deploy.sh" sim ;;
                vr)
                    [[ $# -ge 1 && $# -le 2 ]] || die '用法：sim vr <VisionPro地址> [--auto]'
                    endpoint="$1"; shift
                    [[ -n "$endpoint" && "$endpoint" != -* && "$endpoint" != *[[:space:]]* ]] || die '请输入有效的 Vision Pro IP 或房间码。'
                    [[ $# -eq 0 || "$1" == --auto ]] || die 'sim vr 仅支持可选参数 --auto。'
                    local bridge="$PROJECT_ROOT/bin/sonic-teleop-pick.sh"
                    if [[ $# -eq 1 ]]; then
                        bridge="$PROJECT_ROOT/solo_teleop.sh"
                        if [[ "$DRY_RUN" -eq 0 ]]; then
                            command -v tmux >/dev/null || die '自动校准需要 tmux；可省略 --auto 使用手动校准。'
                            # solo_teleop.sh owns graceful replacement of the old bridge.
                            # guard_controller below runs before it can stop/start a session.
                        fi
                        export REPO="$PROJECT_ROOT"
                    fi
                    [[ "$DRY_RUN" -eq 1 ]] || guard_controller vr
                    printf '仅仿真手指；关闭实物手指 DDS；禁用头部行走/下蹲；保留双手、头显画面。\n'
                    if [[ $# -eq 1 ]]; then
                        printf '自动校准：沿用 solo_teleop.sh，先正常结束旧桥接，再执行 F → ] → S → T；落地后按提示回车。\n'
                    else
                        printf '手动校准：F → ] → 等机器人落地稳定 → S → T；P 暂停，O 退出。\n'
                    fi
                    launch bash "$bridge" "$endpoint" \
                        --host 127.0.0.1 --active-hands both --no-auto-start \
                        --enable-inspire-hand-sim --disable-inspire-hand-dds \
                        --enable-mujoco-fpv --no-head-locomotion --no-head-height-squat \
                        --no-hybrid-locomotion --keyboard-loco-speed 0 --loco-max-speed 0 ;;
                *) die "未知仿真操作：$action；使用 sim 查看启动顺序。" ;;
            esac ;;
    esac
}

main "$@"
