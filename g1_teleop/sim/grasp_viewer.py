"""Independent read-only desktop display and offline snapshot replay."""

from collections import deque
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import font_manager, rcParams

FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    rcParams["axes.unicode_minus"] = False
CHANNEL_LABELS = ["小指", "无名指", "中指", "食指", "拇指弯曲", "拇指根部"]


def read_samples(path):
    samples = []
    with Path(path).open() as stream:
        for line in stream:
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                # Permit a final partially written line while recording.
                if line.endswith("\n") or stream.read().strip():
                    raise
    return samples


class GraspDisplay:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_binary_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=400, width=640, max_geom=self.model.ngeom + 256)
        self.history = deque(maxlen=300)
        self.figure = Figure(figsize=(12.8, 4.0), dpi=100, facecolor="#f4f6fa")
        self.canvas = FigureCanvasAgg(self.figure)
        self.last_key = None
        self.episode = None

    def _arrow(self, start, force):
        scene = self.renderer.scene
        if scene.ngeom >= scene.maxgeom or np.linalg.norm(force) < .001:
            return
        # Display scale is fixed (1 N = 2 cm); very long arrows capped at 15 cm.
        delta = np.asarray(force) * .02
        if np.linalg.norm(delta) > .15:
            delta *= .15 / np.linalg.norm(delta)
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([1., .35, .05, 1.]))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, .003, np.asarray(start), np.asarray(start) + delta)
        scene.ngeom += 1

    def render(self, sample, side="right", angle="side", collision=False, orbit=None, zoom=1.):
        self.data.qpos[:] = sample["qpos"]
        self.data.qvel[:] = sample["qvel"]
        self.data.ctrl[:] = sample["ctrl"]
        mujoco.mj_forward(self.model, self.data)  # Observer copy only; never steps the live sim.
        wrist = self.model.body(f"{side}_wrist_yaw_link").id
        cube = sample.get("cube")
        center = self.data.xpos[wrist].copy()
        if cube is not None and np.linalg.norm(np.asarray(cube["position"]) - center) < .4:
            center = (center + np.asarray(cube["position"])) / 2
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = center
        camera.distance = .48
        camera.azimuth, camera.elevation = {"front": (180, -20), "side": (90, -20), "top": (90, -85), "whole": (125, -20)}[angle]
        if angle == "whole":
            camera.lookat[:] = [.25, 0, .7]
            camera.distance = 2.2
        if orbit is not None:
            camera.azimuth, camera.elevation = orbit
        camera.distance *= zoom
        options = mujoco.MjvOption()
        options.geomgroup[3] = int(collision)
        if collision:
            options.geomgroup[1] = 0
        options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.renderer.update_scene(self.data, camera=camera, scene_option=options)
        for contact in sample["contacts"]:
            if contact["side"] == side:
                self._arrow(contact["position"], contact["force_on_cube_world_n"])
        return Image.fromarray(self.renderer.render().copy())

    def charts(self, sample, side):
        key = (sample["episode"], sample["seq"])
        if self.episode != sample["episode"] or (self.history and sample["seq"] < self.history[-1]["seq"]):
            self.history.clear()
        if key != self.last_key:
            self.history.append(sample)
        self.last_key, self.episode = key, sample["episode"]
        self.figure.clear()
        axes = self.figure.subplots(2, 2)
        hand = sample["hands"][side]
        if "actual_rad" not in hand:
            return Image.new("RGB", (1280, 400), "white")
        x = np.arange(6)
        axes[0, 0].plot(x, np.rad2deg(hand["desired_rad"]), ":", label="原始目标")
        axes[0, 0].plot(x, np.rad2deg(hand["target_rad"]), "o-", label="限速后目标")
        axes[0, 0].plot(x, np.rad2deg(hand["actual_rad"]), "o-", label="实际角度")
        axes[0, 0].set_xticks(x, CHANNEL_LABELS, fontsize=8)
        axes[0, 0].set_ylabel("关节角度（度）")
        axes[0, 0].legend(fontsize=7, ncol=3)
        actual = hand["actuator_torque_nm"]
        axes[0, 1].bar(x - .15, hand["pd_torque_nm"], width=.3, color="#99adc4", label="控制器请求")
        axes[0, 1].bar(x + .15, actual, width=.3,
                       color=["#d64b35" if abs(t) >= 1.39 else "#13889b" for t in actual], label="执行器输出")
        axes[0, 1].axhline(1.4, color="red", ls="--", lw=.7)
        axes[0, 1].axhline(-1.4, color="red", ls="--", lw=.7)
        axes[0, 1].set_xticks(x, CHANNEL_LABELS, fontsize=8)
        axes[0, 1].set_ylabel("关节力矩（N·m）")
        axes[0, 1].legend(fontsize=7)
        hist = list(self.history)
        t = [s["sim_time"] for s in hist]
        axes[1, 0].plot(t, [s["hands"][side]["normal_contact_sum_n"] for s in hist], color="#13889b")
        axes[1, 0].set_ylabel("手与方块法向接触力之和（N）")
        cubes = [s for s in hist if s["cube"] is not None and s["cube"]["bottom_clearance_m"] is not None]
        if cubes:
            tc = [s["sim_time"] for s in cubes]
            axes[1, 1].plot(tc, [s["cube"]["bottom_clearance_m"] * 100 for s in cubes], label="方块底部离桌高度")
            origin = np.asarray(cubes[0]["cube"]["relative_wrist"][side])
            axes[1, 1].plot(tc, [np.linalg.norm(np.asarray(s["cube"]["relative_wrist"][side]) - origin) * 100 for s in cubes], label="相对手腕位移（当前窗口）")
            axes[1, 1].legend(fontsize=7)
        axes[1, 1].set_ylabel("距离（厘米）")
        for ax in axes[1]:
            ax.set_xlabel("仿真时间（秒）")
        for ax in axes.flat:
            ax.grid(alpha=.2)
        self.figure.tight_layout(pad=1.)
        self.canvas.draw()
        return Image.fromarray(np.asarray(self.canvas.buffer_rgba()).copy()).convert("RGB")

    def status(self, sample, side):
        hand = sample["hands"][side]
        age = sample["command_age_s"]
        status = "尚未收到指令" if age is None else ("通信超时，松手" if sample["communication_timeout"] else "持续接收")
        contacts = [c for c in sample["contacts"] if c["side"] == side and c["normal_n"] > 0]
        details = "\n".join(f"{c['other_name']}: 法向 {c['normal_n']:.2f} N / 切向 {c['tangent_n']:.2f} N" for c in contacts[:8]) or "手与方块之间没有承载接触"
        cube = sample["cube"]
        clearance = cube and cube["bottom_clearance_m"]
        height = "n/a" if clearance is None else f"{clearance * 100:.1f} cm"
        return (f"{'右手' if side == 'right' else '左手'} | 仿真 {sample['sim_time']:.2f} 秒 | 回合 {sample['episode']} | 帧 {sample['seq']}\n"
                f"指令：{status} | 最近指令距今：{age if age is None else round(age, 3)} 秒\n"
                f"六通道指令：{hand.get('command', [])}\n"
                f"方块底部离桌高度：{height}\n"
                f"桌面承载：{cube['table_contact'] if cube else '无方块'} | 悬挂辅助：{sample['elastic_band']}\n"
                f"法向接触力之和：{hand['normal_contact_sum_n']:.2f} N（不是向上托举合力）\n\n"
                f"{details}\n\n橙色箭头：方块受到的力；1 N 显示为 2 cm，最长 15 cm。\n"
                f"记录丢帧：{sample['recording_drops']} | 错误：{sample['recording_error']}\n"
                "指令有更新不代表头显追踪有效，追踪状态仍需对照桥接日志。\n"
                "相对手腕位移也包括主动调整位置，不能直接等同于滑落。")

    def dashboard(self, sample, side="right"):
        from PIL import ImageDraw, ImageFont
        image = Image.new("RGB", (1280, 800), "#f4f6fa")
        image.paste(self.render(sample, side), (0, 0))
        font = ImageFont.truetype(str(FONT_PATH), 14) if FONT_PATH.exists() else ImageFont.load_default()
        ImageDraw.Draw(image).multiline_text((650, 12), self.status(sample, side), fill="#17202a", font=font, spacing=4)
        image.paste(self.charts(sample, side), (0, 400))
        return image

    def close(self):
        self.renderer.close()


def run_desktop(endpoint, replay=None, max_runtime=None):
    import tkinter as tk
    from tkinter import ttk
    from PIL import ImageTk
    import zmq

    root = tk.Tk()
    root.title("MuJoCo 抓取诊断 — 只读观察 / Grasp diagnostics")
    side, angle, collision = tk.StringVar(value="right"), tk.StringVar(value="side"), tk.BooleanVar(value=False)
    toolbar = ttk.Frame(root); toolbar.pack(fill="x")
    ttk.Label(toolbar, text="观察手：").pack(side="left")
    ttk.Combobox(toolbar, textvariable=side, values=("left", "right"), state="readonly", width=7).pack(side="left")
    ttk.Combobox(toolbar, textvariable=angle, values=("front", "side", "top", "whole"), state="readonly", width=9).pack(side="left")
    ttk.Checkbutton(toolbar, text="碰撞外形", variable=collision).pack(side="left")
    state = {"sample": None, "display": None, "last_received": 0., "playing": True, "index": 0, "refs": [], "orbit": None, "zoom": 1.}
    rows = read_samples(Path(replay) / "samples.jsonl") if replay else []
    if replay and not rows:
        raise ValueError("recording has no samples")
    context, sub = None, None
    if not replay:
        context = zmq.Context(); sub = context.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b""); sub.setsockopt(zmq.CONFLATE, 1); sub.setsockopt(zmq.LINGER, 0)
        sub.connect(endpoint)
    msg = tk.StringVar(value="等待诊断数据；需使用诊断版仿真入口。")
    ttk.Label(root, textvariable=msg).pack(fill="x")
    upper = ttk.Frame(root); upper.pack(fill="both")
    camera = ttk.Label(upper); camera.pack(side="left")
    def drag(event):
        previous = state.get("mouse")
        if previous is not None:
            az, el = state["orbit"] or {"front": (180, -20), "side": (90, -20), "top": (90, -85), "whole": (125, -20)}[angle.get()]
            state["orbit"] = (az + (event.x - previous[0]) * .6, float(np.clip(el + (event.y - previous[1]) * .4, -89, 89)))
        state["mouse"] = (event.x, event.y)
    camera.bind("<Button-1>", lambda e: state.update(mouse=(e.x, e.y)))
    camera.bind("<B1-Motion>", drag)
    camera.bind("<Button-4>", lambda e: state.update(zoom=max(.35, state["zoom"] / 1.1)))
    camera.bind("<Button-5>", lambda e: state.update(zoom=min(3., state["zoom"] * 1.1)))
    angle.trace_add("write", lambda *a: state.update(orbit=None, zoom=1.))
    ttk.Label(toolbar, text="拖动画面转视角，滚轮缩放").pack(side="right")
    detail = tk.Text(upper, width=70, height=23, font=("monospace", 10), wrap="word")
    detail.pack(side="left", fill="both", expand=True)
    chart = ttk.Label(root); chart.pack()

    def mark(label):
        s = state["sample"]
        if not s or replay:
            return
        path = Path(s["session_dir"]) / "markers.jsonl"
        with path.open("a") as f:
            f.write(json.dumps({"label": label, "wall_time": time.time(), "seq": s["seq"], "sim_time": s["sim_time"], "episode": s["episode"]}) + "\n")
        msg.set(f"已标记 {label}；对应最近显示的采样帧。")

    if not replay:
        for label, key in (("开始试验", "trial_start"), ("抬起", "lift"), ("掉落", "drop"), ("结束试验", "trial_end")):
            ttk.Button(toolbar, text=label, command=lambda k=key: mark(k)).pack(side="left")
    else:
        ttk.Button(toolbar, text="播放/暂停", command=lambda: state.update(playing=not state["playing"])).pack(side="left")
        def seek(value):
            state.update(index=int(float(value)), playing=False)
        tk.Scale(root, from_=0, to=len(rows)-1, orient="horizontal", label="回放帧（仅显示，不重新求解抓取）", command=seek).pack(fill="x")

    def tick():
        try:
            if replay:
                s = dict(rows[state["index"]]); s["model_path"] = str(Path(replay) / "model.mjb")
                state["sample"] = s
                if state["playing"]:
                    # Advance by simulation timestamp; GUI target is 5 FPS.
                    end = s["sim_time"] + .2
                    i = state["index"] + 1
                    while i < len(rows) and rows[i]["episode"] == s["episode"] and rows[i]["sim_time"] < end:
                        i += 1
                    state["index"] = min(i, len(rows)-1)
            else:
                try:
                    state["sample"] = sub.recv_json(zmq.NOBLOCK)
                    state["last_received"] = time.monotonic()
                except zmq.Again:
                    pass
            s = state["sample"]
            if s:
                if state.get("model_path") != s["model_path"]:
                    if state["display"]: state["display"].close()
                    state["display"] = GraspDisplay(s["model_path"])
                    state["model_path"] = s["model_path"]
                display = state["display"]
                photos = [ImageTk.PhotoImage(display.render(s, side.get(), angle.get(), collision.get(), state["orbit"], state["zoom"])),
                          ImageTk.PhotoImage(display.charts(s, side.get()))]
                state["refs"] = photos
                camera.configure(image=photos[0]); chart.configure(image=photos[1])
                detail.delete("1.0", "end"); detail.insert("end", display.status(s, side.get()))
                stale = not replay and time.monotonic() - state["last_received"] > 1
                msg.set(("数据已停止更新 — 显示最后一帧 | " if stale else "") + ("回放 | " if replay else "实时观察 | ") + s["session_dir"])
        except Exception as exc:
            msg.set(f"诊断显示错误：{exc}")
        root.after(200, tick)

    def close():
        if state["display"]: state["display"].close()
        if sub is not None: sub.close(0); context.term()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(100, tick)
    if max_runtime is not None:
        root.after(int(max_runtime * 1000), close)
    root.mainloop()
