"""Manual tool-only reset, applied on the simulation thread between steps."""

from contextlib import contextmanager
from threading import Event

import mujoco
import numpy as np


def reset_key_callback(request, previous=None):
    def on_key(key):
        if key in (ord("R"), ord("r")):
            request()  # GUI thread only queues a request; never edits MjData.
        elif previous is not None:
            previous(key)
    return on_key


@contextmanager
def viewer_reset_key(request):
    """Add R while the upstream constructor creates this task's viewer."""
    import mujoco.viewer
    original = mujoco.viewer.launch_passive

    def launch(*args, **kwargs):
        kwargs["key_callback"] = reset_key_callback(request, kwargs.get("key_callback"))
        return original(*args, **kwargs)

    mujoco.viewer.launch_passive = launch
    try:
        yield
    finally:
        mujoco.viewer.launch_passive = original


class PaintToolReset:
    def __init__(self, model, pending=None):
        self.model = model
        free = model.joint("paint_tool_free")
        hinge = model.joint("paint_roller_hinge")
        if free.type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Tool reset requires paint_tool_free to be a free joint")
        self.qpos = np.r_[np.arange(free.qposadr[0], free.qposadr[0] + 7), hinge.qposadr[0]]
        self.dofs = np.r_[np.arange(free.dofadr[0], free.dofadr[0] + 6), hinge.dofadr[0]]
        tool = model.body("paint_tool_body").id
        self.bodies = set()
        for body in range(1, model.nbody):
            ancestor = body
            while ancestor and ancestor != tool:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == tool:
                self.bodies.add(body)
        self.geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] in self.bodies}
        self._pending = pending if pending is not None else Event()
        self._scratch = mujoco.MjData(model)

    def request(self):
        self._pending.set()

    def apply_pending(self, data):
        if not self._pending.is_set():
            return None
        self._pending.clear()
        self._scratch.qpos[:] = data.qpos
        self._scratch.qpos[self.qpos] = self.model.qpos0[self.qpos]
        self._scratch.qvel[:] = 0
        mujoco.mj_forward(self.model, self._scratch)
        for contact in self._scratch.contact:
            if ((int(contact.geom1) in self.geoms) != (int(contact.geom2) in self.geoms)
                    and contact.dist < -1e-4):
                print("[paint_reset] Return position occupied. Move hands clear of the holder, then press R again.", flush=True)
                return None
        event = {"reason": "manual_tool_reset", "sim_time": float(data.time),
                 "tool_qpos_before": data.qpos[self.qpos].tolist(),
                 "tool_qpos_after": self.model.qpos0[self.qpos].tolist()}
        data.qpos[self.qpos] = self.model.qpos0[self.qpos]
        for field in ("qvel", "qacc_warmstart", "qfrc_applied"):
            getattr(data, field)[self.dofs] = 0
        data.xfrc_applied[sorted(self.bodies)] = 0
        print("[paint_reset] Brush returned to its holder; robot, calibration and connection kept.", flush=True)
        return event
