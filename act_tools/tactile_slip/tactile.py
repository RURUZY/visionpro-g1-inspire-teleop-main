"""Simulated tactile readout for the Inspire hand model, shared by the teacher variants.

The model has 17 contact pads that match the 17 tactile groups of the real Inspire FTP hand
(per finger: 1 = proximal pad, 2 = middle, 3 = tip; thumb 1-4; one palm pad). Each reading is
the summed normal contact force on that pad. The real hand reports taxel arrays per group, so
these are an approximation to compare against after calibration, not a sensor model.
"""
from __future__ import annotations

import numpy as np

TACTILE_BODIES = (
    [f'right_{f}_force_sensor_{k}' for f in ('little', 'ring', 'middle', 'index') for k in (1, 2, 3)]
    + [f'right_thumb_force_sensor_{k}' for k in (1, 2, 3, 4)] + ['right_palm_force_sensor'])
HAND_ACTIVE = ('little_1', 'ring_1', 'middle_1', 'index_1', 'thumb_2', 'thumb_1')

FEATURES = {
    'observation.tactile': {'dtype': 'float32', 'shape': (17,),
                            'names': [b.replace('right_', '') + '.normal_force_N' for b in TACTILE_BODIES]},
    'observation.hand_force': {'dtype': 'float32', 'shape': (6,),
                               'names': [f'right_hand_{j}.actuator_torque_Nm' for j in HAND_ACTIVE]},
    # Privileged simulation truth: tool pose in the palm frame. Labels only, never a policy input.
    'privileged.tool_in_palm': {'dtype': 'float32', 'shape': (6,),
                                'names': ['x', 'y', 'z', 'rotvec_x', 'rotvec_y', 'rotvec_z']},
}


class TactileMixin:
    """Mix into a WipeTeacher subclass; needs self.model/data/mj, relative_tool(), hand_act."""

    def setup_tactile(self):
        m = self.model
        self.tactile_index = {m.body(b).id: i for i, b in enumerate(TACTILE_BODIES)}
        self._force6 = np.zeros(6)

    def tactile(self):
        m, d = self.model, self.data
        out = np.zeros(len(TACTILE_BODIES), np.float32)
        for i in range(d.ncon):
            c = d.contact[i]
            for b in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                k = self.tactile_index.get(b)
                if k is not None:
                    self.mj.mj_contactForce(m, d, i, self._force6)
                    out[k] += abs(self._force6[0])
        return out

    def tool_in_palm(self):
        pos, rot = self.relative_tool()
        quat, rotvec = np.zeros(4), np.zeros(3)
        self.mj.mju_mat2Quat(quat, rot.flatten())
        self.mj.mju_quat2Vel(rotvec, quat, 1.)
        return np.r_[pos, rotvec].astype(np.float32)

    def extra_obs(self):
        return {'observation.tactile': self.tactile(),
                'observation.hand_force': self.data.actuator_force[self.hand_act].astype(np.float32),
                'privileged.tool_in_palm': self.tool_in_palm()}
