"""Scene randomization shared by data generation (domain randomization) and evaluation
(generalization tests). See Perturber for the kinds and ranges. Simulation only.
"""
from __future__ import annotations

import numpy as np

import wipe_teacher as W

PERTURBATIONS = ('light', 'color', 'camera', 'physics', 'region')
REGION_SHIFT = .04  # m; training data used +-0.015


class Perturber:
    """Randomized scene conditions, redrawn every episode or trial.

    light    light intensity and direction, headlight brightness
    color    table, floor, sponge pad and region-marker colors
    camera   head camera shifted +-1.5 cm, tilted up to 3 deg, field of view +-4 deg
    physics  tool mass/inertia x0.6-1.6, pad and table friction x0.5-1.5
    region   both regions shifted up to +-4 cm (training data: +-1.5 cm)
    The first ACT datasets used none of these, so for those models every kind is out of
    distribution. Draws come from their own RNG, so the scene seeds (region side, start pose,
    plan) stay identical to an unperturbed run.
    """
    FIELDS = ('light_diffuse', 'light_pos', 'light_dir', 'geom_rgba', 'site_rgba', 'cam_pos', 'cam_quat',
              'cam_fovy', 'body_mass', 'body_inertia', 'geom_friction')

    def __init__(self, t, kinds, seed):
        m = t.model
        self.t, self.kinds, self.rng = t, set(kinds), np.random.default_rng(seed)
        self.nominal = {f: getattr(m, f).copy() for f in self.FIELDS}
        self.head = (m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy())
        self.geoms = {n: m.geom(n).id for n in ('table_top', 'floor', 'sponge_pad')}
        self.sites = [m.site(f'region_{s}').id for s in W.REGIONS]
        self.cam = m.camera('head_camera').id

    def apply(self):
        """Redraw the conditions; returns a region offset when 'region' is active, else None."""
        mj, m, r = self.t.mj, self.t.model, self.rng
        for f, v in self.nominal.items():
            getattr(m, f)[:] = v
        m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = self.head
        if 'light' in self.kinds:
            m.light_diffuse[:] *= r.uniform(.4, 1.4)
            m.light_pos[:] += r.uniform([-1, -1, -.5], [1, 1, .5], m.light_pos.shape)
            d = m.light_dir + r.uniform(-.4, .4, m.light_dir.shape)
            m.light_dir[:] = d / np.linalg.norm(d, axis=1, keepdims=True)
            k = r.uniform(.5, 1.3)
            m.vis.headlight.ambient[:] = self.head[0] * k
            m.vis.headlight.diffuse[:] = self.head[1] * k
        if 'color' in self.kinds:
            for g in self.geoms.values():
                m.geom_rgba[g, :3] = r.uniform(.15, .95, 3)
            for s in self.sites:
                m.site_rgba[s] = np.r_[r.uniform(.1, .9, 3), r.uniform(.15, .45)]
        if 'camera' in self.kinds:
            m.cam_pos[self.cam] += r.uniform(-.015, .015, 3)
            axis = r.normal(size=3)
            dq, q = np.zeros(4), np.zeros(4)
            mj.mju_axisAngle2Quat(dq, axis / np.linalg.norm(axis), np.radians(r.uniform(0, 3)))
            mj.mju_mulQuat(q, m.cam_quat[self.cam], dq)
            m.cam_quat[self.cam] = q
            m.cam_fovy[self.cam] += r.uniform(-4, 4)
        if 'physics' in self.kinds:
            k = r.uniform(.6, 1.6)
            m.body_mass[self.t.tool] *= k
            m.body_inertia[self.t.tool] *= k
            m.geom_friction[self.geoms['sponge_pad'], 0] *= r.uniform(.5, 1.5)
            m.geom_friction[self.geoms['table_top'], 0] *= r.uniform(.5, 1.5)
        if 'region' in self.kinds:
            return r.uniform(-REGION_SHIFT, REGION_SHIFT, 2)
        return None
