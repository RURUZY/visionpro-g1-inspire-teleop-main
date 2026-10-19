#!/usr/bin/env python3
"""Printable handle + backplate of the simulated sponge tool (sponge_demo.py, 'sponge_tool').

Sim geometry (MuJoCo sizes are half sizes):
  short_handle      capsule r 12 mm, 100 mm segment -> 24 mm diameter, 124 mm tall above the plate
  sponge_backplate  box 120 x 70 x 12 mm
  sponge_pad        box 120 x 70 x 24 mm (a real sponge, cut and glued under the plate)
The handle stands vertically at the plate centre. A 5 mm fillet (not in the sim) strengthens the
joint. Writes one watertight binary STL in millimetres: sponge_tool_handle_plate.stl.
"""
import struct
import numpy as np

PLATE = (120., 70., 12.)      # x, y, thickness
R, TOP, FILLET = 12., 124., 5.  # handle radius, height of the rounded top above the plate, fillet
N = 96


def profile():
    pts = [(R + FILLET - FILLET * np.sin(a), FILLET - FILLET * np.cos(a)) for a in np.linspace(0, np.pi / 2, 9)]
    pts += [(R * np.cos(a), TOP - R + R * np.sin(a)) for a in np.linspace(0, np.pi / 2, 17)]
    return pts  # last point is the apex (r = 0)


def build():
    a, b, t = PLATE[0] / 2, PLATE[1] / 2, PLATE[2]
    corners = [np.arctan2(sy * b, sx * a) % (2 * np.pi) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    ang = np.unique(np.r_[np.linspace(0, 2 * np.pi, N, endpoint=False), corners])
    c, s = np.cos(ang), np.sin(ang)
    reach = np.minimum(a / np.maximum(abs(c), 1e-12), b / np.maximum(abs(s), 1e-12))
    rect = np.c_[reach * c, reach * s]
    tris = []
    n = len(ang)
    ring = lambda r, z: np.c_[r * c, r * s, np.full(n, z)]
    top, bot = np.c_[rect, np.zeros(n)], np.c_[rect, np.full(n, -t)]
    centre = np.array([0., 0., -t])
    prof = profile()
    rings = [ring(r, z) for r, z in prof[:-1]]
    apex = np.array([0., 0., prof[-1][1]])
    for i in range(n):
        j = (i + 1) % n
        tris.append((centre, bot[j], bot[i]))                              # bottom
        tris += [(bot[i], bot[j], top[j]), (bot[i], top[j], top[i])]       # plate sides
        tris += [(rings[0][i], top[i], top[j]), (rings[0][i], top[j], rings[0][j])]  # plate top
        for lo, up in zip(rings, rings[1:]):                               # fillet + handle
            tris += [(lo[i], lo[j], up[j]), (lo[i], up[j], up[i])]
        tris.append((rings[-1][i], rings[-1][j], apex))                     # rounded top
    return np.array(tris)


def check(tris):
    edges = {}
    key = lambda p: tuple(np.round(p, 6))
    for tri in tris:
        for k in range(3):
            e = (key(tri[k]), key(tri[(k + 1) % 3]))
            edges[e] = edges.get(e, 0) + 1
    assert all(v == 1 and edges.get((e[1], e[0])) == 1 for e, v in edges.items()), 'mesh not closed'
    vol = sum(np.dot(p, np.cross(q, r)) for p, q, r in tris) / 6
    assert vol > 0, 'normals point inwards'
    return vol


def write_stl(path, tris):
    with open(path, 'wb') as f:
        f.write(b'sponge tool handle + backplate (mm)'.ljust(80, b' '))
        f.write(struct.pack('<I', len(tris)))
        for p, q, r in tris:
            nrm = np.cross(q - p, r - p)
            nrm = nrm / (np.linalg.norm(nrm) or 1)
            f.write(struct.pack('<12fH', *nrm, *p, *q, *r, 0))


if __name__ == '__main__':
    tris = build()
    vol = check(tris)
    write_stl('sponge_tool_handle_plate.stl', tris)
    print(f'{len(tris)} triangles, solid volume {vol / 1000:.1f} cm^3 '
          f'(PLA solid ~{vol / 1000 * 1.24:.0f} g)')
