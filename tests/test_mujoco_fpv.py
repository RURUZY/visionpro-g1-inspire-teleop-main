"""Separate camera availability from headset delivery without opening connections."""

import sys
from types import SimpleNamespace

import numpy as np

from g1_teleop.sim.mujoco_fpv import MujocoFpvStreamer


def test_camera_buffers_before_tracking_and_reports_video_connection(monkeypatch, capsys):
    image = np.full((2, 4, 3), [10, 20, 30], dtype=np.uint8)

    class Camera:
        def __init__(self, **kwargs):
            self.closed = False

        def read(self, **kwargs):
            return {"images": {"ego_view": image}}

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "gear_sonic.camera.composed_camera",
                        SimpleNamespace(ComposedCameraClientSensor=Camera))
    frames = []
    streamer = SimpleNamespace(
        latest=None, _webrtc_connected=False,
        configure_video=lambda **kwargs: None,
        start_webrtc=lambda **kwargs: True,
        update_frame=frames.append,
    )
    fpv = MujocoFpvStreamer(streamer, video_size="4x2")
    assert fpv.push_latest()
    np.testing.assert_array_equal(frames[0], image[:, :, ::-1])
    output = capsys.readouterr().out
    assert "First camera frame buffered" in output
    assert "Headset video connection established" not in output

    streamer._webrtc_connected = True
    fpv.push_latest()
    assert "Headset video connection established" in capsys.readouterr().out
    fpv.push_latest()
    assert "Headset video connection established" not in capsys.readouterr().out

    streamer._webrtc_connected = False
    fpv.push_latest()
    assert "Waiting for headset video connection" in capsys.readouterr().out
    fpv.close()
    assert fpv._client.closed
