"""Exercise reconnect/cancellation without contacting a headset or opening sockets."""

import threading
import time
import sys
from types import SimpleNamespace

import pytest

from g1_teleop.bridge.avp_connection import LocalTrackingTransport, managed_streamer_class


class BaseStreamer:
    def __init__(self, ip, **kwargs):
        self.ip = ip
        self._cross_network_mode = ip == "ROOM-CODE"
        self.latest = None
        self.info_server = None
        self.cleaned = 0
        self._server_running = False
        self._webrtc_connected = False
        if not self._cross_network_mode:
            self._start_hand_tracking()

    def get_local_ip(self):
        return "127.0.0.1"

    def _process_hand_update(self, update, source="grpc"):
        self.latest = update

    def get_latest(self, *args, **kwargs):
        return self.latest

    def cleanup(self):
        self.cleaned += 1


class Call:
    def __init__(self, frames=(), end=False):
        self.frames, self.end = frames, end
        self.cancelled = threading.Event()

    def __iter__(self):
        yield from self.frames
        if not self.end:
            self.cancelled.wait(3)

    def cancel(self):
        self.cancelled.set()


class Channel:
    closed = False

    def close(self):
        self.closed = True


class Transport:
    def __init__(self, calls):
        self.calls = calls
        self.opened = []
        self.advertisements = []
        self.advertise_failures = 0

    def open(self, *args):
        call = self.calls[min(len(self.opened), len(self.calls) - 1)]
        channel = Channel()
        self.opened.append((channel, call))
        return channel, call

    def advertise(self, endpoint, fields, timeout):
        self.advertisements.append((endpoint, fields.copy(), timeout))
        if self.advertise_failures:
            self.advertise_failures -= 1
            raise TimeoutError("headset not ready")


class LiveCall(Call):
    def __init__(self, ready=None):
        super().__init__()
        self.ready = ready

    def __iter__(self):
        while not self.cancelled.wait(.01):
            if self.ready is None or self.ready.is_set():
                yield {"head": object()}


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.005)
    assert predicate()


def test_retries_normal_end_and_stalled_first_frame_then_cleans_up():
    calls = [Call(end=True), Call(), Call([{"head": "fresh"}])]
    transport = Transport(calls)
    streamer = managed_streamer_class(BaseStreamer)(
        "127.0.0.1", tracking_transport=transport, tracking_timeout=.08, tracking_retry=.01)
    try:
        wait_until(lambda: streamer.get_latest() is not None)
        assert streamer.get_latest() == {"head": "fresh"}
        assert len(transport.opened) == 3
        assert all(channel.closed and call.cancelled.is_set()
                   for channel, call in transport.opened[:2])
        with streamer._tracking_guard:
            streamer._tracking_received = time.monotonic() - 1
        assert streamer.get_latest() is None
    finally:
        streamer.cleanup()
    assert all(channel.closed and call.cancelled.is_set() for channel, call in transport.opened)
    assert not streamer._tracking_thread.is_alive()
    assert not streamer._tracking_watchdog.is_alive()
    streamer.cleanup()
    assert streamer.cleaned == 1


def test_can_quit_while_waiting_for_first_frame():
    transport = Transport([Call()])
    streamer = managed_streamer_class(BaseStreamer)("127.0.0.1", tracking_transport=transport)
    wait_until(lambda: streamer._tracking_call is not None)
    assert streamer.get_latest() is None
    streamer.cleanup()
    assert not streamer._tracking_thread.is_alive()
    assert transport.opened[0][0].closed


def test_cross_network_mode_keeps_sdk_connection_behavior():
    transport = Transport([])
    streamer = managed_streamer_class(BaseStreamer)("ROOM-CODE", tracking_transport=transport)
    streamer.latest = {"head": "sdk data"}
    assert streamer.get_latest() == streamer.latest
    assert not transport.opened
    streamer.cleanup()


def test_video_waits_for_late_headset_and_listening_server_then_reannounces_on_reconnect():
    ready = threading.Event()
    calls = [LiveCall(ready), LiveCall()]
    transport = Transport(calls)
    streamer = managed_streamer_class(BaseStreamer)(
        "127.0.0.1", tracking_transport=transport, tracking_retry=.01)
    try:
        streamer._send_webrtc_info_via_grpc("192.168.2.22", 9999, video_enabled=True)
        assert streamer._video_announcement["m00"] == 999.0
        assert streamer._video_announcement["m21"] == 1.0
        time.sleep(.12)
        assert not transport.advertisements
        ready.set()
        wait_until(lambda: streamer.get_latest() is not None)
        time.sleep(.12)
        assert not transport.advertisements  # Server still not listening.
        streamer._server_running = True
        wait_until(lambda: len(transport.advertisements) == 1)
        assert transport.advertisements[0][1] == {
            "m00": 999.0, "m01": 192.0, "m02": 168.0, "m03": 2.0,
            "m10": 22.0, "m11": 9999.0, "m12": 0.0, "m13": 0.0,
            "m20": 0.0, "m21": 1.0, "m22": 0.0}
        time.sleep(.12)
        assert len(transport.advertisements) == 1  # Successful announcement is one-shot.
        calls[0].cancel()
        wait_until(lambda: len(transport.advertisements) == 2)
    finally:
        streamer.cleanup()
    assert not streamer._video_advertiser.is_alive()


def test_video_retries_failed_announcement_without_restarting_tracking():
    transport = Transport([LiveCall()])
    transport.advertise_failures = 1
    streamer = managed_streamer_class(BaseStreamer)(
        "127.0.0.1", tracking_transport=transport, video_retry=.01)
    try:
        streamer._server_running = True
        streamer._send_webrtc_info_via_grpc("127.0.0.1", 9999, video_enabled=True)
        wait_until(lambda: len(transport.advertisements) == 2)
        assert len(transport.opened) == 1
        assert all(timeout == 2.0 for _, _, timeout in transport.advertisements)
    finally:
        streamer.cleanup()


def test_video_does_not_interrupt_connected_media_and_stops_when_exiting():
    transport = Transport([LiveCall()])
    streamer = managed_streamer_class(BaseStreamer)("127.0.0.1", tracking_transport=transport)
    try:
        streamer._server_running = streamer._webrtc_connected = True
        streamer._send_webrtc_info_via_grpc("127.0.0.1", 9999, video_enabled=True)
        wait_until(lambda: streamer.get_latest() is not None)
        time.sleep(.15)
        assert not transport.advertisements
    finally:
        streamer.cleanup()
    assert not streamer._video_advertiser.is_alive()


@pytest.mark.parametrize("fails", [False, True])
def test_video_rpc_sends_full_flags_with_deadline_and_always_closes(monkeypatch, fails):
    channel = Channel()
    request_args = []
    ready_timeouts = []

    class Rpc:
        cancelled = False

        def __next__(self):
            if fails:
                raise TimeoutError("reply timed out")
            return object()

        def cancel(self):
            self.cancelled = True

    rpc = Rpc()

    def stream_updates(message, timeout):
        request_args.append((message, timeout))
        return rpc

    sdk = SimpleNamespace(
        LIBRARY_VERSION_CODE=251,
        handtracking_pb2=SimpleNamespace(HandUpdate=lambda: SimpleNamespace(Head=SimpleNamespace())),
        handtracking_pb2_grpc=SimpleNamespace(HandTrackingServiceStub=lambda _: SimpleNamespace(StreamHandUpdates=stream_updates)))
    grpc = SimpleNamespace(
        insecure_channel=lambda _: channel,
        channel_ready_future=lambda _: SimpleNamespace(result=lambda timeout: ready_timeouts.append(timeout)))
    monkeypatch.setitem(sys.modules, "grpc", grpc)
    monkeypatch.setitem(sys.modules, "avp_stream", SimpleNamespace(streamer=sdk))
    fields = {"m00": 999.0, "m21": 1.0}
    if fails:
        with pytest.raises(TimeoutError):
            LocalTrackingTransport().advertise("127.0.0.1", fields)
    else:
        LocalTrackingTransport().advertise("127.0.0.1", fields)
    message, timeout = request_args[0]
    assert message.Head.m00 == 999.0 and message.Head.m21 == 1.0
    assert message.Head.m30 == 251.0
    assert ready_timeouts == [2.0] and 0 < timeout <= 2.0
    assert rpc.cancelled and channel.closed
