"""Manage local Tracking Streamer connections without restarting the headset app.

The installed avp_stream 2.51 blocks construction until the first tracking frame,
does not retain/cancel its local gRPC call on cleanup, and exits its reader when
a stream ends normally. Keep its pose decoding and cross-network behavior, but
own the local connection lifetime here.
"""

import threading
import time


class LocalTrackingTransport:
    def open(self, endpoint, local_ip, webrtc_info=None):
        import grpc
        from avp_stream import streamer as sdk

        request = sdk.handtracking_pb2.HandUpdate()
        request.Head.m00 = 888.0
        for field, part in zip(("m01", "m02", "m03", "m10"), local_ip.split(".")):
            setattr(request.Head, field, float(part))
        request.Head.m30 = float(sdk.LIBRARY_VERSION_CODE)
        metadata = None
        if webrtc_info is not None:
            metadata = [("webrtc-info", "|".join(str(webrtc_info[field]) for field in
                         ("m01", "m02", "m03", "m10", "m11")))]
        channel = grpc.insecure_channel(f"{endpoint}:12345")
        try:
            grpc.channel_ready_future(channel).result(timeout=2)
            stub = sdk.handtracking_pb2_grpc.HandTrackingServiceStub(channel)
            return channel, stub.StreamHandUpdates(request, metadata=metadata)
        except Exception:
            channel.close()
            raise

    def advertise(self, endpoint, fields, timeout=2.0):
        """Send the full video announcement with a bounded total RPC lifetime."""
        import grpc
        from avp_stream import streamer as sdk

        request = sdk.handtracking_pb2.HandUpdate()
        for field, value in fields.items():
            setattr(request.Head, field, value)
        request.Head.m30 = float(sdk.LIBRARY_VERSION_CODE)
        channel = grpc.insecure_channel(f"{endpoint}:12345")
        call = None
        deadline = time.monotonic() + timeout
        try:
            grpc.channel_ready_future(channel).result(timeout=timeout)
            stub = sdk.handtracking_pb2_grpc.HandTrackingServiceStub(channel)
            call = stub.StreamHandUpdates(request, timeout=max(0.01, deadline - time.monotonic()))
            next(call)
        finally:
            if call is not None:
                call.cancel()
            channel.close()


def managed_streamer_class(base):
    class ManagedVisionProStreamer(base):
        def __init__(self, *args, tracking_transport=None, tracking_timeout=5.0,
                     tracking_retry=0.75, video_retry=2.0, **kwargs):
            self._tracking_transport = tracking_transport or LocalTrackingTransport()
            self._tracking_timeout = tracking_timeout
            self._tracking_retry = tracking_retry
            self._video_retry = video_retry
            self._tracking_stop = threading.Event()
            self._tracking_guard = threading.RLock()
            self._tracking_call = self._tracking_channel = None
            self._tracking_progress = time.monotonic()
            self._tracking_received = None
            self._tracking_thread = self._tracking_watchdog = None
            self._tracking_generation = 0
            self._video_announcement = None
            self._video_revision = 0
            self._video_advertiser = None
            self._managed_closed = False
            super().__init__(*args, **kwargs)

        def _start_hand_tracking(self):
            # Let the bridge start and handle quit/stop while the headset connects.
            print(f"[avp_connection] Connecting to {self.ip}; keep Tracking Streamer running. Automatic retry enabled.", flush=True)
            self._tracking_thread = threading.Thread(target=self.stream, daemon=True)
            self._tracking_watchdog = threading.Thread(target=self._watch_tracking, daemon=True)
            self._tracking_thread.start()
            self._tracking_watchdog.start()

        def _watch_tracking(self):
            while not self._tracking_stop.wait(min(0.25, self._tracking_timeout / 4)):
                with self._tracking_guard:
                    call = self._tracking_call
                    stalled = time.monotonic() - self._tracking_progress > self._tracking_timeout
                if call is not None and stalled:
                    # Cancellation interrupts a stream that is connected but has
                    # never delivered a frame (or stopped delivering frames).
                    call.cancel()

        def _process_hand_update(self, update, source="grpc"):
            previous = self.latest
            super()._process_hand_update(update, source=source)
            if self.latest is not previous:
                with self._tracking_guard:
                    first = self._tracking_received is None
                    self._tracking_received = self._tracking_progress = time.monotonic()
                    if first:
                        self._tracking_generation += 1
                if first:
                    print("[avp_connection] Fresh tracking received.", flush=True)

        def stream(self):
            if self._cross_network_mode:
                return super().stream()
            while not self._tracking_stop.is_set():
                channel = call = None
                try:
                    channel, call = self._tracking_transport.open(
                        self.ip, self.get_local_ip(), self._video_announcement)
                    with self._tracking_guard:
                        self._tracking_channel, self._tracking_call = channel, call
                        self._tracking_progress = time.monotonic()
                        self._tracking_received = None
                    if self._tracking_stop.is_set():
                        break
                    for response in call:
                        if self._tracking_stop.is_set():
                            break
                        self._process_hand_update(response, source="grpc")
                except Exception as exc:
                    if not self._tracking_stop.is_set():
                        print(f"[avp_connection] Tracking reconnect: {type(exc).__name__}.", flush=True)
                finally:
                    if call is not None:
                        call.cancel()
                    if channel is not None:
                        channel.close()
                    with self._tracking_guard:
                        self._tracking_call = self._tracking_channel = None
                        self._tracking_received = None
                if self._tracking_stop.wait(self._tracking_retry):
                    break

        def _send_webrtc_info_via_grpc(self, host, port, stereo_video=False,
                                       stereo_audio=False, audio_enabled=False,
                                       video_enabled=False, sim_enabled=False):
            if self._cross_network_mode:
                return super()._send_webrtc_info_via_grpc(
                    host, port, stereo_video, stereo_audio, audio_enabled,
                    video_enabled, sim_enabled)
            # SDK 2.51 otherwise announces only once, before the video server is
            # listening, and loses the announcement if the headset starts late.
            fields = {"m00": 999.0, "m11": float(port),
                      "m12": float(stereo_video), "m13": float(stereo_audio),
                      "m20": float(audio_enabled), "m21": float(video_enabled),
                      "m22": float(sim_enabled)}
            fields.update(zip(("m01", "m02", "m03", "m10"), map(float, host.split("."))))
            with self._tracking_guard:
                self._video_announcement = fields
                self._video_revision += 1
                if self._video_advertiser is None:
                    self._video_advertiser = threading.Thread(
                        target=self._advertise_video, daemon=True)
                    self._video_advertiser.start()
            print("[avp_connection] Video enabled; waiting for fresh tracking to notify the headset.", flush=True)

        def _advertise_video(self):
            sent = None
            retry_after = 0.0
            while not self._tracking_stop.wait(0.1):
                now = time.monotonic()
                with self._tracking_guard:
                    fields = self._video_announcement
                    received = self._tracking_received
                    generation = (self._tracking_generation, self._video_revision)
                if (fields is None or received is None or now - received > 0.5
                        or not getattr(self, "_server_running", False)
                        or generation == sent or now < retry_after):
                    continue
                if getattr(self, "_webrtc_connected", False):
                    # A tracking reconnect must not interrupt an existing video.
                    sent = generation
                    continue
                try:
                    self._tracking_transport.advertise(self.ip, fields, timeout=2.0)
                    sent = generation
                    print("[avp_connection] Video connection info sent to headset.", flush=True)
                except Exception as exc:
                    if not self._tracking_stop.is_set():
                        print(f"[avp_connection] Video notification retry: {type(exc).__name__}.", flush=True)
                    retry_after = time.monotonic() + self._video_retry

        def get_latest(self, *args, **kwargs):
            if not self._cross_network_mode:
                with self._tracking_guard:
                    received = self._tracking_received
                if received is None or time.monotonic() - received > 0.5:
                    return None  # Do not replay a frozen tracking pose as fresh input.
            return super().get_latest(*args, **kwargs)

        def cleanup(self):
            if self._managed_closed:
                return
            self._managed_closed = True
            self._tracking_stop.set()
            with self._tracking_guard:
                call, channel = self._tracking_call, self._tracking_channel
            if call is not None:
                call.cancel()
            if channel is not None:
                channel.close()
            for thread in (self._tracking_thread, self._tracking_watchdog, self._video_advertiser):
                if thread is not None and thread is not threading.current_thread():
                    thread.join(timeout=2.5)
            try:
                super().cleanup()
            finally:
                server = getattr(self, "info_server", None)
                if server is not None:
                    server.shutdown()
                    server.server_close()
                    self.info_server = None

    return ManagedVisionProStreamer
