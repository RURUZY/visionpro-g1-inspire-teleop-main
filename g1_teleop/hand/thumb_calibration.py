"""Fit only the two thumb channels, preserving effective finger calibration."""

from copy import deepcopy
import numpy as np

from g1_teleop.hand.mapping import HandCalibration


def fit_thumb_calibration(base, samples):
    channels = deepcopy(base.channels)
    report = {}
    for channel in ("thumb_bend", "thumb_rot"):
        values = {}
        for pose in ("open", "close"):
            a = np.asarray(samples[f"{channel}_{pose}"], dtype=float)
            if a.ndim != 1 or len(a) < 30 or not np.isfinite(a).all():
                raise ValueError(f"{channel}/{pose}: need 30 valid, distinct samples")
            values[pose] = {"median": float(np.median(a)),
                            "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
                            "count": len(a)}
        span = abs(values["close"]["median"] - values["open"]["median"])
        # Degrees: reject indistinguishable poses and highly unstable captures.
        if span < max(5.0, 3.0 * max(v["iqr"] for v in values.values())):
            raise ValueError(f"{channel}: poses too similar or unstable; repeat capture")
        channels[channel] = {pose: values[pose]["median"] for pose in ("open", "close")}
        report[channel] = values
    return HandCalibration(channels), report
