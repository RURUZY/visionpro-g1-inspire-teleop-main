"""Explicit walking permission while arm/hand teleoperation stays live."""

from dataclasses import dataclass

import numpy as np

from g1_teleop.locomotion.head import (
    HeadLocomotionState,
    SonicPlannerCommand,
    horizontal_heading_from_calib_rot,
    reset_head_locomotion_state,
    wrap_to_pi,
)


@dataclass
class LocomotionClutch:
    enabled: bool = False
    walking: bool = False
    facing_angle: float = 0.0

    @property
    def locked(self) -> bool:
        return self.enabled and not self.walking

    def lock(self, facing_angle: float) -> None:
        self.walking = False
        self.facing_angle = wrap_to_pi(float(facing_angle))

    def unlock(self, head_state: HeadLocomotionState, head_pose: np.ndarray) -> bool:
        if not self.locked:
            return False  # Repeated B key events must not repeatedly reset motion.
        # Re-anchor the operator's current look direction to the held robot
        # direction. Keep the robot IMU origin in the existing planner frame.
        # Discard head displacement/velocity accumulated while manipulating.
        imu_origin = head_state.robot_base_yaw_at_calib
        head_yaw = horizontal_heading_from_calib_rot(head_pose[:3, :3])
        reset_head_locomotion_state(
            head_state, calib_yaw=wrap_to_pi(head_yaw - self.facing_angle))
        head_state.facing_angle = self.facing_angle
        head_state.robot_base_yaw_at_calib = imu_origin
        self.walking = True
        return True

    def filter(self, command: SonicPlannerCommand | None) -> SonicPlannerCommand | None:
        if not self.locked:
            return command
        return SonicPlannerCommand(
            mode=0,
            movement=np.zeros(3, dtype=np.float64),
            facing=np.array([np.cos(self.facing_angle), np.sin(self.facing_angle), 0.0]),
            speed=-1.0,
            height=-1.0,
        )
