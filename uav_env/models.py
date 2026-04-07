from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List


class UAVAction(Action):
    """
    9D velocity commands: [vx, vy, vz] × 3 agents.
    Each value must be in [-1, 1] (scaled internally by cmd_scale = 8 m/s).
    """
    commands: List[float] = Field(..., min_length=9, max_length=9)


class UAVObservation(Observation):
    """
    48D feature vector: 16 features × 3 agents.

    Per-agent block (16 values):
      [0:3]   rel_pos  — target_pos - uav_pos (metres)
      [3:6]   rel_vel  — target_vel - uav_vel (m/s)
      [6:9]   uav_vel  — own velocity (m/s)
      [9:12]  wind     — wind vector (m/s)
      [12:15] nfz_vec  — vector to nearest NFZ center (metres)
      [15]    d_nfz    — scalar distance to NFZ surface (metres)

    Reward returned by step() is always normalised to [0.0, 1.0].
    """
    features: List[float] = Field(..., min_length=48, max_length=48)