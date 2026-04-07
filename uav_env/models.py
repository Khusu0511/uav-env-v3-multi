from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Optional


class UAVAction(Action):
    """
    9D velocity commands: [vx, vy, vz] x 3 agents.
    Each value must be in [-1, 1] (scaled internally by cmd_scale = 8 m/s).
    Defaults to all-zeros (hover) when no commands are supplied.
    """
    commands: List[float] = Field(
        default_factory=lambda: [0.0] * 9,
        min_length=9,
        max_length=9,
    )


class UAVObservation(Observation):
    """
    48D feature vector: 16 features x 3 agents.

    Per-agent block (16 values):
      [0:3]  rel_pos  -- target_pos - uav_pos  (metres)
      [3:6]  rel_vel  -- target_vel - uav_vel  (m/s)
      [6:9]  uav_vel  -- own velocity           (m/s)
      [9:12] wind     -- wind vector             (m/s)
      [12:15] nfz_vec -- vector to nearest NFZ center (metres)
      [15]   d_nfz   -- scalar distance to NFZ surface (metres)

    reward:     raw step reward (sum across 3 agents, unclipped).
    done:       always False -- episodes are time-limited by MAX_STEPS.
    episode_id: stamped by openenv-core after reset().
    step_count: stamped by openenv-core after each step(); must be int.
    """
    features: List[float] = Field(..., min_length=48, max_length=48)
    reward: float = 0.0
    done: bool = False
    episode_id: Optional[str] = None
    step_count: int = 0          # int (not Optional) -- ActionLog requires it