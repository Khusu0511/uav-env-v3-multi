from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Optional, Dict, Any


class UAVAction(Action):
    """
    9D velocity commands: [vx, vy, vz] × 3 agents.
    Each value in [-1, 1], scaled internally by cmd_scale = 8 m/s.

    WHY default_factory:
    The OpenEnv web playground UI sends an empty JSON body ({}) when the
    user clicks the Step button without entering commands. Without a default,
    Pydantic raises:
        "1 validation error for UAVAction — commands — Field required"
    The default of 9 zeros means an empty-body Step is treated as a
    "hold position" command — safe and valid.
    """
    commands: List[float] = Field(
        default_factory=lambda: [0.0] * 9,
        min_length=9,
        max_length=9,
        description="9 velocity commands [vx,vy,vz] × 3 UAVs, each in [-1, 1]",
    )


class UAVObservation(Observation):
    """
    48D feature vector: 16 features × 3 agents.

    Per-agent block (16 values):
      [0:3]   rel_pos   — target_pos - uav_pos (metres)
      [3:6]   rel_vel   — target_vel - uav_vel (m/s)
      [6:9]   uav_vel   — own velocity (m/s)
      [9:12]  wind      — wind vector (m/s)
      [12:15] nfz_vec   — vector to nearest NFZ center (metres)
      [15]    d_nfz     — scalar distance to NFZ surface (metres)

    All extra fields are Optional so the OpenEnv create_app() framework
    can attach episode_id, reward, done etc without AttributeError.
    Reward is always normalised to [0.0, 1.0] by the environment.
    """
    features:   List[float]              = Field(default_factory=list)
    episode_id: Optional[str]            = Field(default=None)
    reward:     Optional[float]          = Field(default=0.0)
    done:       Optional[bool]           = Field(default=False)
    metadata:   Optional[Dict[str, Any]] = Field(default_factory=dict)
    step_count: Optional[int]            = Field(default=0)