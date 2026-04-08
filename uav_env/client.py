# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""UAV Fleet Tracking Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from models import UAVAction, UAVObservation
except ImportError:
    from .models import UAVAction, UAVObservation


class UavEnv(
    EnvClient[UAVAction, UAVObservation, State]
):
    """
    Client for the UAV Fleet Tracking Environment.
    Example (sync):
        with UavEnv(base_url="http://localhost:8000").sync() as client:
            result = client.reset()
            print(result.observation.features[:3])  # rel_pos of UAV 1
            action = UAVAction(commands=[0.5, -0.3, 0.1,
                                          0.0,  0.8, -0.2,
                                         -0.4,  0.1,  0.6])
            result = client.step(action)
            print(result.reward)
    Example (async):
        async with UavEnv(base_url="http://localhost:8000") as client:
            result = await client.reset()
            result = await client.step(UAVAction(commands=[0.0]*9))
    """

    def _step_payload(self, action: UAVAction) -> Dict:
        """Convert UAVAction to JSON payload for the /step WebSocket message."""
        return {"commands": action.commands}

    def _parse_result(self, payload: Dict) -> StepResult[UAVObservation]:
        """
        Parse server response into StepResult[UAVObservation].
        Handles both flat payloads (fields at top level) and nested payloads
        (fields under "observation" key), reading all required log fields:
          done, reward, metadata, features, episode_id, step_count
        """
        obs_data = payload.get("observation", {})

        # Support flat (top-level) and nested (under "observation") layouts
        def _get(key, default):
            return obs_data.get(key, payload.get(key, default))

        features   = _get("features",   [0.0] * 48)
        reward     = float(_get("reward",  1e-5 ))
        done       = bool(_get("done",       False))
        episode_id = _get("episode_id",  "uav_episode")
        step_count = int(_get("step_count",  0))
        metadata   = _get("metadata",    None)

        _safe_reward = float(np.clip(reward, 1e-5, 0.99999))

        observation = UAVObservation(
            features=features,
            reward=_safe_reward,
            done=done,
            episode_id=episode_id,
            step_count=step_count,
            metadata=metadata,
        )

        return StepResult(
            observation=observation,
            reward=_safe_reward,
            done=done,
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id", "uav_episode"),
            step_count=payload.get("step_count", 0),
        )