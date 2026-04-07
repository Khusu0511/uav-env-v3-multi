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

from .models import UAVAction, UAVObservation


class UavEnv(
    EnvClient[UAVAction, UAVObservation, State]
):
    """
    Client for the UAV Fleet Tracking Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example (sync):
        with UavEnv(base_url="http://localhost:8000").sync() as client:
            result = client.reset()
            print(result.observation.features[:3])   # rel_pos of UAV 1

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
        """
        Convert UAVAction to JSON payload for the /step WebSocket message.

        FIX: was sending {"message": action.message} (Echo env template).
             UAVAction has no 'message' field — it has 'commands'.
        """
        return {
            "commands": action.commands,
        }

    def _parse_result(self, payload: Dict) -> StepResult[UAVObservation]:
        """
        Parse server response into StepResult[UAVObservation].

        FIX: was building UAVObservation with Echo env fields
             (echoed_message, message_length, metadata) — none of which
             exist on UAVObservation. Now correctly reads 'features'.
        """
        obs_data = payload.get("observation", {})

        observation = UAVObservation(
            features=obs_data.get("features", [0.0] * 48),
            reward=float(payload.get("reward", 0.0)),
            done=bool(payload.get("done", False)),
        )

        return StepResult(
            observation=observation,
            reward=float(payload.get("reward", 0.0)),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )