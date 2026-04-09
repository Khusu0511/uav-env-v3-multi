---
title: UAV Env v3 Multi
emoji: 🚁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
base_path: /web
---

# UAV Fleet Tracking Environment — `uav_env_v3_multi`

> **Meta × PyTorch OpenEnv Hackathon** submission  
> Real-world RL environment: multi-UAV pursuit of evasive targets in a constrained 3-D airspace.

---

## Team — FullmetalDevs

| Name | Role |
|---|---|
| **Kushagra Gupta** | Team Lead |
| **Tanmay Sharad Mathurvaishya** | Member |
| **Shivam Chaturvedi** | Member |

---

## 🌍 Real-World Applications

This environment is designed to closely reflect real-world multi-agent UAV operations, where coordination, safety, and adaptability are critical. Each scenario maps directly to practical deployment challenges faced in modern autonomous aerial systems:

| Domain                                        | Relevance                                                                                                                                                                                                                                                                           |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🛡️ **Border & Perimeter Surveillance**       | Models continuous tracking of potentially evasive targets across large areas while strictly respecting restricted zones such as military bases or sensitive infrastructure. Demonstrates how UAV fleets can coordinate to maintain coverage without violating airspace constraints. |
| 🔥 **Wildfire & Disaster Response**           | Simulates harsh and unpredictable atmospheric conditions using stochastic wind dynamics, similar to real wildfire environments. UAVs must adapt in real-time to unstable airflow while tracking moving hotspots or rescue targets.                                                  |
| 🚢 **Maritime Vessel Interception**           | Reflects real-world interception of evasive ships or illegal vessels. Targets actively change direction, requiring predictive pursuit strategies rather than simple following behavior. NFZs can represent restricted maritime zones or territorial waters.                         |
| 🏙️ **Urban Air Traffic Management (UTM)**    | Mirrors future smart-city drone ecosystems where UAVs must navigate dense environments with strict geofencing. The NFZ hard-wall constraint directly models no-fly zones around buildings, airports, or populated areas.                                                            |
| 🔋 **Infrastructure Inspection & Monitoring** | Enables coordinated multi-UAV inspection of moving assets (e.g., vehicles, pipelines, or dynamic systems) while ensuring safety constraints. Demonstrates scalable coordination strategies under dynamic environmental conditions.                                                  |

These applications highlight the system’s ability to handle **multi-agent coordination, safety-critical constraints, stochastic environments, and adaptive decision-making**, making it suitable for both research and real-world deployment scenarios.

## 🎬 Demo Videos

Three difficulty levels captured from live inference runs using `inference.py`.

### Easy — Static Targets, No Wind, No NFZ
![Easy mode demo](submission_video_easy.gif)

### Medium — Random-Walk Targets, Light Wind (±1.5 m/s), No NFZ
![Medium mode demo](submission_video_medium.gif)

### Hard — Evasive Targets, Full OU Wind (±3 m/s), NFZ Active
![Hard mode demo](submission_video_hard.gif)

> MP4 versions: [`easy`](submission_video_easy.mp4) · [`medium`](submission_video_medium.mp4) · [`hard`](submission_video_hard.mp4)

---
## 🚀 Overview

**Category:** Strategic Multi-Agent Systems / Infrastructure

`uav_env_v3_multi` simulates a **fleet of 3 UAVs** intercepting 3 independently moving targets in a bounded 3-D airspace (500 × 500 × 300 m). Agents face stochastic wind dynamics (Ornstein–Uhlenbeck process), a 48-D state space, and strict No-Fly Zone (NFZ) safety constraints — all cleanly scoped per difficulty level.

| Task     | Wind              | Targets                    | NFZ    | Default   | Key Challenge       |
| -------- | ----------------- | -------------------------- | ------ | --------- | ------------------- |
| `easy`   | None              | Static                     | Off    | No        | Basic 3D pursuit    |
| `medium` | Light OU ±1.5 m/s | Random-walk                | Off    | No        | Wind compensation   |
| `hard`   | Full OU ±3 m/s    | Evasive (80 m flee radius) | Active | **✅ Yes** | Full constraint set |

> The environment **defaults to `hard`** when no `task` option is provided to `/reset`.

---

## 🔧 Core Technical Innovations

* **Dynamic Evasion Logic:** Targets actively flee when a UAV closes within 80 m (hard only), requiring lead-pursuit strategies rather than simple tail-chasing.
* **Atmospheric Realism:** Smoothed OU wind noise (±3 m/s on hard, ±1.5 m/s on medium, zero on easy) forces continuous micro-adjustments mirroring real-world flight instability.
* **NFZ Hard-Wall Enforcement:** A physical repulsion layer projects UAVs back to the sphere's surface on collision and nullifies inward velocity — safety is physically grounded, not just reward-penalised.
* **OpenEnv State Compliance:** Full `UAVState` model with `episode_id`, `step_count`, and `done` satisfies the OpenEnv `GET /state` spec for automated grader compatibility.
* **Race-Condition-Safe Reset:** `inference.py` polls `/health` after each `reset()` until the server confirms the correct task is live before recording any frames.

---

## 🏗️ Inference & Strategy Architecture

* **Multi-Task Runner:** `inference.py` runs all three tasks (`easy` → `medium` → `hard`) in sequence, emitting `[START]`/`[STEP]`/`[END]` log blocks per task for automated grading.
* **Hybrid Scheduled Controller:** LLM strategic guidance every 25 steps; rule-based fallback fills the rest.
* **Task-Aware Rule Controller:** NFZ flee logic (`d_nfz < 85`) is gated exclusively to hard, preventing phantom avoidance on easy/medium.
* **API-Independent Success:** Embedded lead-pursuit and boundary-repulsion logic ensures positive normalised rewards and zero NFZ violations even without cloud API access. `SUCCESS_THRESHOLD = 0.25`.

---

## 📐 Action & Observation Spaces

### Action Space

Nine velocity commands — `[vx, vy, vz]` for each of the 3 UAVs, scaled by `cmd_scale = 8 m/s`.

| Field      | Type          | Shape  | Range     | Default             |
| ---------- | ------------- | ------ | --------- | ------------------- |
| `commands` | `List[float]` | `(9,)` | `[-1, 1]` | `[0.0] × 9` (hover) |

```python
from models import UAVAction
action = UAVAction(commands=[0.5, -0.3, 0.1,  0.0, 0.8, -0.2,  -0.4, 0.1, 0.6])
# Default (hover): UAVAction()  →  commands=[0.0]*9
```

### Observation Space — 48D (16 features × 3 agents)

| Index (per agent) | Name      | Unit | Description                                                    |
| ----------------- | --------- | ---- | -------------------------------------------------------------- |
| 0–2               | `rel_pos` | m    | `target_pos − uav_pos` — direction and range to pursue         |
| 3–5               | `rel_vel` | m/s  | `target_vel − uav_vel` — relative closing speed                |
| 6–8               | `uav_vel` | m/s  | Own velocity; used for boundary repulsion and velocity lock    |
| 9–11              | `wind`    | m/s  | Current OU wind vector; agent should counter-compensate        |
| 12–14             | `nfz_vec` | m    | Vector from UAV to nearest NFZ centre                          |
| 15                | `d_nfz`   | m    | Scalar distance to NFZ surface (trigger avoidance when < 85 m) |

### State Space (`GET /state`)

| Field          | Type   | Description                             |
| -------------- | ------ | --------------------------------------- |
| `episode_id`   | `str`  | `"uav_episode"`                         |
| `step_count`   | `int`  | Steps since last `/reset`               |
| `done`         | `bool` | `false` (episodes never auto-terminate) |
| `current_task` | `str`  | `"easy"`, `"medium"`, or `"hard"`       |
| `num_agents`   | `int`  | `3`                                     |
| `obs_size`     | `int`  | `48`                                    |

---

## 🏆 Reward Function

Normalised to `[0.0, 1.0]` per step across all 3 UAVs.
**Returned:** `clip(sum_raw / 480.0, 0.0, 1.0)` where 480 = max raw per step (160 × 3 agents).

| Condition                              | Raw Reward (per UAV)         |
| -------------------------------------- | ---------------------------- |
| dist < 15 m (capture + velocity match) | 100 + 60 × exp(−vel_err / 5) |
| 15 m ≤ dist < 60 m (approach)          | 20 + 80 × (1 − (dist−15)/45) |
| dist ≥ 60 m (long-range)               | 20 × exp(−(dist−60)/80)      |
| Hard NFZ violation                     | −200 per UAV                 |
| Soft NFZ buffer penetration            | −1.5 × penetration^1.2       |
| Near boundary (< 15 m margin)          | −2 × gap                     |

At 420 m distance, reward ≈ 0.0017 (long-range zone). As UAVs close in, reward rises sharply toward `1.0`.

```python
# in server/uav_env_environment.py
USE_RAW_REWARD = False  # set True for debugging/research
```

---

## 📡 Step Response

Every `/step` call returns:

```json
{
  "done": false,
  "reward": 0.0017,
  "metadata": {
    "task": "hard",
    "current_step": 13,
    "avg_target_distance": 420.17,
    "wind_magnitude": 1.48,
    "nfz_active": true
  },
  "features": [336.16, 334.36, 106.14, 4.66, 3.46, -1.52,
               0.0, 0.0, 0.0, 0.28, -1.33, 0.57,
               127.94, 166.65, 83.00, 225.89, "..."],
  "episode_id": "uav_episode",
  "step_count": 13
}
```

---

## 🔌 API Endpoints

| Method | Endpoint      | Description                                                            |
| ------ | ------------- | ---------------------------------------------------------------------- |
| `POST` | `/reset`      | Reset env; accepts `{\"options\": {\"task\": \"easy\|medium\|hard\"}}` |
| `POST` | `/step`       | Apply action; returns observation + normalised reward                  |
| `GET`  | `/state`      | Current `UAVState`                                                     |
| `GET`  | `/health`     | Server + env status (includes active `task`)                           |
| `GET`  | `/render`     | Current 3-D frame as PNG                                               |
| `GET`  | `/nfz_status` | Per-UAV NFZ compliance report (meaningful on hard only)                |
| `GET`  | `/web`        | HTML dashboard                                                         |
| `GET`  | `/docs`       | Interactive Swagger UI                                                 |

### Validation Script

Run the submission validator from the project root:

```bash
./validate-submission.sh https://kushagra0511-uav-env-v3-multi.hf.space
```

This checks that `POST /reset` responds correctly, the Docker image builds, and `openenv validate` passes.

### Validation Screenshot
 
![Validation Screenshot](Validation_Screenshot.png)

> The screenshot above shows all validation checks passing, including the final `All 3/3 checks passed!` result.

### Task Selection

```bash
# Easy — static targets, no wind, no NFZ
curl -X POST -H "Content-Type: application/json" \
  -d '{"options": {"task": "easy"}}' \
  https://kushagra0511-uav-env-v3-multi.hf.space/reset

# Medium — random-walk targets, light wind, no NFZ
curl -X POST -H "Content-Type: application/json" \
  -d '{"options": {"task": "medium"}}' \
  https://kushagra0511-uav-env-v3-multi.hf.space/reset

# Hard — evasive targets, full wind, active NFZ (default)
curl -X POST -H "Content-Type: application/json" \
  -d '{"options": {"task": "hard"}}' \
  https://kushagra0511-uav-env-v3-multi.hf.space/reset
```

### Manual Testing

```bash
# Health (shows active task)
curl https://kushagra0511-uav-env-v3-multi.hf.space/health

# State
curl https://kushagra0511-uav-env-v3-multi.hf.space/state

# Step with zero commands (hover)
curl -X POST -H "Content-Type: application/json" \
  -d '{"commands": [0,0,0, 0,0,0, 0,0,0]}' \
  https://kushagra0511-uav-env-v3-multi.hf.space/step

# NFZ compliance report
curl https://kushagra0511-uav-env-v3-multi.hf.space/nfz_status

# Live 3D render
curl https://kushagra0511-uav-env-v3-multi.hf.space/render --output frame.png
```

Interactive Swagger UI: [`/docs`](https://kushagra0511-uav-env-v3-multi.hf.space/docs)

> **Note:** The web Playground "Reset" button sends `{}` (empty body), which defaults to `hard`. Use `curl` or `/docs` to access `easy` and `medium` modes.

---

## ⚙️ Local Setup

```bash
# Clone
git clone https://huggingface.co/spaces/kushagra0511/uav-env-v3-multi
cd uav-env-v3-multi

# Install dependencies
pip install openenv-core openai numpy requests imageio matplotlib Pillow

# Configure
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=hf_YOUR_TOKEN_HERE
export ENV_URL=http://localhost:8000

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run inference (second terminal — all 3 tasks)
python inference.py
```

### Docker

```bash
docker build -t uav-env-v3 .

docker run -p 8000:8000 \
  -e HF_TOKEN=hf_YOUR_TOKEN \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
  uav-env-v3
```

---

## 📋 Inference Logs

`inference.py` emits spec-compliant structured logs for all three tasks:

```
[START] {"env_name": "uav_env_v3_multi", "task": "Multi-UAV pursuit [easy]...", "max_steps": 150, "model": "..."}
[STEP]  {"step": 1, "action": [...], "reward": 0.312, "done": false, "action_source": "rule", "nfz_violations": 0}
[END]   {"total_steps": 150, "avg_reward": 0.318, "success": true, "nfz_hard_violations": 0, ...}
```

`success = (avg_reward >= 0.25) AND (nfz_hard_violations == 0)`

The rule-based fallback controller ensures positive scores even without an API key.

---

## 📁 Project Structure

```
uav-env-v3-multi/
├── server/
│   ├── app.py                    # FastAPI: /reset /step /state /render /health /nfz_status
│   ├── shared.py                 # Global environment pointer (active_env)
│   ├── uav_env_environment.py    # Core RL env: physics, task dispatch, reward, render
│   └── requirements.txt
├── models.py                     # UAVAction, UAVObservation, UAVState
├── client.py                     # UAVEnv(EnvClient) — OpenEnv client wrapper
├── inference.py                  # Baseline inference: all 3 tasks, structured logs
├── openenv.yaml                  # OpenEnv spec: tasks, runtime, port
├── pyproject.toml
├── Dockerfile
├── submission_video_easy.gif/mp4
├── submission_video_medium.gif/mp4
├── submission_video_hard.gif/mp4
├── Validation_Screenshot.png
├── validate-submission.sh
└── README.md
```

---

## 🔑 Key Design Decisions

* **Hard default task** — `/reset` with no `task` option starts a hard episode; the Playground always lands on the most challenging configuration.
* **Strict NFZ hard wall** — elastic sphere collision response prevents penetration, blocking reward hacking via wall-pass.
* **Task-scoped features** — wind, NFZ, and target evasion are independently toggled per task with zero feature bleed between episodes.
* **`d_nfz` scalar** — observation index 15 gives an unambiguous distance-to-boundary signal; NFZ flee gated to hard only so no phantom avoidance on easy/medium.
* **Normalised reward** — `clip(raw / 480, 0, 1)` gives the automated grader always-valid `[0, 1]` values. `SUCCESS_THRESHOLD = 0.25`.
* **Race-condition-safe reset** — `wait_for_task()` polls `/health` until the server confirms the correct task before capturing any frames.
* **Default hover action** — `UAVAction.commands` defaults to `[0.0] × 9` so empty `/step {}` calls never raise Pydantic validation errors.
* **Task-aware rendering** — NFZ wireframes and wind arrows drawn only when those features are active.
