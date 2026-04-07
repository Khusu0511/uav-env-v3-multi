---
title: UAV Env v3 Multi
emoji: 🚁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# UAV Fleet Tracking Environment — `uav_env_v3_multi`

> **Meta × PyTorch OpenEnv Hackathon** submission  
> Real-world RL environment: multi-UAV pursuit of evasive targets in a constrained 3-D airspace.

---

## Team — FullmetalDevs 

| Name | Role |
| --- | --- |
| **Kushagra Gupta** | Team Lead |
| **Tanmay Sharad Mathurvaishya** | Member |
| **Shivam Chaturvedi** | Member |

---

## 🚀 Submission Description: `uav_env_v3_multi`

**Category:** Strategic Multi-Agent Systems / Infrastructure

### 1. Executive Summary

Our submission, `uav_env_v3_multi`, is a sophisticated multi-agent reinforcement learning environment simulating a fleet of three UAVs tasked with intercepting Evasive Targets in a constrained 3D airspace. The environment challenges agents with stochastic wind dynamics (Ornstein–Uhlenbeck process), high-dimensional state spaces (48D), and strict No-Fly Zone (NFZ) safety constraints.

### 2. Core Technical Innovations

* **Dynamic Evasion Logic:** Unlike static targets, our targets utilize proximity-aware swerving. They actively flee when a chaser enters an 80m radius, requiring the agent to master Lead Pursuit strategies rather than simple tail-chasing.
* **Atmospheric Realism:** The integration of smoothed OU wind noise (±3 m/s) forces the controller to perform continuous micro-adjustments, preventing "perfect" trajectories and mirroring real-world flight instability.
* **NFZ Hard-Wall Enforcement:** We implemented a physical repulsion layer that projects UAVs back to the sphere's surface upon collision and nullifies inward velocity, ensuring safety compliance is physically grounded, not just penalized via rewards.

### 3. Inference & Strategy Architecture

Our `inference.py` is engineered for maximum reliability and compliance with the Round 1 Rules:

* **Multi-Task Runner:** The script runs all three tasks (`easy`, `medium`, `hard`) in sequence, emitting separate `[START]`/`[STEP]`/`[END]` log blocks per task for automated grading.
* **Hybrid Scheduled Controller:** The script uses a sophisticated scheduling logic (`LLM_CALL_EVERY = 25`). It leverages LLM strategic guidance for long-range pathfinding while utilizing a high-frequency Rule-Based Fallback for 0.1s physics updates.
* **API-Independent Success:** By embedding tactical lead-pursuit and boundary-repulsion logic directly into the rule-based controller, our submission achieves positive normalized rewards and zero NFZ violations even in 100% offline environments without cloud access.
* **Structured Log Compliance:** The agent strictly emits the hackathon-mandated `[START]`, `[STEP]`, and `[END]` prefixed JSON logs to stdout, facilitating seamless automated grading and metric extraction.

### 4. Reward & Observation Design

* **48D State Representation:** Each agent receives 16 features, including a dedicated scalar `d_nfz` (distance to NFZ surface) to provide an unambiguous safety signal for collision avoidance.
* **Dense Signal Shaping:** We utilize a Three-Zone Reward (Capture, Approach, and Long-Range) normalized to `[0.0, 1.0]` to ensure the agent receives a constant gradient toward the target, significantly reducing the "sparse reward" problem common in 3D tracking tasks.

### 5. Hardware & Runtime Efficiency

Optimized for the hackathon's 2 vCPU / 8GB RAM limit:

* **Execution Time:** A full 150-step mission per task (3 tasks) with 3D rendering completes in approximately 10–14 minutes total, well within the 20-minute evaluation window.
* **Resource Footprint:** Lightweight FastAPI/Gymnasium implementation ensures minimal memory overhead.

---

## Overview

`uav_env_v3_multi` simulates a **fleet of 3 UAVs** tasked with autonomously intercepting 3 independently moving, evasion-aware targets inside a bounded 3-D airspace. The environment introduces:

* **No-Fly Zone (NFZ)** — a spherical hard exclusion boundary centred at (250, 250, 150) m with a 60 m hard wall and a 85 m soft-warning buffer.
* **3D evasive targets** — targets sense proximity and actively flee when a UAV closes within 80 m (hard task only).
* **Ornstein–Uhlenbeck wind** — smoothed stochastic wind up to ±3 m/s on all axes (medium/hard tasks).
* **Three-zone reward** — dense normalized signal at capture (<15 m), approach (15–60 m), and long-range pursuit (>60 m).

---

## Action Space

| Field | Type | Shape | Range |
| --- | --- | --- | --- |
| `commands` | `List[float]` | `(9,)` | `[-1, 1]` |

Nine velocity commands — `[vx, vy, vz]` for each of the 3 UAVs — scaled internally by `cmd_scale = 8 m/s`.

```python
from models import UAVAction
action = UAVAction(commands=[0.5, -0.3, 0.1,  0.0, 0.8, -0.2,  -0.4, 0.1, 0.6])
```

---

## Observation Space

| Field | Type | Shape |
| --- | --- | --- |
| `features` | `List[float]` | `(48,)` |

**16 features × 3 agents = 48 values.** Per-agent block:

| Index | Name | Description | Unit |
| --- | --- | --- | --- |
| 0–2 | `rel_pos` | target\_pos − uav\_pos | m |
| 3–5 | `rel_vel` | target\_vel − uav\_vel | m/s |
| 6–8 | `uav_vel` | own velocity | m/s |
| 9–11 | `wind` | wind vector | m/s |
| 12–14 | `nfz_vec` | vector from UAV to nearest NFZ centre | m |
| 15 | `d_nfz` | scalar distance to NFZ surface | m |

---

## Reward Function

Rewards are returned per step, normalized to [0.0, 1.0] across all 3 UAVs to ensure standardized metric tracking for hackathon evaluation.

## ⚠️ Important Note on Performance Evaluation

Reward signals are normalized to the range **[0, 1]** to comply with competition requirements and ensure consistent leaderboard comparisons. However, normalization can sometimes obscure subtle yet meaningful performance differences between agents.

To maintain flexibility, this project provides a **toggle-based mechanism** to switch between normalized and raw reward signals.

**Key Points:**
* By default, rewards are **normalized** for standard evaluation and submission.
* You can enable raw rewards by setting:

  ```python
  # in uav_env_environment.py
  USE_RAW_REWARD = True
  ```

* Raw rewards provide **more granular insights** for debugging, analysis, and research.
* Normalized rewards ensure **fair and consistent benchmarking** across different submissions.

> ⚡ For deeper analysis and development, consider using raw rewards, but switch back to normalized rewards for official evaluation.

| Condition | Raw Reward (per UAV) |
| --- | --- |
| dist < 15 m (capture + velocity match) | 100 + 60 × exp(−vel\_err / 5) |
| 15 m ≤ dist < 60 m (approach) | 20 + 80 × (1 − (dist−15)/45) |
| dist ≥ 60 m (long-range) | 20 × exp(−(dist−60)/80) |
| Hard NFZ violation (per UAV, per step) | −200 |
| Soft NFZ buffer penetration | −1.5 × penetration^1.2 |
| Near boundary (< 15 m margin) | −2 × gap |

**Returned reward** = `clip(sum_raw / 480, 0.0, 1.0)` where 480 = max raw per step (160 × 3 agents).

---

## Tasks (Easy → Medium → Hard)

The environment exposes three task difficulties via the `task` key in the `options` body on `/reset`:

| Task | Wind | Targets | NFZ | Key Challenge |
| --- | --- | --- | --- | --- |
| `easy` | None | Static | Off | Basic pursuit |
| `medium` | Light (±1.5 m/s) | Random-walk | Off | Wind compensation |
| `hard` *(default)* | Full OU (±3 m/s) | Evasive | Active | Full spec |

**Example reset call:**
```json
POST /reset
{ "options": { "task": "easy" } }
```

---

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/reset` | Reset environment (accepts `options.task`), returns initial observation |
| `POST` | `/step` | Apply action, returns observation + normalized reward |
| `GET` | `/state` | Current observation without stepping |
| `GET` | `/health` | Server + environment status |
| `GET` | `/render` | Current 3-D frame as PNG |
| `GET` | `/nfz_status` | Per-UAV NFZ compliance report |

---

## Environment Variables

Required before running inference:

```bash
API_BASE_URL=https://router.huggingface.co/v1   # LLM endpoint
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct    # model identifier
HF_TOKEN=hf_YOUR_TOKEN_HERE                      # Hugging Face API key
ENV_URL=http://localhost:8000                     # environment server URL
```

---

## Local Setup

```bash
# 1. Clone
git clone https://huggingface.co/spaces/khusu-133/uav-env-v3-fullmetaldevs
cd uav-env-v3-fullmetaldevs

# 2. Install dependencies
pip install openenv-core openai numpy requests imageio matplotlib

# 3. Copy and fill in environment variables
cp .env.example .env   # then edit .env with your HF_TOKEN

# 4. Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# 5. In a second terminal, run inference
export $(cat .env | xargs)
python inference.py
```

---

## Docker

```bash
# Build
docker build -t uav-env-v3 .

# Run server
docker run -p 8000:8000 uav-env-v3

# Run inference against running server
export $(cat .env | xargs)
python inference.py
```

---

## Inference Script

`inference.py` (root of project) uses the **OpenAI client** against your `API_BASE_URL` / `MODEL_NAME` / `HF_TOKEN` and emits structured `[START]` / `[STEP]` / `[END]` prefixed stdout logs required by the hackathon evaluator. It runs **all three tasks** in sequence:

```
[START] {"env_name": "uav_env_v3_multi", "task": "Multi-UAV pursuit [easy]: ...", "max_steps": 150, "model": "..."}
[STEP]  {"step": 1, "action": [...], "reward": 0.312, "done": false, "action_source": "rule", "nfz_violations": 0}
// ... one line per step ...
[END]   {"total_steps": 150, "avg_reward": 0.318, "success": true, "nfz_hard_violations": 0, "model": "...", "llm_calls_ok": 6, "llm_calls_fail": 0}
[START] {"env_name": "uav_env_v3_multi", "task": "Multi-UAV pursuit [medium]: ...", ...}
...
[START] {"env_name": "uav_env_v3_multi", "task": "Multi-UAV pursuit [hard]: ...", ...}
...
```

A rule-based fallback controller runs on all non-LLM steps, ensuring positive scores even without an API key.

---

## Project Structure

```
uav_env/
├── server/
│   ├── __init__.py
│   ├── app.py                   # FastAPI app with /render, /health, /nfz_status
│   ├── shared.py                # Global env pointer
│   └── uav_env_environment.py  # Core RL environment logic (task dispatch + norm reward)
├── models.py                    # UAVAction + UAVObservation Pydantic models
├── inference.py                 # Baseline inference script (hackathon spec, all 3 tasks)
├── .env.example                 # Template for API_BASE_URL, MODEL_NAME, HF_TOKEN
├── Dockerfile
├── openenv.yaml                 # OpenEnv spec with tasks block
├── pyproject.toml
└── README.md
```

---

## Key Design Decisions

* **Strict NFZ hard wall** — elastic sphere collision response prevents any penetration, maintaining physical realism and preventing reward hacking via wall-pass.
* **3D evasive targets** — flee direction blended with random walk; evasion weight increases as UAV closes in, creating a natural difficulty gradient.
* **Dense normalized reward signal** — three-zone proximity reward normalized to `[0.0, 1.0]`, giving the agent a useful gradient at all distances.
* **48-D observation** — scalar `d_nfz` added to the 15-D base to give the agent an unambiguous distance-to-boundary signal for NFZ avoidance learning.
* **Task-aware rendering** — NFZ wireframes only rendered when NFZ is active; task name shown in plot title.