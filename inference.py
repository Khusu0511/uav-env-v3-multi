"""
UAV Fleet Tracking — Inference Script (v3)
==========================================
Conforms to the Meta/PyTorch OpenEnv Hackathon submission spec:
  - Uses OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN from env
  - HF_TOKEN is REQUIRED — raises ValueError if missing (no silent fallback)
  - Emits FLAT [START] / [STEP] / [END] stdout lines — NO JSON blobs
  - Rule-based fallback when LLM output cannot be parsed
  - [END] is always emitted via finally block
  - Rewards are strictly in (0, 1) — clamped to [0.01, 0.99]
  - done / success are lowercase true/false strings
  - Runs all 3 tasks: easy → medium → hard

Environment variables:
    API_BASE_URL  — LLM endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME    — model id      (default: meta-llama/Llama-3.3-70B-Instruct)
    HF_TOKEN      — HF / API key  (REQUIRED — no default, raises if absent)
    ENV_URL       — env server    (default: http://localhost:8000)
"""

import os
import re
import sys
import json
import time
import textwrap
import requests
import numpy as np

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

from openai import OpenAI
from openenv.core.generic_client import GenericEnvClient

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

# REQUIRED — raise immediately at startup, never fall back silently
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable is required. "
        "Set it before running inference.py."
    )

ENV_URL           = os.getenv("ENV_URL", "http://localhost:8000")
MAX_STEPS         = 150
LLM_CALL_EVERY    = 25
SUCCESS_THRESHOLD = 0.35

NFZ_ACTIVE_TASKS = {"hard"}
OBS_PER_AGENT    = 16
NUM_AGENTS       = 3
_SAFE_ACTION     = [0.0] * (NUM_AGENTS * 3)

# ---------------------------------------------------------------------------
# REWARD CLAMPING
# Spec: task score must be STRICTLY between 0 and 1 — not 0.0, not 1.0.
# At 2 decimal places, 0.01 → "0.01" and 0.99 → "0.99" — both safe.
# 1e-5 rounds to "0.00" which FAILS the validator. Do NOT use 1e-5.
# ---------------------------------------------------------------------------
def _clamp(r: float) -> float:
    """Clamp reward to the open interval (0, 1) — strictly inside."""
    return float(np.clip(r, 0.01, 0.99))


# ---------------------------------------------------------------------------
# OPENAI CLIENT
# ---------------------------------------------------------------------------
def build_openai_client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent(f"""
    You control {NUM_AGENTS} UAVs in a 500×500×300 m airspace to intercept moving targets.
    Avoid walls and the No-Fly Zone (NFZ). Large penalties for violations.
    Observation per UAV ({OBS_PER_AGENT} values, {NUM_AGENTS} UAVs = {OBS_PER_AGENT * NUM_AGENTS} total):
      [0:3]  rel_pos  — target_pos - uav_pos (m)
      [3:6]  rel_vel  — target_vel - uav_vel (m/s)
      [6:9]  uav_vel  — own velocity (m/s)
      [9:12] wind     — wind vector (m/s)
      [12:15] nfz_vec — vector from UAV to NFZ center (m)
      [15]   d_nfz   — distance to NFZ surface (m)
    Action priority rules:
      1. WALLS: if uav_vel[dim]>4 subtract 0.4; if <-4 add 0.4.
      2. NFZ:   if d_nfz<85, flee: cmd=-normalize(nfz_vec). Skip other rules.
      3. LOCK:  if dist<15, match velocity: cmd=normalize(uav_vel+rel_vel).
      4. CHASE: cmd=normalize(rel_pos)*clip(dist/60,0.3,1) - 0.25*normalize(wind).
    Respond with ONLY a JSON array of exactly 9 floats in [-1, 1], 2 decimal places.
    Example: [-0.82, 0.45, 0.12, 0.67, -0.23, 0.05, -0.41, 0.33, 0.09]
""").strip()


# ---------------------------------------------------------------------------
# OBSERVATION HELPERS
# ---------------------------------------------------------------------------
def get_features(res) -> list:
    try:
        obs = res.observation
        if isinstance(obs, dict):
            return obs.get("features", [])
        if hasattr(obs, "features"):
            return list(obs.features)
        return list(obs) if obs else []
    except Exception:
        return []


def format_obs(features: list) -> str:
    lines = []
    for i in range(NUM_AGENTS):
        b = i * OBS_PER_AGENT
        blk = features[b: b + OBS_PER_AGENT]
        dist = round(float(np.linalg.norm(blk[0:3])), 1)
        lines.append(
            f"UAV{i+1}|dist={dist}"
            f"|rp={[round(v,2) for v in blk[0:3]]}"
            f"|rv={[round(v,2) for v in blk[3:6]]}"
            f"|vel={[round(v,2) for v in blk[6:9]]}"
            f"|wind={[round(v,2) for v in blk[9:12]]}"
            f"|nfz={[round(v,1) for v in blk[12:15]]}"
            f"|d_nfz={round(blk[15],1)}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RULE-BASED CONTROLLER
# ---------------------------------------------------------------------------
def rule_based_action(features: list, task: str) -> list:
    action = []
    for i in range(NUM_AGENTS):
        b = i * OBS_PER_AGENT
        rel_pos = np.array(features[b + 0: b + 3])
        rel_vel = np.array(features[b + 3: b + 6])
        uav_vel = np.array(features[b + 6: b + 9])
        wind    = np.array(features[b + 9: b + 12])
        nfz_vec = np.array(features[b + 12: b + 15])
        d_nfz   = float(features[b + 15])
        dist    = float(np.linalg.norm(rel_pos))

        # P1: Boundary repulsion
        boundary = np.zeros(3)
        for dim in range(3):
            if uav_vel[dim] > 4.0:
                boundary[dim] -= 0.4
            elif uav_vel[dim] < -4.0:
                boundary[dim] += 0.4

        # P2: NFZ avoidance (only on hard)
        if task in NFZ_ACTIVE_TASKS and d_nfz < 85.0:
            nfz_n   = np.linalg.norm(nfz_vec)
            flee    = -nfz_vec / (nfz_n + 1e-8)
            urgency = 1.0 - (d_nfz / 85.0)
            cmd     = flee * (0.6 + 0.4 * urgency) + boundary
            action.extend(np.clip(cmd, -1.0, 1.0).tolist())
            continue

        # P3: Velocity lock (capture zone)
        if dist < 15.0:
            match = uav_vel + rel_vel
            n     = np.linalg.norm(match)
            cmd   = (match / (n + 1e-8)) + boundary
            action.extend(np.clip(cmd, -1.0, 1.0).tolist())
            continue

        # P4: Intercept + wind feedforward
        intercept = rel_pos / (dist + 1e-8)
        urgency   = float(np.clip(dist / 60.0, 0.3, 1.0))
        wind_n    = np.linalg.norm(wind)
        wind_comp = (-wind / (wind_n + 1e-8)) * min(wind_n / 8.0, 0.25)
        cmd       = intercept * urgency + wind_comp + boundary
        action.extend(np.clip(cmd, -1.0, 1.0).tolist())

    return action


# ---------------------------------------------------------------------------
# LLM HELPERS
# ---------------------------------------------------------------------------
def parse_llm_action(raw: str):
    try:
        match = re.search(r'\[([^\[\]]+)\]', raw)
        if not match:
            return None
        inner = match.group(1)
        inner = re.sub(r'(?<=[\d.])\s+(?=-?[\d.])', ',', inner)
        inner = re.sub(r'(?<=\d)(?=-[0-9])', ',', inner)
        vals  = json.loads(f'[{inner}]')
        if not isinstance(vals, list) or len(vals) != 9:
            return None
        return [float(np.clip(v, -1.0, 1.0)) for v in vals]
    except Exception:
        return None


def query_llm(client: OpenAI, obs_text: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": obs_text},
        ],
        temperature=0.0,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# ENV HELPERS
# ---------------------------------------------------------------------------
def wait_for_env(timeout: int = 60) -> bool:
    for _ in range(timeout):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    print("[WARN] Environment health check timed out; proceeding anyway.",
          file=sys.stderr, flush=True)
    return False


def wait_for_task(expected_task: str, timeout: int = 20) -> bool:
    for _ in range(timeout):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=2)
            if r.status_code == 200:
                data = r.json()
                if data.get("task") == expected_task:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    print(f"[WARN] Timed out waiting for task={expected_task}; proceeding anyway.",
          file=sys.stderr, flush=True)
    return False


def check_nfz_violations(task: str) -> int:
    if task not in NFZ_ACTIVE_TASKS:
        return 0
    violations = 0
    try:
        r = requests.get(f"{ENV_URL}/nfz_status", timeout=2)
        if r.status_code == 200:
            for uav in r.json().get("uav_nfz_report", []):
                for chk in uav.get("nfz_checks", []):
                    if chk.get("hard_violation"):
                        violations += 1
    except Exception:
        pass
    return violations


def fetch_frame():
    try:
        r = requests.get(f"{ENV_URL}/render", timeout=3)
        return r.content if r.status_code == 200 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# STRUCTURED LOG HELPERS
# ---------------------------------------------------------------------------
# Spec requires FLAT key=value lines — NOT JSON blobs.
#
#   [START] task=<str> env=<str> model=<str>
#   [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
#   [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
#
# Rewards formatted to exactly 2 decimal places.
# Boolean values as lowercase "true" / "false".
# ---------------------------------------------------------------------------

def log_start(task: str, env_name: str, model: str):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: list, reward: float,
             done: bool, error: str = None):
    safe_reward = _clamp(reward)
    action_str  = str([round(v, 2) for v in action])   # plain str, NOT json.dumps
    done_str    = "true" if done else "false"
    error_str   = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={safe_reward:.2f}"
        f" done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list):
    """
    [END] always emitted from finally block.
    Each reward in the list is individually clamped so no value is 0.0 or 1.0.
    """
    success_str  = "true" if success else "false"
    safe_rewards = [_clamp(r) for r in rewards]
    rewards_str  = ",".join(f"{r:.2f}" for r in safe_rewards)
    print(
        f"[END] success={success_str} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# SINGLE TASK RUNNER
# ---------------------------------------------------------------------------
def run_task(env_client, llm_client, task: str, env_name: str) -> float:
    """Run one full episode. Returns avg_reward (raw, before clamping)."""
    rewards      = []
    nfz_total    = 0
    llm_disabled = (llm_client is None)
    frames       = []

    log_start(
        task     = task,
        env_name = env_name,
        model    = MODEL_NAME if not llm_disabled else "rule_based",
    )

    try:
        res = env_client.reset(options={"task": task})
        wait_for_task(task, timeout=20)

        for step in range(1, MAX_STEPS + 1):
            features  = get_features(res)
            obs_valid = (len(features) == OBS_PER_AGENT * NUM_AGENTS)

            # Mask NFZ fields on easy/medium — d_nfz has no meaning there
            if obs_valid and task not in NFZ_ACTIVE_TASKS:
                for i in range(NUM_AGENTS):
                    features[i * OBS_PER_AGENT + 15] = 999.0

            exec_action   = _SAFE_ACTION[:]
            step_error    = None

            if obs_valid:
                llm_action = None

                if not llm_disabled and (step == 1 or step % LLM_CALL_EVERY == 0):
                    try:
                        raw        = query_llm(llm_client, f"Step {step}:\n{format_obs(features)}")
                        llm_action = parse_llm_action(raw)
                    except Exception as e:
                        step_error = str(e)
                        if any(code in step_error for code in ["402", "403"]):
                            llm_disabled = True

                if llm_action is not None:
                    exec_action = llm_action
                else:
                    exec_action = rule_based_action(features, task)

            # Step the environment
            res        = env_client.step({"commands": exec_action})
            raw_reward = float(res.reward)
            done       = bool(getattr(res, "done", False))
            rewards.append(raw_reward)

            # NFZ check every 10 steps
            step_nfz = 0
            if step % 10 == 0:
                step_nfz   = check_nfz_violations(task)
                nfz_total += step_nfz

            log_step(
                step   = step,
                action = exec_action,
                reward = raw_reward,
                done   = done,
                error  = step_error,
            )

            # Capture render frame
            if IMAGEIO_AVAILABLE:
                fb = fetch_frame()
                if fb:
                    try:
                        frames.append(imageio.v3.imread(fb))
                    except Exception:
                        pass

            if done:
                break

    except Exception as e:
        print(f"[ERROR] Task '{task}' aborted: {e}", file=sys.stderr, flush=True)

    finally:
        # Save per-task video (best-effort)
        if IMAGEIO_AVAILABLE and frames:
            try:
                imageio.mimsave(
                    f"submission_video_{task}.mp4", frames,
                    fps=15, macro_block_size=1,
                )
            except Exception:
                try:
                    imageio.mimsave(f"submission_video_{task}.gif", frames, fps=15)
                except Exception:
                    pass

        avg_reward = float(np.mean(rewards)) if rewards else 0.5
        success    = (avg_reward >= SUCCESS_THRESHOLD) and (nfz_total == 0)

        # [END] is guaranteed via finally — rewards list clamped inside log_end
        log_end(
            success = success,
            steps   = len(rewards),
            rewards = rewards,
        )

    return avg_reward


# ---------------------------------------------------------------------------
# MAIN — easy → medium → hard
# ---------------------------------------------------------------------------
def main():
    llm_client = build_openai_client()
    env_name   = "uav_env_v3_multi"
    tasks      = ["easy", "medium", "hard"]

    wait_for_env(timeout=60)

    all_results = {}

    with GenericEnvClient(base_url=ENV_URL).sync() as env_client:
        for task in tasks:
            avg = run_task(env_client, llm_client, task, env_name)
            all_results[task] = avg
            time.sleep(1.0)

    # Summary goes to stderr only — not seen by grader
    print("\n[SUMMARY]", file=sys.stderr)
    for task, avg in all_results.items():
        print(f"  {task:6s} | avg_reward={avg:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()