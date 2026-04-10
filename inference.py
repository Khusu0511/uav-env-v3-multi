"""
UAV Fleet Tracking — Inference Script (v3)
==========================================
Conforms to the Meta/PyTorch OpenEnv Hackathon submission spec:
- Uses OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN from env
- stdout: ONLY [START] / [STEP] / [END] lines — nothing else
- stderr: all debug / info / warn / error messages
- [END] includes score field strictly within (0.01, 0.99)
- Rule-based fallback when LLM output cannot be parsed
- Must complete within 20 min on vcpu=2, memory=8 GB
- Runs all 3 tasks: easy, medium, hard
"""

import os
import re
import sys
import json
import time
import textwrap
import requests
import numpy as np
from typing import List

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
API_BASE_URL      = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME        = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN          = os.getenv("HF_TOKEN")

ENV_URL           = os.getenv("ENV_URL", "http://localhost:8000")
MAX_STEPS         = 150
LLM_CALL_EVERY    = 25
SUCCESS_THRESHOLD = 0.25   # rule-based easily achieves this without API key

NFZ_ACTIVE_TASKS  = {"hard"}
OBS_PER_AGENT     = 16
NUM_AGENTS        = 3
_SAFE_ACTION      = [0.0] * (NUM_AGENTS * 3)


# ---------------------------------------------------------------------------
# SCORE CLAMPING
# Clamp final score to (0.01, 0.99) so f"{s:.3f}" NEVER produces "0.000"
# or "1.000" — which would fail the validator's strict-open-interval check.
# Individual step rewards are NOT clamped; they are passed through as-is.
# ---------------------------------------------------------------------------
def _clamp_score(v: float) -> float:
    return float(np.clip(v if v is not None else 0.01, 0.01, 0.99))


# ---------------------------------------------------------------------------
# OPENAI CLIENT
# ---------------------------------------------------------------------------
def build_openai_client():
    if not HF_TOKEN:
        print("[INFO] HF_TOKEN not set — running rule-based only.", file=sys.stderr, flush=True)
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent(f"""
    You control {NUM_AGENTS} UAVs in a 500x500x300 m airspace to intercept moving targets.
    Avoid walls and the No-Fly Zone (NFZ). Large penalties for violations.
    Observation per UAV ({OBS_PER_AGENT} values, {NUM_AGENTS} UAVs = {OBS_PER_AGENT * NUM_AGENTS} total):
    [0:3]  rel_pos  -- target_pos - uav_pos (m)
    [3:6]  rel_vel  -- target_vel - uav_vel (m/s)
    [6:9]  uav_vel  -- own velocity (m/s)
    [9:12] wind     -- wind vector (m/s)
    [12:15] nfz_vec -- vector from UAV to NFZ center (m)
    [15]   d_nfz   -- distance to NFZ surface (m)
    Action priority rules:
    1. WALLS: if uav_vel[dim]>4 subtract 0.4; if <-4 add 0.4.
    2. NFZ: if d_nfz<85, flee: cmd=-normalize(nfz_vec). Skip other rules.
    3. LOCK: if dist<15, match velocity: cmd=normalize(uav_vel+rel_vel).
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
        b    = i * OBS_PER_AGENT
        blk  = features[b: b + OBS_PER_AGENT]
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


def extract_obs_summary(features: list) -> list:
    summary = []
    for i in range(NUM_AGENTS):
        b   = i * OBS_PER_AGENT
        blk = features[b: b + OBS_PER_AGENT]
        summary.append({
            "agent":          i + 1,
            "dist_to_target": round(float(np.linalg.norm(blk[0:3])), 2),
            "rel_pos":        [round(v, 3) for v in blk[0:3]],
            "uav_vel":        [round(v, 3) for v in blk[6:9]],
            "d_nfz":          round(blk[15], 2),
        })
    return summary


# ---------------------------------------------------------------------------
# RULE-BASED CONTROLLER
# ---------------------------------------------------------------------------
def rule_based_action(features: list, task: str) -> list:
    action = []
    for i in range(NUM_AGENTS):
        b       = i * OBS_PER_AGENT
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

        # P2: NFZ avoidance — hard task only
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
    print("[WARN] Env not ready after timeout, proceeding.", file=sys.stderr, flush=True)
    return False


def wait_for_task(expected_task: str, timeout: int = 15) -> bool:
    for _ in range(timeout):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=2)
            if r.status_code == 200 and r.json().get("task") == expected_task:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    print(f"[WARN] Timeout waiting for task={expected_task}.", file=sys.stderr, flush=True)
    return False


def check_nfz_violations(task: str) -> int:
    if task not in NFZ_ACTIVE_TASKS:
        return 0
    try:
        r = requests.get(f"{ENV_URL}/nfz_status", timeout=2)
        if r.status_code == 200:
            return sum(
                1 for uav in r.json().get("uav_nfz_report", [])
                for chk in uav.get("nfz_checks", [])
                if chk.get("hard_violation")
            )
    except Exception:
        pass
    return 0


def fetch_frame():
    try:
        r = requests.get(f"{ENV_URL}/render", timeout=3)
        return r.content if r.status_code == 200 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# STRUCTURED LOG HELPERS
# stdout: ONLY [START] / [STEP] / [END] — nothing else ever printed here
# stderr: all info / debug / warn / error messages
#
# [END] format per hackathon spec:
#   [END] success=<true|false> steps=<n> score=<X.XXX> rewards=<r1,r2,...>
#   score is strictly within (0.01, 0.99) — never 0.000 or 1.000
# ---------------------------------------------------------------------------
def log_start(env_name: str, task: str, max_steps: int, model: str) -> None:
    payload = {
        "env_name":  env_name,
        "task":      task,
        "max_steps": max_steps,
        "model":     model,
    }
    print(f"[START] {json.dumps(payload)}", flush=True)


def log_step(step: int, action: list, reward: float, done: bool,
             action_source: str, nfz_violations: int,
             obs_state: list = None) -> None:
    payload = {
        "step":           step,
        "action":         [round(v, 4) for v in action],
        "reward":         round(float(reward), 4) if reward is not None else 0.0,
        "done":           done,
        "action_source":  action_source,
        "nfz_violations": nfz_violations,
    }
    if obs_state:
        payload["observation"] = obs_state
    print(f"[STEP] {json.dumps(payload)}", flush=True)


def log_end(success: bool, steps: int, score: float,
            rewards: List[float], nfz_total: int, model: str,
            llm_ok: int, llm_fail: int) -> None:
    """
    Emit [END] line per hackathon spec.
    score is clamped to strictly open interval (0.01, 0.99).
    rewards list is the raw per-step normalised rewards.
    """
    safe_score  = _clamp_score(score)
    rewards_str = ",".join(f"{r:.3f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={safe_score:.3f} rewards={rewards_str}",
        flush=True
    )


# ---------------------------------------------------------------------------
# SINGLE TASK RUNNER
# [END] guaranteed via finally block — emitted even on exception
# ---------------------------------------------------------------------------
def run_task(env_client, llm_client, task: str, env_name: str):
    rewards      = []
    nfz_total    = 0
    llm_ok       = 0
    llm_fail     = 0
    frames       = []
    llm_disabled = llm_client is None
    model_label  = MODEL_NAME if not llm_disabled else "rule_based"

    # [START] — exactly once per task
    log_start(env_name, f"Multi-UAV pursuit [{task}]", MAX_STEPS, model_label)

    try:
        res = env_client.reset(options={"task": task})
        wait_for_task(task, timeout=20)

        for step in range(1, MAX_STEPS + 1):
            features  = get_features(res)
            obs_valid = len(features) == OBS_PER_AGENT * NUM_AGENTS

            # Mask NFZ features on easy/medium — prevents phantom avoidance
            if obs_valid and task not in NFZ_ACTIVE_TASKS:
                for i in range(NUM_AGENTS):
                    features[i * OBS_PER_AGENT + 15] = 999.0

            exec_action   = _SAFE_ACTION[:]
            action_source = "safe"

            if obs_valid:
                llm_action = None
                if not llm_disabled and (step == 1 or step % LLM_CALL_EVERY == 0):
                    try:
                        raw        = query_llm(llm_client, f"Step {step}:\n{format_obs(features)}")
                        llm_action = parse_llm_action(raw)
                        if llm_action:
                            llm_ok += 1
                        else:
                            llm_fail += 1
                    except Exception as e:
                        llm_fail += 1
                        print(f"[WARN] LLM error: {e}", file=sys.stderr, flush=True)
                        if any(c in str(e) for c in ["402", "403"]):
                            llm_disabled = True

                if llm_action is not None:
                    exec_action, action_source = llm_action, "llm"
                else:
                    exec_action, action_source = rule_based_action(features, task), "rule"

            res           = env_client.step({"commands": exec_action})
            step_reward   = float(res.reward) if res.reward is not None else 0.0
            rewards.append(step_reward)

            post_features = get_features(res)
            obs_state     = extract_obs_summary(post_features) if post_features else None

            step_nfz = 0
            if step % 10 == 0:
                step_nfz   = check_nfz_violations(task)
                nfz_total += step_nfz

            # [STEP] — exactly once per step
            log_step(step, exec_action, step_reward,
                     bool(getattr(res, "done", False)),
                     action_source, step_nfz, obs_state)

            if IMAGEIO_AVAILABLE:
                fb = fetch_frame()
                if fb:
                    try:
                        frames.append(imageio.v3.imread(fb))
                    except Exception:
                        pass

            if bool(getattr(res, "done", False)):
                break

    except Exception as e:
        print(f"[ERROR] Task '{task}' aborted: {e}", file=sys.stderr, flush=True)

    finally:
        # Save video — must not prevent [END] from printing
        if IMAGEIO_AVAILABLE and frames:
            try:
                imageio.mimsave(f"submission_video_{task}.mp4", frames, fps=15, macro_block_size=1)
            except Exception:
                try:
                    imageio.mimsave(f"submission_video_{task}.gif", frames, fps=15)
                except Exception:
                    pass

        # [END] — exactly once per task, GUARANTEED even on exception
        avg_score = sum(rewards) / len(rewards) if rewards else 0.01
        success   = _clamp_score(avg_score) >= SUCCESS_THRESHOLD and nfz_total == 0
        log_end(success, len(rewards), avg_score, rewards,
                nfz_total, model_label, llm_ok, llm_fail)

    return _clamp_score(avg_score if rewards else 0.01), nfz_total


# ---------------------------------------------------------------------------
# MAIN — runs easy → medium → hard in sequence
# ---------------------------------------------------------------------------
def main():
    llm_client = build_openai_client()
    env_name   = "uav_env_v3_multi"
    tasks      = ["easy", "medium", "hard"]

    wait_for_env(timeout=60)

    all_results = {}

    with GenericEnvClient(base_url=ENV_URL).sync() as env_client:
        for task in tasks:
            score, nfz = run_task(env_client, llm_client, task, env_name)
            all_results[task] = {"score": score, "nfz_violations": nfz}
            time.sleep(1.0)

    # Summary to stderr only
    print("\n[SUMMARY]", file=sys.stderr)
    for task, r in all_results.items():
        print(
            f"  {task:6s} | score={r['score']:.4f} | nfz_violations={r['nfz_violations']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()