"""
UAV Fleet Tracking — Inference Script (v3)
==========================================
Conforms to the Meta/PyTorch OpenEnv Hackathon submission spec:
- Uses OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN from env
- Emits structured [START] / [STEP] / [END] stdout logs
- Rule-based fallback when LLM output cannot be parsed
- Must complete within 20 min on vcpu=2, memory=8 GB
- Runs all 3 tasks: easy, medium, hard

Setup:
    pip install openai openenv numpy requests imageio

Environment variables (define in .env or export before running):
    API_BASE_URL — LLM endpoint (e.g. https://router.huggingface.co/v1)
    MODEL_NAME   — model id (e.g. meta-llama/Llama-3.3-70B-Instruct)
    HF_TOKEN     — HF / API key

Run:
    export $(cat .env | xargs)
    python inference.py
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
# CONFIG — all sensitive values come from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

ENV_URL           = os.getenv("ENV_URL", "http://localhost:8000")
MAX_STEPS         = 150     # safe under 20 min on 2 vCPU / 8 GB
LLM_CALL_EVERY    = 25      # call LLM every N steps; rule-based fills the rest
SUCCESS_THRESHOLD = 0.35    # normalised reward (0–1) to mark run successful

NFZ_ACTIVE_TASKS = {"hard"}
OBS_PER_AGENT = 16
NUM_AGENTS    = 3
_SAFE_ACTION  = [0.0] * (NUM_AGENTS * 3)

# ---------------------------------------------------------------------------
# OPENAI CLIENT (hackathon mandates OpenAI client for all LLM calls)
# ---------------------------------------------------------------------------
def build_openai_client():
    if not HF_TOKEN:
        print("[INFO] HF_TOKEN not set — LLM disabled, running rule-based only.")
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent(f"""
    You control {NUM_AGENTS} UAVs in a 500×500×300 m airspace to intercept moving targets.
    Avoid walls and the No-Fly Zone (NFZ). Large penalties for violations.

    Observation per UAV ({OBS_PER_AGENT} values, {NUM_AGENTS} UAVs = {OBS_PER_AGENT * NUM_AGENTS} total):
    [0:3]  rel_pos — target_pos - uav_pos (m)
    [3:6]  rel_vel — target_vel - uav_vel (m/s)
    [6:9]  uav_vel — own velocity (m/s)
    [9:12] wind    — wind vector (m/s)
    [12:15] nfz_vec — vector from UAV to NFZ center (m)
    [15]   d_nfz   — distance to NFZ surface (m)

    Action priority rules:
    1. WALLS: if uav_vel[dim]>4 subtract 0.4; if <-4 add 0.4.
    2. NFZ:  if d_nfz<85, flee: cmd=-normalize(nfz_vec). Skip other rules.
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
# RULE-BASED CONTROLLER (no LLM required — produces positive rewards)
# ---------------------------------------------------------------------------
def rule_based_action(features: list) -> list:
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

        # P2: NFZ avoidance
        if d_nfz < 85.0:
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
# LLM ACTION PARSER
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
def wait_for_env(timeout: int = 30) -> bool:
    for _ in range(timeout):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def check_nfz_violations(task: str) -> int:
    if task not in NFZ_ACTIVE_TASKS:
        return 0   # NFZ disabled for easy/medium — ignore ghost violations

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
# Hackathon spec: lines must be prefixed [START] / [STEP] / [END]
# followed by a JSON payload on the same line.
# ---------------------------------------------------------------------------
def log_start(env_name: str, task: str, max_steps: int, model: str):
    payload = {
        "env_name":  env_name,
        "task":      task,
        "max_steps": max_steps,
        "model":     model,
    }
    print(f"[START] {json.dumps(payload)}", flush=True)


def log_step(step: int, action: list, reward: float, done: bool,
             action_source: str, nfz_violations: int):
    payload = {
        "step":           step,
        "action":         [round(v, 4) for v in action],
        "reward":         round(float(reward), 4),
        "done":           done,
        "action_source":  action_source,   # "llm" | "rule" | "safe"
        "nfz_violations": nfz_violations,
    }
    print(f"[STEP] {json.dumps(payload)}", flush=True)


def log_end(total_steps: int, avg_reward: float, success: bool,
            nfz_total: int, model: str, llm_ok: int, llm_fail: int):
    payload = {
        "total_steps":         total_steps,
        "avg_reward":          round(avg_reward, 4),
        "success":             success,
        "nfz_hard_violations": nfz_total,
        "model":               model,
        "llm_calls_ok":        llm_ok,
        "llm_calls_fail":      llm_fail,
    }
    print(f"[END] {json.dumps(payload)}", flush=True)


# ---------------------------------------------------------------------------
# SINGLE TASK RUNNER
# ---------------------------------------------------------------------------
def run_task(env_client, llm_client, task: str, env_name: str):
    """Run one full episode for a given task. Returns (avg_reward, nfz_total)."""
    rewards      = []
    nfz_total    = 0
    llm_ok       = 0
    llm_fail     = 0
    frames       = []
    llm_disabled = llm_client is None

    task_label = (
        f"Multi-UAV pursuit [{task}]: "
        f"intercept targets while avoiding NFZ and boundaries"
    )

    log_start(
        env_name  = env_name,
        task      = task_label,
        max_steps = MAX_STEPS,
        model     = MODEL_NAME if not llm_disabled else "rule_based",
    )

    try:
        # Reset with task option — inside try so [END] is always emitted
        res = env_client.reset(options={"task": task})
        wait_for_env(timeout=30)

        for step in range(1, MAX_STEPS + 1):
            features  = get_features(res)
            obs_valid = len(features) == OBS_PER_AGENT * NUM_AGENTS

            # Mask NFZ data on easy/medium tasks so neither LLM nor
            # rule-based logic tries to flee a non-existent NFZ.
            if obs_valid and task not in NFZ_ACTIVE_TASKS:
                for i in range(NUM_AGENTS):
                    features[i * OBS_PER_AGENT + 15] = 999.0

            action_source = "safe"
            exec_action   = _SAFE_ACTION[:]

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
                        err = str(e)
                        if any(code in err for code in ["402", "403"]):
                            llm_disabled = True

                if llm_action is not None:
                    exec_action   = llm_action
                    action_source = "llm"
                else:
                    exec_action   = rule_based_action(features)
                    action_source = "rule"

            # Step environment
            res = env_client.step({"commands": exec_action})
            rewards.append(res.reward)

            # NFZ check every 10 steps
            step_nfz = 0
            if step % 10 == 0:
                step_nfz  = check_nfz_violations(task)
                nfz_total += step_nfz

            log_step(
                step           = step,
                action         = exec_action,
                reward         = res.reward,
                done           = bool(getattr(res, "done", False)),
                action_source  = action_source,
                nfz_violations = step_nfz,
            )

            # Capture render frame
            if IMAGEIO_AVAILABLE:
                frame_bytes = fetch_frame()
                if frame_bytes:
                    try:
                        frames.append(imageio.v3.imread(frame_bytes))
                    except Exception:
                        pass

            # Early exit if episode is done
            if bool(getattr(res, "done", False)):
                break

    except Exception as e:
        print(f"[ERROR] Task '{task}' aborted: {e}", file=sys.stderr, flush=True)

    finally:
        # Save per-task video
        if IMAGEIO_AVAILABLE and frames:
            try:
                imageio.mimsave(
                    f"submission_video_{task}.mp4", frames,
                    fps=15, macro_block_size=1
                )
            except Exception:
                try:
                    imageio.mimsave(
                        f"submission_video_{task}.gif", frames, fps=15
                    )
                except Exception:
                    pass

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        success    = avg_reward >= SUCCESS_THRESHOLD and nfz_total == 0

        log_end(
            total_steps = len(rewards),
            avg_reward  = avg_reward,
            success     = success,
            nfz_total   = nfz_total,
            model       = MODEL_NAME if not llm_disabled else "rule_based",
            llm_ok      = llm_ok,
            llm_fail    = llm_fail,
        )

    return avg_reward, nfz_total


# ---------------------------------------------------------------------------
# MAIN — runs easy → medium → hard in sequence
# ---------------------------------------------------------------------------
def main():
    llm_client = build_openai_client()
    env_name   = "uav_env_v3_multi"
    tasks      = ["easy", "medium", "hard"]

    wait_for_env(timeout=60)

    all_results = {}

    # Use context manager to ensure clean connection lifecycle
    with GenericEnvClient(base_url=ENV_URL).sync() as env_client:
        for task in tasks:
            avg_reward, nfz_total = run_task(env_client, llm_client, task, env_name)
            all_results[task] = {
                "avg_reward":     avg_reward,
                "nfz_violations": nfz_total,
            }
            time.sleep(1.0)   # brief pause between tasks

    # Final summary to stderr (not part of grader log)
    print("\n[SUMMARY]", file=sys.stderr)
    for task, r in all_results.items():
        print(
            f"  {task:6s} | avg_reward={r['avg_reward']:.4f}"
            f" | nfz_violations={r['nfz_violations']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()