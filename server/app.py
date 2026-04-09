import os
import sys
import numpy as np
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image

# --- ROOT PATH INJECTION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from server import shared
from openenv.core.env_server import create_app
from models import UAVAction, UAVObservation
from server.uav_env_environment import UavEnvironment

# ---------------------------------------------------------------------------
# Create the OpenEnv FastAPI app.
# env_name must match the name used in inference.py log_start() calls.
# ---------------------------------------------------------------------------
app = create_app(
    UavEnvironment,
    UAVAction,
    UAVObservation,
    env_name            = "uav_env_v3_multi",
    max_concurrent_envs = 1,
)


# ---------------------------------------------------------------------------
# /render — returns the current 3-D frame as PNG
# ---------------------------------------------------------------------------
@app.get("/render")
async def get_frame():
    """
    Returns the current fleet frame as a PNG.
    Visualises:
      - 3 UAV-Target pairs (unique colours + labels)
      - Curved trajectory trails
      - Red NFZ hard-boundary + orange buffer sphere wireframes
      - Wind indicator arrow
    """
    try:
        env = shared.active_env
        if env is None:
            return Response(
                status_code=404,
                content="Fleet environment not yet initialised. Call /reset first.",
            )

        frame_array = env.render()
        img = Image.fromarray(frame_array)
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        img.close()
        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        print(f"[ERROR] Render route failed: {e}")
        return Response(status_code=500, content=f"Render Error: {str(e)}")


# ---------------------------------------------------------------------------
# /health — confirms server is up and whether an env is active
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    env = shared.active_env
    if env is None:
        return JSONResponse({"status": "ok", "environment": "Waiting"})

    details = {
        "step":               int(env.current_step),
        "task":               env.current_task,
        "num_agents":         env.num_agents,
        "obs_size":           48,
        "nfz_count":          len(env.nfz_centers),
        "nfz_hard_radius":    env.nfz_hard_radius,
        "nfz_buffer_radius":  env.nfz_buffer_radius,
        "wind":               env.wind.tolist(),
    }
    return JSONResponse({"status": "ok", "environment": "Active", **details})


# ---------------------------------------------------------------------------
# /nfz_status — per-UAV NFZ compliance report
# ---------------------------------------------------------------------------
@app.get("/nfz_status")
async def nfz_status():
    """
    Returns per-UAV NFZ compliance status.
    Used by inference.py to count hard violations every 10 steps.
    """
    env = shared.active_env
    if env is None:
        return Response(status_code=404, content="Environment not initialised.")

    report = []
    for i in range(env.num_agents):
        violations = []
        for j, center in enumerate(env.nfz_centers):
            d = float(np.linalg.norm(env.uav_pos[i] - center))
            violations.append({
                "nfz_index":           j,
                "distance_to_center":  round(d, 2),
                "hard_radius":         env.nfz_hard_radius,
                "buffer_radius":       env.nfz_buffer_radius,
                "hard_violation":      d < env.nfz_hard_radius,
                "buffer_warning":      d < env.nfz_buffer_radius,
            })
        report.append({
            "uav_index": i,
            "position":  [round(x, 2) for x in env.uav_pos[i].tolist()],
            "nfz_checks": violations,
        })

    return JSONResponse({"uav_nfz_report": report})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()