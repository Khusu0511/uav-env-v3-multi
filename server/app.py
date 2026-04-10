import os
import sys
from fastapi import Response
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
# Create the OpenEnv FastAPI app via create_app — handles /reset, /step, /state
# env_name must match the name used in inference.py log_start() calls.
# ---------------------------------------------------------------------------

app = create_app(
    UavEnvironment,
    UAVAction,
    UAVObservation,
    env_name="uav_env_v3_multi",
    max_concurrent_envs=1,
)

@app.get("/reset")
async def reset_get():
    """Liveness probe used by the hackathon validator."""
    return JSONResponse({"status": "ok", "env_name": "uav_env_v3_multi"})


@app.get("/render")
async def get_frame():
    """Returns the current 3D fleet frame as a PNG."""
    try:
        env = shared.active_env
        if env is None:
            return Response(status_code=404,
                            content="Environment not initialised. Call /reset first.")
        frame_array = env.render()
        img = Image.fromarray(frame_array)
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        img.close()
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        print(f"[ERROR] Render failed: {e}")
        return Response(status_code=500, content=f"Render Error: {str(e)}")


@app.get("/health")
async def health():
    """Health check — confirms server is up and whether an env is active."""
    env = shared.active_env
    if env is None:
        return JSONResponse({"status": "ok", "environment": "Waiting"})
    return JSONResponse({
        "status":          "ok",
        "environment":     "Active",
        "step":            int(env.current_step),
        "task":            env.current_task,
        "num_agents":      env.num_agents,
        "obs_size":        48,
        "nfz_count":       len(env.nfz_centers),
        "nfz_hard_radius": env.nfz_hard_radius,
        "nfz_buffer_radius": env.nfz_buffer_radius,
        "wind":            env.wind.tolist(),
    })


@app.get("/nfz_status")
async def nfz_status():
    """Returns per-UAV NFZ compliance status."""
    import numpy as np
    env = shared.active_env
    if env is None:
        return Response(status_code=404, content="Environment not initialised.")
    report = []
    for i in range(env.num_agents):
        violations = []
        for j, center in enumerate(env.nfz_centers):
            d = float(np.linalg.norm(env.uav_pos[i] - center))
            violations.append({
                "nfz_index":          j,
                "distance_to_center": round(d, 2),
                "hard_radius":        env.nfz_hard_radius,
                "buffer_radius":      env.nfz_buffer_radius,
                "hard_violation":     d < env.nfz_hard_radius,
                "buffer_warning":     d < env.nfz_buffer_radius,
            })
        report.append({
            "uav_index": i,
            "position":  [round(x, 2) for x in env.uav_pos[i].tolist()],
            "nfz_checks": violations,
        })
    return JSONResponse({"uav_nfz_report": report})

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()