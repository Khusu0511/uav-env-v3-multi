import numpy as np
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import UAVAction, UAVObservation
from openenv.core.env_server import Environment
from server import shared


class UavEnvironment(Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_agents = 3
        self.grid_size = np.array([500.0, 500.0, 300.0])
        self.max_speed = 22.0
        self.cmd_scale = 8.0
        self.target_speed = 6.0
        self.dt = 0.5

        # Wind (OU Process)
        self.wind_strength = 3.0
        self.wind_smooth_alpha = 0.92
        self._wind_ou_state = np.zeros(3)
        self._wind_smoothed = np.zeros(3)
        self.wind = np.zeros(3)

        self.pair_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        self.nfz_centers = [np.array([250.0, 250.0, 150.0])]
        self.nfz_radius = 60.0
        self.nfz_hard_radius = self.nfz_radius
        self.nfz_buffer_radius = self.nfz_radius + 25.0

        self.uav_history = [deque(maxlen=50) for _ in range(3)]
        self.uav_pos = np.zeros((3, 3))
        self.uav_vel = np.zeros((3, 3))
        self.target_pos = np.zeros((3, 3))
        self.target_vel = np.zeros((3, 3))

        self.current_task = "hard"
        self._task_wind_strength = 3.0
        self._task_target_evasive = True
        self._task_nfz_active = True
        self.current_step = 0

        self.last_reward = 0.0
        self._max_raw_reward = 480.0

        shared.active_env = self

    @property
    def state(self) -> UAVObservation:
        return self._get_obs()

    def reset(self, seed=None, options=None, **kwargs) -> UAVObservation:
        """
        Finalized Reset for OpenEnv Multi-mode validation.
        """
        # 1. Handle Seed for Reproducibility
        if seed is not None:
            np.random.seed(seed)

        # 2. Extract Task with explicit fallback
        if isinstance(options, dict):
            task = options.get("task", "hard")
        else:
            task = "hard"
        
        self.current_task = task

        # --- Task Logic ---
        if task == "easy":
            self._task_wind_strength = 0.0
            self._task_target_evasive = False
            self._task_nfz_active = False
        elif task == "medium":
            self._task_wind_strength = 1.5
            self._task_target_evasive = False
            self._task_nfz_active = False
        else:  # hard (default)
            self._task_wind_strength = 3.0
            self._task_target_evasive = True
            self._task_nfz_active = True

        self.wind_strength = self._task_wind_strength
        self.current_step = 0
        
        self.last_reward = 0.0 

        # 3. Reset State Arrays 
        for trail in self.uav_history:
            trail.clear()

        self.uav_pos = self._safe_spawn([50, 50, 50], [150, 150, 120], n=3)
        self.target_pos = self._safe_spawn([300, 300, 100], [450, 450, 250], n=3)
        self.uav_vel = np.zeros((3, 3))
        self.target_vel = np.zeros((3, 3))

        for i in range(3):
            v = np.random.normal(size=3)
            self.target_vel[i] = (v / (np.linalg.norm(v) + 1e-8)) * self.target_speed

        self._wind_ou_state = np.zeros(3)
        self._wind_smoothed = np.zeros(3)
        self.wind = np.zeros(3)

        return self._get_obs()

    def _safe_spawn(self, low, high, n=3):
        """Spawn n positions guaranteed to be outside all NFZ hard radii."""
        positions = []
        for _ in range(n):
            for attempt in range(200):
                pos = np.random.uniform(low, high)
                safe = all(
                    np.linalg.norm(pos - c) > self.nfz_hard_radius + 10.0
                    for c in self.nfz_centers
                )
                if safe:
                    positions.append(pos)
                    break
            else:
                positions.append(np.array(low, dtype=float))
        return np.array(positions)

    def step(self, action: UAVAction) -> UAVObservation:
        self.current_step += 1

        # OU Wind Update
        if self.wind_strength > 0.0:
            noise = np.random.normal(size=3)
            self._wind_ou_state = 0.9 * self._wind_ou_state + 0.5 * noise
            self._wind_smoothed = (
                self.wind_smooth_alpha * self._wind_smoothed
                + (1 - self.wind_smooth_alpha) * self._wind_ou_state
            )
            self.wind = np.clip(self._wind_smoothed, -self.wind_strength, self.wind_strength)
        else:
            self.wind = np.zeros(3)

        cmds = np.array(action.commands).reshape((3, 3))
        total_raw_reward = 0.0

        for i in range(3):
            dist = np.linalg.norm(self.target_pos[i] - self.uav_pos[i])

            # Smooth inertia
            inertia = 0.45 + 0.40 * np.tanh(dist / 30.0)
            desired_vel = cmds[i] * self.cmd_scale
            self.uav_vel[i] = inertia * self.uav_vel[i] + (1.0 - inertia) * desired_vel

            # Soft NFZ repulsion (warning zone)
            if self._task_nfz_active:
                for center in self.nfz_centers:
                    vec = self.uav_pos[i] - center
                    d = np.linalg.norm(vec)
                    if d < self.nfz_buffer_radius:
                        strength = self.max_speed * 1.5 * (1.0 - d / self.nfz_buffer_radius) ** 2
                        self.uav_vel[i] += (vec / (d + 1e-8)) * strength

            # Speed clamp
            speed = np.linalg.norm(self.uav_vel[i])
            if speed > self.max_speed:
                self.uav_vel[i] = (self.uav_vel[i] / speed) * self.max_speed

            # Tentative position update
            next_pos = self.uav_pos[i] + self.uav_vel[i] * self.dt + self.wind * self.dt * 0.3

            # STRICT NFZ ENFORCEMENT: Hard boundary collision response
            nfz_violated = False
            if self._task_nfz_active:
                for center in self.nfz_centers:
                    vec_to_center = next_pos - center
                    d = np.linalg.norm(vec_to_center)
                    if d < self.nfz_hard_radius:
                        nfz_violated = True
                        next_pos = center + (vec_to_center / (d + 1e-8)) * (self.nfz_hard_radius + 0.1)
                        normal = vec_to_center / (d + 1e-8)
                        inward = min(0.0, np.dot(self.uav_vel[i], normal))
                        self.uav_vel[i] -= inward * normal

            # Boundary clamp
            next_pos = np.clip(next_pos, [0, 0, 0], self.grid_size)
            self.uav_pos[i] = next_pos
            self.uav_history[i].append(self.uav_pos[i].copy())

            # TARGET MOVEMENT
            if self._task_target_evasive:
                flee_dir = self.target_pos[i] - self.uav_pos[i]
                flee_dist = np.linalg.norm(flee_dir)
                if flee_dist < 80.0:
                    evasion_weight = 1.0 - (flee_dist / 80.0)
                    flee_unit = flee_dir / (flee_dist + 1e-8)
                else:
                    evasion_weight = 0.0
                    flee_unit = np.zeros(3)

                rand_walk = np.random.normal(0, 1.2, size=3)
                self.target_vel[i] += (1.0 - evasion_weight) * rand_walk + evasion_weight * flee_unit * 3.0
                spd = np.linalg.norm(self.target_vel[i])
                self.target_vel[i] = (self.target_vel[i] / (spd + 1e-8)) * self.target_speed
                next_t = self.target_pos[i] + self.target_vel[i] * self.dt
                for dim in range(3):
                    if next_t[dim] <= 5.0 or next_t[dim] >= self.grid_size[dim] - 5.0:
                        self.target_vel[i][dim] *= -1.0
                self.target_pos[i] += self.target_vel[i] * self.dt

            elif self.current_task == "medium":
                rand_walk = np.random.normal(0, 1.2, size=3)
                self.target_vel[i] += rand_walk
                spd = np.linalg.norm(self.target_vel[i])
                self.target_vel[i] = (self.target_vel[i] / (spd + 1e-8)) * self.target_speed
                next_t = self.target_pos[i] + self.target_vel[i] * self.dt
                for dim in range(3):
                    if next_t[dim] <= 5.0 or next_t[dim] >= self.grid_size[dim] - 5.0:
                        self.target_vel[i][dim] *= -1.0
                self.target_pos[i] += self.target_vel[i] * self.dt
            # else: easy -- target is static

            # REWARD: Three-zone proximity + NFZ penalty
            if dist < 15.0:
                v_err = np.linalg.norm(self.uav_vel[i] - self.target_vel[i])
                reward = 100.0 + 60.0 * np.exp(-v_err / 5.0)
            elif dist < 60.0:
                reward = 20.0 + 80.0 * (1.0 - (dist - 15.0) / 45.0)
            else:
                reward = 20.0 * np.exp(-(dist - 60.0) / 80.0)

            if nfz_violated:
                reward -= 200.0

            if self._task_nfz_active:
                for center in self.nfz_centers:
                    d_nfz = np.linalg.norm(self.uav_pos[i] - center)
                    if d_nfz < self.nfz_buffer_radius:
                        penetration = self.nfz_buffer_radius - d_nfz
                        reward -= 1.5 * (penetration ** 1.2)

            margin = 15.0
            for dim in range(3):
                lo = self.uav_pos[i][dim]
                hi = self.grid_size[dim] - self.uav_pos[i][dim]
                if lo < margin:
                    reward -= 2.0 * (margin - lo)
                if hi < margin:
                    reward -= 2.0 * (margin - hi)

            total_raw_reward += reward

        # Toggle: set to True if you want to use raw rewards
        
        raw_reward = float(total_raw_reward)
        normalized_reward = float(np.clip(total_raw_reward / self._max_raw_reward, 0.0, 1.0))
        USE_RAW_REWARD = False

        self.last_reward = raw_reward if USE_RAW_REWARD else normalized_reward
        
        if USE_RAW_REWARD:
            return UAVObservation(
                features=self._get_obs_list(),
                reward=raw_reward,
                done=False,
                step_count=self.current_step,   # <-- required by ActionLog
            )
        else:
            return UAVObservation(
                features=self._get_obs_list(),
                reward=normalized_reward,
                done=False,
                step_count=self.current_step,   # <-- required by ActionLog
            )

    def _get_obs_list(self):
        obs = []
        for i in range(3):
            nv = [c - self.uav_pos[i] for c in self.nfz_centers]
            near_nfz_vec = nv[np.argmin([np.linalg.norm(v) for v in nv])]
            d_nfz = float(np.linalg.norm(near_nfz_vec))
            obs.extend(
                np.concatenate([
                    self.target_pos[i] - self.uav_pos[i],   # (3) relative target pos
                    self.target_vel[i] - self.uav_vel[i],   # (3) relative velocity
                    self.uav_vel[i],                         # (3) own velocity
                    self.wind,                               # (3) wind vector
                    near_nfz_vec,                            # (3) vector to NFZ center
                    [d_nfz],                                 # (1) scalar distance to NFZ
                ]).tolist()
            )
        return obs  # 16 features x 3 agents = 48 total

    def render(self) -> np.ndarray:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(right=0.75)

        # Wind indicator
        wind_mag = np.linalg.norm(self.wind)
        if wind_mag > 0.1:
            ax.quiver(450, 450, 250,
                      self.wind[0], self.wind[1], self.wind[2],
                      length=40, color='cyan', linewidth=2,
                      label=f'Wind ({wind_mag:.1f} m/s)')
            ax.text(450, 450, 280, f"Wind: {wind_mag:.1f} m/s",
                    color='darkcyan', fontsize=8, fontweight='bold')

        # NFZ spheres (hard = red, buffer = orange)
        if self._task_nfz_active:
            for center in self.nfz_centers:
                for radius, color, alpha in [
                    (self.nfz_hard_radius, 'red', 0.25),
                    (self.nfz_buffer_radius, 'orange', 0.08),
                ]:
                    u, v = np.mgrid[0:2 * np.pi:14j, 0:np.pi:10j]
                    ax.plot_wireframe(
                        radius * np.cos(u) * np.sin(v) + center[0],
                        radius * np.sin(u) * np.sin(v) + center[1],
                        radius * np.cos(v) + center[2],
                        color=color, alpha=alpha
                    )

        # UAVs and Targets
        for i in range(3):
            c = self.pair_colors[i]
            pts = np.array(self.uav_history[i])
            if len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=c, alpha=0.5, linewidth=1.8)
            ax.scatter(*self.uav_pos[i], c=c, s=60, edgecolors='k', label=f'UAV {i + 1}')
            ax.scatter(*self.target_pos[i], c=c, s=80, marker='X', label=f'Target {i + 1}')

        ax.set_xlim(0, 500); ax.set_ylim(0, 500); ax.set_zlim(0, 300)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title(f'Task: {self.current_task.upper()} | Step: {self.current_step}')
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='x-small')

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,)
        )
        plt.close(fig)
        return img[:, :, :3]

    def _get_obs(self) -> UAVObservation:

        avg_dist = 0.0
        if len(self.uav_pos) > 0:
            dists = [np.linalg.norm(self.uav_pos[i] - self.target_pos[i]) for i in range(3)]
            avg_dist = float(np.mean(dists))

        # Gather LIVE info for the metadata field
        info = {
            "task": self.current_task,
            "current_step": int(self.current_step),  # Changes every step
            "avg_target_distance": round(avg_dist, 2), # Changes as UAVs move
            "wind_magnitude": round(float(np.linalg.norm(self.wind)), 2), # Changes due to OU process
            "nfz_active": self._task_nfz_active,
        }

        return UAVObservation(
            features=self._get_obs_list(),
            reward=self.last_reward,
            step_count=self.current_step,   # <-- required by ActionLog
            metadata=info
        )