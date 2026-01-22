#!/usr/bin/env python3
"""
rt_inference_model_control.py
Inferencia en tiempo real en F1TENTH Gym usando un modelo PyTorch entrenado.
- Carga rl_policy_offline.pth (checkpoint con "state_dict", "state_columns", "action_columns")
- Construye vector de entrada con las mismas columnas usadas en entrenamiento (14)
- Ejecuta inferencia y envía steer, v al env.step
"""

import time
import yaml
import gym
import numpy as np
import torch
import torch.nn as nn
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
import math
import csv
import os

# ------------------------------
# Utils: cálculo de curvatura simple similar al dataset
# ------------------------------
def calculate_average_curvature_simple(wpts, i, window_size=5):
    n = wpts.shape[0]
    start_idx = max(0, i - window_size)
    end_idx = min(n, i + window_size + 1)
    indices = np.arange(start_idx, end_idx) % n
    window_points = wpts[indices]
    if len(window_points) < 3:
        return 0.0
    vectors = np.zeros((len(window_points) - 1, 2))
    for j in range(len(window_points) - 1):
        vectors[j] = window_points[j + 1] - window_points[j]
    lengths = np.sqrt(np.sum(vectors ** 2, axis=1))
    if np.any(lengths < 1e-6):
        return 0.0
    directions = vectors / lengths[:, np.newaxis]
    curvatures = []
    for j in range(len(directions) - 1):
        v1 = directions[j]
        v2 = directions[j + 1]
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        curvature = angle / lengths[j]
        curvatures.append(curvature)
    if not curvatures:
        return 0.0
    return float(np.mean(curvatures))

# ------------------------------
# Simple nearest waypoint helper
# ------------------------------
def nearest_waypoint_index(waypoints, pos):
    dists = np.linalg.norm(waypoints - pos, axis=1)
    return int(np.argmin(dists))

# ------------------------------
# Modelo dinámico (con input_dim desde checkpoint)
# ------------------------------
class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden=128, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------
# MAIN
# ------------------------------
def main():
    # Paths (ajusta si hace falta)
    CHECKPOINT_PATH = "model.pth"   # checkpoint guardado por tu script de entrenamiento
    CONFIG_PATH = "config_example_map.yaml"    # tu config
    TELEMETRY_OUT = "telemetry_rt.csv"         # opcional

    # -------------------------
    # Cargar checkpoint
    # -------------------------
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint no encontrado: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # por compatibilidad si el checkpoint es el state_dict directamente
        state_dict = checkpoint

    state_columns = checkpoint.get("state_columns", None)
    action_columns = checkpoint.get("action_columns", None)
    if state_columns is None or action_columns is None:
        raise RuntimeError("Checkpoint no contiene 'state_columns' o 'action_columns' necesarios.")

    input_dim = len(state_columns)
    print("[INFO] state_columns (input order):", state_columns)
    print("[INFO] action_columns (output order):", action_columns)
    print("[INFO] input dim:", input_dim)

    # -------------------------
    # Construir y cargar modelo
    # -------------------------
    model = PolicyNet(input_dim=input_dim, output_dim=len(action_columns))
    model.load_state_dict(state_dict)
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    print("[INFO] Modelo cargado y en modo eval.")

    # -------------------------
    # Cargar config
    # -------------------------
    with open(CONFIG_PATH) as f:
        conf = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    # -------------------------
    # Crear entorno
    # -------------------------

    def render_callback(env_renderer):
        e = env_renderer
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800
        

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                   num_agents=1, timestep=0.01, integrator=Integrator.RK4, render_mode="human")
   
    # actual reset initial pose
    obs, _, _, _ = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))

    env.add_render_callback(render_callback)

    env.render(mode='human')



    # Cargar waypoints (planner uses same)
    wpts = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
    wpts_xy = np.vstack((wpts[:, conf.wpt_xind], wpts[:, conf.wpt_yind])).T

    # -------------------------
    # Variables de estado/tiempo para construir las 14 features
    # -------------------------
    prev_yaw = None
    prev_time = time.time()
    laptime = 0.0
    segment_time = 0.0
    lap_count = 1
    prev_waypoint_idx = -1
    dt = 0.01  # coincidente con el env timestep
    prev_steer_cmd = 0.0
    prev_throttle_cmd = 0.0
    total_steps = 0

    # Telemetry file header
    telemetry_file = open(TELEMETRY_OUT, "w", newline="")
    csv_writer = csv.writer(telemetry_file)
    csv_writer.writerow(state_columns + action_columns + ["timestamp"])

    # -------------------------
    # Main loop
    # -------------------------
    done = False
    crash_count = 0

    print("[INFO] Iniciando loop de inferencia en tiempo real... (Ctrl+C para parar)")

    try:
        while True:
            # Step environment with last commanded action (initially zeros)
            action_to_send = np.array([[prev_steer_cmd, prev_throttle_cmd]])
            obs, _, done, info = env.step(action_to_send)
            env.render(mode='human')

            # Time bookkeeping
            current_time = time.time()
            elapsed = current_time - prev_time
            prev_time = current_time
            # use env timestep (preferred deterministic)
            # dt = env.timestep if hasattr(env, 'timestep') else 0.01
            laptime += dt
            segment_time += dt
            total_steps += 1

            # Observations (adaptado a estructura del gym que usaste para generar dataset)
            # Tu script generador usó obs['poses_x'][0], etc.
            x = float(obs['poses_x'][0])
            y = float(obs['poses_y'][0])
            yaw = float(obs['poses_theta'][0])
            v = float(np.sqrt(obs['linear_vels_x'][0]**2 + obs['linear_vels_y'][0]**2))

            # slip angle (como en tu generador)
            slip_angle = float(np.arctan2(obs['linear_vels_y'][0], obs['linear_vels_x'][0] + 1e-6) - yaw)
            slip_angle = float(np.clip(slip_angle, -np.pi/2, np.pi/2))

            # omega yaw (numérico)
            if prev_yaw is not None:
                omega_yaw = float((yaw - prev_yaw) / dt)
            else:
                omega_yaw = 0.0
            prev_yaw = yaw

            # roll estimation (igual que en tu script)
            L = (conf.lf if hasattr(conf, 'lf') else 0.15875) + (conf.lr if hasattr(conf, 'lr') else 0.15875)
            try:
                r = L / (math.tan(max(min(prev_steer_cmd, 0.5), -0.5)) + 1e-6)
                roll = float(math.atan(v**2 / (9.81 * max(abs(r), 1e-6))))
            except Exception:
                roll = 0.0

            # Waypoint relativo: nearest point + next point
            pos = np.array([x, y])
            idx = nearest_waypoint_index(wpts_xy, pos)
            # compute a "next" waypoint index (i+1)
            next_idx = (idx + 1) % wpts_xy.shape[0]
            next_wpt = wpts_xy[next_idx]
            dx = next_wpt[0] - x
            dy = next_wpt[1] - y
            # rotation to vehicle frame (consistent with generator)
            x_rel_wpt = float(np.cos(-yaw) * dx - np.sin(-yaw) * dy + np.random.normal(0, 0.0))
            y_rel_wpt = float(np.sin(-yaw) * dx + np.cos(-yaw) * dy + np.random.normal(0, 0.0))

            # curvature: use average curvature around nearest index
            curvature = float(calculate_average_curvature_simple(wpts_xy, idx, window_size=5))

            # crash flag from info
            crash_flag = 1 if info.get('collision', False) else 0
            if crash_flag:
                crash_count += 1

            # reward: we don't have the same reward shaping as offline; keep a small proxy (progress)
            # Here we set to 0.0 (alternatively compute progress-based reward if desired)
            reward = 0.0

            # Prepare other fields that were in the dataset
            throttle_input = float(prev_throttle_cmd)   # use previous throttle command as "throttle" input
            crash_input = int(crash_flag)
            laptime_input = float(laptime)
            segment_time_input = float(segment_time)
            delta_t_input = float(dt)
            lap_input = int(lap_count)

            # Compose state vector in EXACT order from checkpoint/state_columns
            state_vec = []
            for col in state_columns:
                if col == 'x_rel_wpt':
                    state_vec.append(x_rel_wpt)
                elif col == 'y_rel_wpt':
                    state_vec.append(y_rel_wpt)
                elif col == 'yaw':
                    state_vec.append(yaw)
                elif col == 'slip_angle':
                    state_vec.append(slip_angle)
                elif col == 'omega_yaw':
                    state_vec.append(omega_yaw)
                elif col == 'roll':
                    state_vec.append(roll)
                elif col == 'throttle':
                    state_vec.append(throttle_input)
                elif col == 'reward':
                    state_vec.append(reward)
                elif col == 'crash':
                    state_vec.append(crash_input)
                elif col == 'laptime':
                    state_vec.append(laptime_input)
                elif col == 'segment_time':
                    state_vec.append(segment_time_input)
                elif col == 'delta_t':
                    state_vec.append(delta_t_input)
                elif col == 'lap':
                    state_vec.append(lap_input)
                elif col == 'curvature':
                    state_vec.append(curvature)
                else:
                    # fallback: zero
                    state_vec.append(0.0)

            x_input = torch.tensor(np.array(state_vec, dtype=np.float32), device=device).unsqueeze(0)  # shape (1, input_dim)

            # Model inference
            with torch.no_grad():
                y_out = model(x_input).cpu().numpy().squeeze(0)

            # Map outputs to actual actions according to checkpoint['action_columns']
            # checkpoint["action_columns"] was e.g. ["v", "steer"]
            if action_columns[0] == "v" and action_columns[1] == "steer":
                speed_pred = float(y_out[0])
                steer_pred = float(y_out[1])
            elif action_columns[0] == "steer" and action_columns[1] == "v":
                steer_pred = float(y_out[0])
                speed_pred = float(y_out[1])
            else:
                # Generic mapping (if action_columns not exactly v/steer)
                mapping = {}
                for i, name in enumerate(action_columns):
                    mapping[name] = float(y_out[i])
                speed_pred = mapping.get("v", mapping.get("speed", 0.0))
                steer_pred = mapping.get("steer", mapping.get("s", 0.0))

            # Saturate/clip to physical limits
            steer_cmd = float(np.clip(steer_pred, -0.4189, 0.4189))
            throttle_cmd = float(np.clip(speed_pred, 0.0, 8.0))

            # Optionally smooth commands (simple exponential smoothing)
            alpha = 0.6
            prev_steer_cmd = alpha * steer_cmd + (1 - alpha) * prev_steer_cmd
            prev_throttle_cmd = alpha * throttle_cmd + (1 - alpha) * prev_throttle_cmd

            # Send action next loop iteration (we already sent prev values at the start of the loop)
            # But for more immediate effect, update prev_* now and next env.step will use them.
            # Logging / telemetry
            row = state_vec + [prev_throttle_cmd, prev_steer_cmd] + [time.time()]
            csv_writer.writerow(row)
            telemetry_file.flush()

            # Detect lap crossing (simple heuristic: if near starting pose and laptime > 1s)
            start_x, start_y = conf.sx, conf.sy
            if (abs(x - start_x) < 0.1 and abs(y - start_y) < 0.1 and laptime > 1.0):
                lap_count += 1
                laptime = 0.0
                segment_time = 0.0
                print(f"[INFO] Vuelta completada: {lap_count}")

            # If many crashes, break
            if crash_count > 10:
                print("[WARN] Múltiples colisiones detectadas, terminando.")
                break

            # Small sleep to avoid busy loop (env.step already enforces timing, but keep tiny sleep)
            # time.sleep(0.0001)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupción por usuario, cerrando...")

    finally:
        telemetry_file.close()
        env.close()
        print("[INFO] Finalizado.")

if __name__ == "__main__":
    main()
