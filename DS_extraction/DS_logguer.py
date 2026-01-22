import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator

from controllers.base_controller import BaseController
from controllers.pp_controller import PurePursuitController


# ============================================================
# CONTROLLER FACTORY (Open–Closed Principle)
# ============================================================

def create_controller(conf) -> BaseController:
    return PurePursuitController(
        conf=conf,
        wheelbase=0.17145 + 0.15875,
    )


# ============================================================
# RENDER CALLBACK
# ============================================================

def build_render_callback(controller: BaseController):
    def render_callback(env_renderer):
        e = env_renderer

        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]

        e.left = min(x) - 800
        e.right = max(x) + 800
        e.bottom = min(y) - 800
        e.top = max(y) + 800

        controller.render_waypoints(e)

    return render_callback


# ============================================================
# MAIN
# ============================================================

def main():
    work = {
        'tlad': 0.82461887897713965,
        'vgain': 1.375,
    }

    with open('config_map.yaml') as f:
        conf = Namespace(**yaml.safe_load(f))

    controller = create_controller(conf)

    env = gym.make(
        'f110_gym:f110-v0',
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
    )

    env.add_render_callback(build_render_callback(controller))

    obs, _, done, _ = env.reset(
        np.array([[conf.sx, conf.sy, conf.stheta]])
    )

    env.render()

    laptime = 0.0
    start = time.time()

    while not done:
        speed, steer = controller.plan(
            obs['poses_x'][0],
            obs['poses_y'][0],
            obs['poses_theta'][0],
            tlad=work['tlad'],
            vgain=work['vgain'],
        )

        obs, step_reward, done, _ = env.step(
            np.array([[steer, speed]])
        )

        laptime += step_reward
        env.render(mode='human')

    print(
        'Sim elapsed time:', laptime,
        'Real elapsed time:', time.time() - start,
    )


if __name__ == '__main__':
    main()
