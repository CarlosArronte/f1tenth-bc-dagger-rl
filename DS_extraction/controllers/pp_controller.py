import numpy as np
from numba import njit
from pyglet.gl import GL_POINTS

from controllers.base_controller import BaseController


# ============================================================
# NUMBA HELPERS
# ============================================================

@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    diffs = trajectory[1:] - trajectory[:-1]
    l2s = diffs[:, 0]**2 + diffs[:, 1]**2

    dots = np.empty(len(l2s))
    for i in range(len(dots)):
        dots[i] = np.dot(point - trajectory[i], diffs[i])

    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0

    projections = trajectory[:-1] + (t[:, None] * diffs)

    dists = np.empty(len(projections))
    for i in range(len(dists)):
        d = point - projections[i]
        dists[i] = np.sqrt(np.dot(d, d))

    idx = np.argmin(dists)
    return projections[idx], dists[idx], t[idx], idx


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(
    point, radius, trajectory, t=0.0, wrap=False
):
    start_i = int(t)
    start_t = t % 1.0

    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i]
        end = trajectory[i + 1] + 1e-6
        V = end - start

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) \
            - 2.0 * np.dot(start, point) - radius**2

        disc = b*b - 4*a*c
        if disc < 0:
            continue

        disc = np.sqrt(disc)
        t1 = (-b - disc) / (2*a)
        t2 = (-b + disc) / (2*a)

        if i == start_i:
            if 0 <= t1 <= 1 and t1 >= start_t:
                return start + t1*V, i, t1
            if 0 <= t2 <= 1 and t2 >= start_t:
                return start + t2*V, i, t2
        else:
            if 0 <= t1 <= 1:
                return start + t1*V, i, t1
            if 0 <= t2 <= 1:
                return start + t2*V, i, t2

    if wrap:
        for i in range(start_i):
            start = trajectory[i]
            end = trajectory[i + 1] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) \
                - 2.0 * np.dot(start, point) - radius**2

            disc = b*b - 4*a*c
            if disc < 0:
                continue

            disc = np.sqrt(disc)
            t1 = (-b - disc) / (2*a)
            t2 = (-b + disc) / (2*a)

            if 0 <= t1 <= 1:
                return start + t1*V, i, t1
            if 0 <= t2 <= 1:
                return start + t2*V, i, t2

    return None, None, None


@njit(fastmath=False, cache=True)
def get_actuation(theta, lookahead_point, position, Ld, wheelbase):
    waypoint_y = np.dot(
        np.array([np.sin(-theta), np.cos(-theta)]),
        lookahead_point[:2] - position,
    )

    speed = lookahead_point[2]

    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0

    radius = Ld**2 / (2.0 * waypoint_y)
    steering = np.arctan(wheelbase / radius)

    return speed, steering


# ============================================================
# PURE PURSUIT CONTROLLER
# ============================================================

class PurePursuitController(BaseController):
    def __init__(self, conf, wheelbase):
        self.conf = conf
        self.wheelbase = wheelbase
        self.max_reacquire = 20.0
        self.drawn_waypoints = []

        self.waypoints = np.loadtxt(
            conf.wpt_path,
            delimiter=conf.wpt_delim,
            skiprows=conf.wpt_rowskip,
        )

        self.wpts_xy = np.vstack(
            (self.waypoints[:, conf.wpt_xind],
             self.waypoints[:, conf.wpt_yind])
        ).T

    def render_waypoints(self, env_renderer):
        scaled = 50.0 * self.wpts_xy

        for i in range(self.wpts_xy.shape[0]):
            if len(self.drawn_waypoints) < self.wpts_xy.shape[0]:
                b = env_renderer.batch.add(
                    1,
                    GL_POINTS,
                    None,
                    ('v3f/stream', [scaled[i, 0], scaled[i, 1], 0.0]),
                    ('c3B/stream', [183, 193, 222]),
                )
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [
                    scaled[i, 0], scaled[i, 1], 0.0
                ]

    def plan(self, pose_x, pose_y, pose_theta, *, tlad, vgain):
        position = np.array([pose_x, pose_y])

        nearest, dist, t, idx = nearest_point_on_trajectory(
            position, self.wpts_xy
        )

        if dist < tlad:
            p, i2, _ = first_point_on_trajectory_intersecting_circle(
                position, tlad, self.wpts_xy, idx + t, wrap=True
            )
            if p is None:
                return 0.0, 0.0

            wp = np.array([p[0], p[1],
                           self.waypoints[i2, self.conf.wpt_vind]])
        elif dist < self.max_reacquire:
            wp = np.array([self.wpts_xy[idx, 0],
                           self.wpts_xy[idx, 1],
                           self.waypoints[idx, self.conf.wpt_vind]])
        else:
            return 0.0, 0.0

        speed, steer = get_actuation(
            pose_theta, wp, position, tlad, self.wheelbase
        )

        return vgain * speed, steer
