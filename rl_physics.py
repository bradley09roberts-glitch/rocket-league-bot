"""Arena, ball and car physics for FourCans: constants, the ball predictor, the car model and travel-time
estimates. Everything is in Unreal units (1 uu = 1 cm) and seconds."""
import math

import numpy as np


UU = 100.0                                     # Unreal units per metre
G = -650.0                                     # gravity, uu/s^2
GRAVITY = np.array([0.0, 0.0, G])
WORLD_UP = np.array([0.0, 0.0, 1.0])
BALL_R = 92.75
FIELD_X, FIELD_Y, CEILING, CORNER = 4096.0, 5120.0, 2044.0, 8064.0  # CORNER: |x| + |y| of the 45° walls
GOAL_HALF_W, GOAL_H = 892.755, 642.775
BALL_DRAG, RESTITUTION, BOUNCE_GRIP = 0.0305, 0.6, 0.95
AIR_BOOST_ACC, JUMP_IMPULSE, JUMP_HOLD_ACC, JUMP_HOLD_TIME = 1058.0, 291.667, 1458.333, 0.2
CONTACT = BALL_R + 60.0                        # car centre to ball centre when the nose touches the ball

_D = math.sqrt(0.5)
# Arena surfaces as rows (nx, ny, nz, d): the ball centre must keep n.p - d >= BALL_R. Rows 4 and 5 are
# the back walls, which have a goal mouth in them.
PLANES = np.array([
    [0, 0, 1, 0], [0, 0, -1, -CEILING],                        # floor, ceiling
    [1, 0, 0, -FIELD_X], [-1, 0, 0, -FIELD_X],                 # side walls
    [0, 1, 0, -FIELD_Y], [0, -1, 0, -FIELD_Y],                 # back walls
    [_D, _D, 0, -CORNER * _D], [_D, -_D, 0, -CORNER * _D],     # the four 45° corners
    [-_D, _D, 0, -CORNER * _D], [-_D, -_D, 0, -CORNER * _D],
])
BACK_WALLS = (4, 5)


def vec3(v):
    return np.array([v.x, v.y, v.z], dtype=float)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros(3)


def flat(v):
    return np.array([v[0], v[1], 0.0])


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


TURNS = [(0.0, 0.0069), (500.0, 0.00398), (1000.0, 0.00235), (1500.0, 0.001375), (1750.0, 0.0011), (2300.0, 0.00088)]


def turn_speed(curvature):
    """Fastest ground speed at which the car can still turn this tightly (curvature = 1 / radius, per uu)."""
    for (v0, k0), (v1, k1) in zip(TURNS, TURNS[1:]):
        if curvature >= k1:
            return v0 + (v1 - v0) * clamp((k0 - curvature) / (k0 - k1), 0.0, 1.0)
    return 2300.0


def jump_time(height, double):
    """Seconds for a held (optionally double) jump to lift the car's centre by `height`, or None if it can't."""
    z, vz, t, dt = 0.0, JUMP_IMPULSE, 0.0, 1 / 120
    while vz > 0:
        if z >= height:
            return t
        vz += (G + (JUMP_HOLD_ACC if t < JUMP_HOLD_TIME else 0.0)) * dt
        if double and JUMP_HOLD_TIME <= t < JUMP_HOLD_TIME + dt:
            vz += JUMP_IMPULSE
        z += vz * dt
        t += dt
    return None


def predict_ball(pos, vel, horizon, dt=1 / 60):
    """Step the ball forward under gravity and air drag, bouncing off the floor, ceiling, all four walls
    and the corners. Returns (times, positions, velocities). A ball that enters a net stops there."""
    p, v = pos.copy(), vel.copy()
    normals, offsets = PLANES[:, :3], PLANES[:, 3]
    times, ps, vs = [], [], []
    for step in range(1, int(horizon / dt) + 1):
        if abs(p[1]) < FIELD_Y + BALL_R:
            v = (v + GRAVITY * dt) * (1 - BALL_DRAG * dt)
            p = p + v * dt
            gaps = normals @ p - offsets
            for k in np.nonzero(gaps < BALL_R)[0]:
                if k in BACK_WALLS and abs(p[0]) < GOAL_HALF_W - BALL_R / 2 and p[2] < GOAL_H - BALL_R / 2:
                    continue  # that part of the back wall is a goal mouth
                n, vn = normals[k], v @ normals[k]
                p = p + n * (BALL_R - gaps[k])
                if vn < -100:   # a real bounce: lose normal speed to restitution, a little grip on the rest
                    v = (v - vn * n) * BOUNCE_GRIP - vn * RESTITUTION * n
                elif vn < 0:    # rolling or sliding along the surface
                    v = v - vn * n
        else:
            v = np.zeros(3)
        times.append(step * dt)
        ps.append(p)
        vs.append(v)
    return np.array(times), np.array(ps), np.array(vs)


class Car:
    """One car from the packet. Orientation rows are (forward, right, up) in world coordinates."""

    def __init__(self, info, index):
        phys = info.physics
        self.index, self.team, self.boost = index, info.team, info.boost
        self.pos, self.vel, self.angvel = vec3(phys.location), vec3(phys.velocity), vec3(phys.angular_velocity)
        self.on_ground, self.demoed = info.has_wheel_contact, info.is_demolished
        p, y, r = phys.rotation.pitch, phys.rotation.yaw, phys.rotation.roll
        cp, sp, cy, sy, cr, sr = math.cos(p), math.sin(p), math.cos(y), math.sin(y), math.cos(r), math.sin(r)
        self.ori = np.array([[cp * cy, cp * sy, sp],
                             [cy * sp * sr - cr * sy, sy * sp * sr + cr * cy, -cp * sr],
                             [-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr]])
        self.forward, self.right, self.up = self.ori

    def local(self, v):
        """World vector -> (forward, right, up) components."""
        return self.ori @ v


def eta(car, targets):
    """Rough seconds for `car` to reach each row of `targets`: accelerate, cruise, turn, and climb."""
    to = targets - car.pos
    to[:, 2] = 0.0
    span = np.linalg.norm(to, axis=1)
    dirs = to / np.maximum(span, 1.0)[:, None]
    dist = np.maximum(span - CONTACT, 0.0)
    vmax, acc = (2300.0, 1000.0) if car.boost > 10 else (1410.0, 500.0)
    v0 = np.clip(dirs @ car.vel, 0.0, vmax)
    t_acc = (vmax - v0) / acc
    d_acc = (v0 + vmax) / 2 * t_acc
    drive = np.where(dist < d_acc, (np.sqrt(v0 ** 2 + 2 * acc * dist) - v0) / acc, t_acc + (dist - d_acc) / vmax)
    if car.on_ground:
        drive = drive + np.arccos(np.clip(dirs @ unit(flat(car.forward)), -1.0, 1.0)) * 0.35
    climb = np.sqrt(2 * np.maximum(targets[:, 2] - 150.0, 0.0) / 900.0)
    grounded = np.where((targets[:, 2] > 600) & (car.boost < 20), np.inf, 0.0)  # no boost, no aerials
    return np.maximum(drive, climb) + grounded


def first_reachable(car, times, path):
    ok = np.nonzero(eta(car, path) <= times)[0]
    return int(ok[0]) if len(ok) else None
