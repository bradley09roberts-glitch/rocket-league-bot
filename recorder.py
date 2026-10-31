"""Match recorder for FourCans: 60 rows a second of what the bot saw, decided and pressed, written to
recordings/<date>-<time>-<bot name>-<index>.csv next to this file. One row per frame makes whiffs,
late challenges and bad positioning visible from real games instead of guessed at."""
import csv
import os
import time

import numpy as np

COLUMNS = ["t", "state", "move", "x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz", "up_z", "boost", "on_ground",
           "ball_x", "ball_y", "ball_z", "ball_vx", "ball_vy", "ball_vz", "pred_x", "pred_y", "pred_z", "pred_t",
           "reachable", "my_time", "foe_time", "foe_x", "foe_y", "foe_z", "target_x", "target_y", "target_z",
           "throttle", "steer", "pitch", "yaw", "roll", "jump", "boost_btn", "handbrake", "we_touched",
           "last_toucher", "score"]


class Recorder:
    def __init__(self, name, index):
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
        os.makedirs(folder, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in name)
        self.path = os.path.join(folder, f"{time.strftime('%Y%m%d-%H%M%S')}-{safe_name}-{index}.csv")
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(COLUMNS)
        self.last_row, self.last_flush = -1.0, time.monotonic()
        self.jumped = self.touched = False   # one-frame events between rows stick until the next row

    def log(self, bot, c):
        self.jumped, self.touched = self.jumped or bool(c.jump), self.touched or bool(bot.touched)
        if bot.time < self.last_row:      # the game clock restarted (a new match): keep recording
            self.last_row = -1.0
        if bot.time - self.last_row < 1 / 60 - 1e-4:
            return
        self.last_row = bot.time
        me = bot.me
        foe = min(bot.foes, key=lambda f: np.linalg.norm(f.pos - bot.ball_pos), default=None)
        foe_pos = foe.pos if foe is not None else (np.nan, np.nan, np.nan)
        move = bot.routine_name if bot.routine else bot.mechanic
        nums = [*me.pos, *me.vel, *me.forward, me.up[2], me.boost, int(me.on_ground), *bot.ball_pos, *bot.ball_vel,
                *bot.predicted_ball_pos, bot.target_t, int(bot.reachable), bot.my_time, bot.foe_time, *foe_pos,
                *bot.target_point, c.throttle, c.steer, c.pitch, c.yaw, c.roll, int(self.jumped),
                int(bool(c.boost)), int(bool(c.handbrake)), int(self.touched), bot.last_toucher]
        self.writer.writerow([f"{bot.time:.3f}", bot.state, move or "-"] + [f"{float(v):.2f}" for v in nums]
                             + ["-".join(str(s) for s in bot.score)])
        self.jumped = self.touched = False
        if time.monotonic() - self.last_flush > 2.0:   # so a crash or a killed game still leaves a usable file
            self.file.flush()
            self.last_flush = time.monotonic()

    def close(self):
        self.file.close()
