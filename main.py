"""FourCans: a fully autonomous, heuristic Rocket League bot for RLBot v4 (the `rlbot` 1.x Python package).

The bot is the player. It takes kickoffs, shoots, saves, rotates and manages boost with no human input.
A priority state machine (KICKOFF, SAVE, ATTACK, AERIAL, ROTATE) decides what to do, and a handful of
mechanics do it: ground flip-hit, flip-up jump shot, wave volley, and an aerial that tries flip resets.
Distances in CONFIG are metres, as specified; the game itself works in Unreal units (1 uu = 1 cm).
Arena constants, the ball predictor and the car model live next door in rl_physics.py.
"""
import math

import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rl_physics import (AIR_BOOST_ACC, BALL_R, CONTACT, FIELD_X, FIELD_Y, GOAL_HALF_W, GRAVITY, JUMP_HOLD_ACC,
                        JUMP_HOLD_TIME, JUMP_IMPULSE, UU, WORLD_UP, Car, clamp, eta, first_reachable, flat,
                        jump_time, predict_ball, turn_speed, unit, vec3)

CONFIG = {
    "prediction_time": 1.5,        # seconds of ball prediction
    "keeper_clamp_x": 3.8,         # m the keeper may drift off centre (the posts are at ±8.93 m)
    "keeper_clamp_factor": 0.75,   # how much we follow ball.x
    "side_angle_deg": 45,          # approach angle off goal line
    "flip_reset_height_min": 2.0,  # m, below this = ground hit
    "flip_reset_height_max": 6.0,  # m, above this = aerial
    "wave_volley_height_min": 3.0,
    "wave_volley_height_max": 7.0,
    "boost_conserve_below": 15,
    "boost_save_for_kick": 50,
    "pass_distance": 20.0,         # 3v3: pass if teammate is this far from ball
    "rotation_distance": 15.0,     # how far from ball to park in 3v3
    # Knobs the spec describes in prose, pulled up here so they can be tuned too.
    "boost_moderate_below": 40,    # 15-40 boost: still boost to hit, but skip the 45° approach
    "aerial_state_height": 3.0,    # m, AERIAL state trigger
    "ground_jump_time": 0.3,       # s before contact to jump and flip into a ground ball
    "big_pad_detour": 8.0,         # m off our path we'll go for a 100 pad
    "small_pad_detour": 5.0,       # m off our path we'll go for a small pad...
    "small_pad_max_delay": 0.3,    # ...if it costs no more than this many seconds
    "flip_reset_enabled": True,    # point the wheels at the ball on high attacking aerials
    "max_air_touches": 4,          # the 4-tap ceiling
    "reaction_time": 0.12,         # s a non-urgent state change must persist before we act on it
    "debug_render": True,          # state text, a line to our target and a line along the prediction
}

KICKOFF, SAVE, ATTACK, AERIAL, ROTATE = "KICKOFF", "SAVE", "ATTACK", "AERIAL", "ROTATE"
GROUND, FLIP_UP, VOLLEY, FULL_AERIAL = "ground hit", "flip-up", "wave volley", "aerial"


def pd(angle, rate):
    """Air-control stick value from an angle error and its rate (the cubic PD GoslingUtils popularised)."""
    return clamp((35.0 * (angle + rate)) ** 3 / 10.0, -1.0, 1.0)


class Bot(BaseAgent):
    """The whole player. get_output() runs once per game tick (up to 120 times a second)."""

    def initialize_agent(self):
        self.state, self.pending, self.pending_since = ROTATE, None, 0.0
        self.routine, self.routine_name, self.mechanic = None, "", ""
        self.pads, self.score, self.last_touch_time, self.air_touches = [], None, 0.0, 0
        self.target_point = np.zeros(3)

    # ------------------------------------------------------------------ the tick
    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        self.read(packet)
        if self.me.demoed or not (packet.game_info.is_round_active or self.is_kickoff):
            # WHY: during goal replays and demo respawns our inputs do nothing; drop any half-done mechanic
            # so the next kickoff starts clean.
            self.reset_play()
            return SimpleControllerState()
        self.plan()
        controls = self.step_routine() if self.routine else None
        if controls is None:
            # WHY: no mechanic is mid-flight, so we're free to re-decide. While one runs, the state is frozen:
            # changing your mind halfway through a jump is how bots end up looking Bronze.
            self.update_state()
            self.mechanic = ""
            handler = {KICKOFF: self.take_kickoff, SAVE: self.make_save, ATTACK: self.attack,
                       AERIAL: self.attack, ROTATE: self.rotate}[self.state]
            # AERIAL differs from ATTACK only in its trigger; go_hit's height bands already pick an air mechanic.
            controls = self.recover_if_airborne() or handler()
        self.render()
        return controls

    def read(self, packet):
        self.time = packet.game_info.seconds_elapsed
        cars = [Car(packet.game_cars[i], i) for i in range(packet.num_cars)]
        self.me = cars[self.index]
        self.mates = [c for c in cars if c.team == self.team and c.index != self.index and not c.demoed]
        self.foes = [c for c in cars if c.team != self.team and not c.demoed]
        ball = packet.game_ball
        self.ball_pos, self.ball_vel = vec3(ball.physics.location), vec3(ball.physics.velocity)
        # The ball only ever sits dead still on the centre spot before a kickoff.
        self.is_kickoff = abs(self.ball_pos[0]) < 1 and abs(self.ball_pos[1]) < 1 and np.linalg.norm(self.ball_vel) < 1
        touch = ball.latest_touch
        self.touched = touch.player_index == self.index and touch.time_seconds != self.last_touch_time
        self.last_touch_time = touch.time_seconds
        self.goal_sign = -1.0 if self.team == 0 else 1.0          # y sign of the goal we defend
        self.our_goal = np.array([0.0, self.goal_sign * FIELD_Y, 0.0])
        if not self.pads:
            info = self.get_field_info()
            self.pads = [(vec3(info.boost_pads[i].location), info.boost_pads[i].is_full_boost)
                         for i in range(info.num_boosts)]
        self.pad_active = [packet.game_boosts[i].is_active for i in range(min(len(self.pads), packet.num_boost))]
        score = tuple(packet.teams[i].score for i in range(packet.num_teams))
        if score != self.score:
            # WHY: somebody scored. Whatever we were doing is void; the next thing that matters is the kickoff.
            self.score = score
            self.reset_play()

    def reset_play(self):
        self.routine, self.routine_name, self.air_touches = None, "", 0

    def plan(self):
        """Predict the ball, find where we can first meet it, and work out who else gets there first."""
        self.times, self.path, self.path_vel = predict_ball(self.ball_pos, self.ball_vel, CONFIG["prediction_time"])
        in_net = np.nonzero(np.abs(self.path[:, 1]) > FIELD_Y)[0]
        self.on_target = len(in_net) > 0 and self.goal_sign * self.path[in_net[0], 1] > 0
        crossing_x = self.path[in_net[0], 0] if len(in_net) else 0.0
        if len(in_net):  # once it's in a net there is nothing left to hit: plan only on the live part
            live = max(int(in_net[0]), 1)
            self.times, self.path, self.path_vel = self.times[:live], self.path[:live], self.path_vel[:live]
        self.my_time, i = self.arrival(self.me)
        self.reachable = i is not None
        i = i if self.reachable else len(self.times) - 1
        self.predicted_ball_pos, self.predicted_ball_vel, self.target_t = self.path[i], self.path_vel[i], self.times[i]
        self.mate_time = min((self.arrival(c)[0] for c in self.mates), default=math.inf)
        self.foe_time = min((self.arrival(c)[0] for c in self.foes), default=math.inf)
        self.cross_x = crossing_x if self.on_target else self.predicted_ball_pos[0]

    def arrival(self, car):
        """(seconds until `car` can first meet the predicted ball, prediction index or None)."""
        i = first_reachable(car, self.times, self.path)
        if i is None:
            return self.times[-1] + float(eta(car, self.path[-1:])[0]), None
        return self.times[i], i

    # ------------------------------------------------------------------ the state machine
    def zone(self, pos):
        """+1 in our defensive third, -1 in our attacking third, 0 in midfield."""
        depth = self.goal_sign * pos[1]
        return 1 if depth > FIELD_Y / 3 else -1 if depth < -FIELD_Y / 3 else 0

    def choose_state(self):
        ball, me = self.predicted_ball_pos, self.me
        # Once we've committed, a teammate has to be clearly (0.3 s) ahead of us to take the ball off us.
        first_mate = self.my_time <= self.mate_time + (0.3 if self.state in (ATTACK, AERIAL) else 0.0)
        # Over 40 boost we "chase everything": a challenge we arrive at up to 0.3 s late is still ours.
        beat_foe = self.my_time <= self.foe_time + (0.3 if me.boost > CONFIG["boost_moderate_below"] else 0.0)
        if self.is_kickoff:
            # WHY: the table lists KICKOFF last, but it's the only state that means anything while the ball
            # sits still on the centre spot, so it's checked first instead of being outranked.
            return KICKOFF
        if self.on_target or (self.zone(ball) == 1 and self.goal_sign * self.ball_vel[1] > 0):
            # WHY: a shot on target, or the ball is in our third and coming at our net. Nothing else matters.
            return SAVE
        if first_mate and self.zone(ball) == -1:
            # WHY: the ball is in their third and we're our team's first man: go score.
            return ATTACK
        if first_mate and ball[2] > CONFIG["aerial_state_height"] * UU and self.goal_sign * ball[1] > 0:
            # WHY: a high ball on our side that we reach first: win it in the air before it drops to them.
            return AERIAL
        if first_mate and (beat_foe or self.mates):
            # WHY: a midfield or our-half ball we get to first (in 3v3 the first man always pressures).
            # Nobody else on our team is going to take it.
            return ATTACK
        # WHY: someone else gets there first, so be where the ball goes next instead of chasing it.
        return ROTATE

    def update_state(self):
        want = self.choose_state()
        if want == self.state:
            self.pending = None
        elif want in (KICKOFF, SAVE) or self.state == KICKOFF or (
                self.pending == want and self.time - self.pending_since >= CONFIG["reaction_time"]):
            # WHY: kickoffs, saves and the end of a kickoff can't wait. Anything else has to survive a short
            # beat first, so one noisy prediction doesn't make us flip-flop (that, and we're tired).
            self.state, self.pending = want, None
        elif self.pending != want:
            self.pending, self.pending_since = want, self.time

    # ------------------------------------------------------------------ states
    def take_kickoff(self):
        me = self.me
        takers = sorted([me] + self.mates,
                        key=lambda c: (round(np.linalg.norm(flat(c.pos)) / 100), self.goal_sign * c.pos[0]))
        if takers[0] is not me:
            # WHY: one car takes the kickoff (closest, left goes on a tie); the others cover and grab boost.
            return self.rotate()
        # Come in at a slight side angle so the hit deflects the ball instead of making a dead 50/50.
        side = math.copysign(1.0, me.pos[0]) if abs(me.pos[0]) > 50 else 1.0
        target = self.ball_pos + np.array([side * 45.0, self.goal_sign * 30.0, 0.0])
        if np.linalg.norm(flat(target - me.pos)) < 750 and me.vel @ me.forward > 1100:
            return self.start(self.flip_hit(True), "kickoff flip")
        return self.drive_to(target, 2300.0, True)

    def make_save(self):
        ball = self.predicted_ball_pos
        if not self.goalside(self.ball_pos):
            # WHY: we're on the wrong side of the ball; hitting it from here is how own goals happen.
            return self.hold(self.keeper_spot())
        if not self.reachable or not (self.my_time <= self.mate_time or self.on_target):
            # WHY: we can't reach it in time, or a teammate has it: block the line and be the safety net.
            return self.hold(self.keeper_spot())
        # WHY: goal-side and first to it: clear it (go_hit picks flip-clear, bump-clear or air hit by height).
        return self.go_hit(self.clear_aim(ball), "save")

    def attack(self):
        return self.go_hit(self.shot_aim(self.predicted_ball_pos), "hit")

    def rotate(self):
        """Be where the ball goes next: keeper in our third, goal-side and off-ball in midfield, back further up."""
        ball = self.path[min(len(self.path) - 1, 59)]  # roughly where the ball will be in a second
        zone = self.zone(ball)
        if zone == 1:
            # WHY: the ball is in our third and it isn't ours to take: drop to keeper.
            return self.hold(self.keeper_spot())
        away = -math.copysign(1.0, ball[0]) if abs(ball[0]) > 100 else 1.0
        if zone == 0:
            # WHY: midfield: sit goal-side, shaded to the side the ball isn't on, to cover the back post.
            spot = (self.our_goal + ball) / 2
            spot[0] = ball[0] * 0.3 + away * 800.0
        else:
            # WHY: the ball is in their half: park opposite-side, 15 m back at 30°, ready for the clear.
            back, across = unit(flat(self.our_goal - ball)), np.array([away, 0.0, 0.0])
            lean = math.radians(30)
            spot = ball + (back * math.cos(lean) + across * math.sin(lean)) * CONFIG["rotation_distance"] * UU
        spot = np.array([clamp(spot[0], -FIELD_X + 400, FIELD_X - 400),
                         clamp(spot[1], -FIELD_Y + 400, FIELD_Y - 400), 0.0])
        pad = self.pad_on_path(spot)
        target = spot if pad is None else pad
        # Never idle: at least 1000 uu/s, so we loop around the spot like a player keeping momentum.
        speed = clamp(2 * np.linalg.norm(flat(target - self.me.pos)), 1000.0, 2300.0)
        return self.drive_to(target, speed, self.may_boost("position"))

    # ------------------------------------------------------------------ targets
    def goalside(self, pos):
        return self.goal_sign * (self.me.pos[1] - pos[1]) > -200

    def keeper_spot(self):
        limit = CONFIG["keeper_clamp_x"] * UU
        x = clamp(self.cross_x * CONFIG["keeper_clamp_factor"], -limit, limit)
        if self.on_target:
            # A shot is coming: stand where it crosses the line. The 3.8 m shadow clamp would leave the far
            # half of a 17.9 m goal open, so here we only stay inside the posts.
            x = clamp(self.cross_x, -GOAL_HALF_W + 100, GOAL_HALF_W - 100)
        spot = np.array([x, self.goal_sign * (FIELD_Y - 150), 0.0])
        if any(np.linalg.norm(m.pos - spot) < np.linalg.norm(self.me.pos - spot) for m in self.mates):
            # A teammate already has the net; one keeper is enough, so cover the back post a bit higher up.
            spot = np.array([-math.copysign(700.0, x or 1.0), self.goal_sign * (FIELD_Y - 1100), 0.0])
        return spot

    def shot_aim(self, ball):
        their_y = -self.goal_sign * FIELD_Y
        if self.mates and abs(ball[0]) > 2000 and abs(their_y - ball[1]) < 2500:
            # 3v3, deep in their corner: the shot angle is hopeless, so centre it for a teammate who's close.
            for m in self.mates:
                if np.linalg.norm(m.pos - ball) < CONFIG["pass_distance"] * UU and abs(m.pos[0]) < 1500:
                    return np.array([m.pos[0] * 0.5, their_y + self.goal_sign * 1500, 200.0])
        post = GOAL_HALF_W - 2 * BALL_R
        return np.array([clamp(ball[0], -post, post), their_y, 200.0])

    def clear_aim(self, ball):
        """Low balls: hard to the far corner on the ball's own side ("opposite corner" read as the far end of
        the pitch, never across our own box). Anything in the air: popped up and away toward midfield."""
        if ball[2] < CONFIG["flip_reset_height_min"] * UU:
            side = math.copysign(1.0, ball[0]) if abs(ball[0]) > 1 else 1.0
            return np.array([side * (FIELD_X - 500), -self.goal_sign * (FIELD_Y - 1000), 100.0])
        return np.array([0.0, 0.0, 1500.0])

    def approach(self, ball, aim):
        """The contact point behind the ball, reached along a line no more than side_angle_deg off the shot."""
        me = self.me
        shot = unit(flat(aim - ball))
        come_in = unit(flat(ball - me.pos))
        off = math.atan2(shot[0] * come_in[1] - shot[1] * come_in[0], shot @ come_in)
        limit = math.radians(CONFIG["side_angle_deg"])
        if abs(off) <= limit:
            return ball - shot * CONTACT
        # Coming in more than 45° off the shot line knocks it into the side wall; swing out to the 45° line.
        c, s = math.cos(math.copysign(limit, off)), math.sin(math.copysign(limit, off))
        line = np.array([c * shot[0] - s * shot[1], s * shot[0] + c * shot[1], 0.0])
        swing = ball - line * clamp(0.5 * np.linalg.norm(flat(ball - me.pos)), 300.0, 1500.0)
        if np.linalg.norm(flat(swing - me.pos)) < 300:
            return ball - shot * CONTACT   # close enough to the line: commit instead of circling the point
        return swing

    def pad_on_path(self, target, max_delay=None):
        """The boost pad worth a detour on the way to `target` (big within 8 m, small within 5 m), or None."""
        me = self.me
        if me.boost > 85:
            return None
        seg = flat(target - me.pos)
        length = max(np.linalg.norm(seg), 1.0)
        speed = max(np.linalg.norm(me.vel), 1000.0)
        best, best_delay = None, math.inf
        for (loc, big), active in zip(self.pads, self.pad_active):
            rel = flat(loc - me.pos)
            off_path = np.linalg.norm(rel - seg * clamp(rel @ seg / length ** 2, 0.0, 1.0))
            delay = (np.linalg.norm(rel) + np.linalg.norm(flat(target - loc)) - length) / speed
            reach = (CONFIG["big_pad_detour"] if big else CONFIG["small_pad_detour"]) * UU
            cap = (math.inf if max_delay is None else max_delay) if big else CONFIG["small_pad_max_delay"]
            if active and off_path < reach and delay <= cap and delay < best_delay:
                best, best_delay = loc, delay
        return best

    def may_boost(self, purpose):
        """Boost tiers: saves always; positioning holds 50+; under 15 only for a ball in our half."""
        boost = self.me.boost
        if purpose == "save":
            return True
        if purpose == "position":
            return boost > CONFIG["boost_save_for_kick"]
        if boost < CONFIG["boost_conserve_below"]:
            return self.goal_sign * self.predicted_ball_pos[1] > 0
        return True

    # ------------------------------------------------------------------ hitting the ball
    def go_hit(self, aim, purpose):
        """Meet the predicted ball and send it toward `aim`, with the mechanic its height calls for."""
        me, ball, T = self.me, self.predicted_ball_pos, self.target_t
        h = ball[2]
        if h < CONFIG["flip_reset_height_min"] * UU:
            mech = GROUND
        elif h <= CONFIG["flip_reset_height_max"] * UU and self.predicted_ball_vel[2] < 0:
            mech = FLIP_UP
        elif CONFIG["wave_volley_height_min"] * UU <= h <= CONFIG["wave_volley_height_max"] * UU and me.pos[2] < h:
            mech = VOLLEY
            falling = np.nonzero((self.times >= T) & (self.path_vel[:, 2] < 0))[0]
            if len(falling):  # a volley is struck on the ball's way down
                ball, T = self.path[falling[0]], self.times[falling[0]]
        elif h < CONFIG["wave_volley_height_min"] * UU:
            mech = FLIP_UP
        else:
            mech = FULL_AERIAL
        self.mechanic = mech
        boost_ok = self.may_boost(purpose)
        contact = ball - unit(aim - ball) * CONTACT
        self.target_point = contact
        under = flat(contact)
        dist = np.linalg.norm(under - flat(me.pos))
        speed_now = me.vel @ me.forward
        # Arrive when the ball does. Out of reach, that means settling under where it will be at the end of
        # the prediction instead of sprinting past it.
        want_speed = dist / max(T, 0.05)
        if mech == GROUND and self.reachable and self.ball_pos[2] < 2 * BALL_R:
            want_speed = 2300.0   # it's rolling: nothing to time, so get there (pacing to T just stalls us)
        to_contact = me.local(under - flat(me.pos))
        lined_up = abs(math.atan2(to_contact[1], to_contact[0])) < 0.35

        if mech == GROUND:
            to_ball = me.local(ball - me.pos)
            if (self.reachable and T < CONFIG["ground_jump_time"] and speed_now > 500
                    and abs(math.atan2(to_ball[1], to_ball[0])) < 0.5 and dist < speed_now * T + 300):
                # WHY: 0.3 s out and lined up: jump and flip through it; the dodge adds a lot of shot power.
                return self.start(self.flip_hit(boost_ok), "flip into ball")
            fancy = purpose == "hit" and me.boost > CONFIG["boost_moderate_below"]   # clears don't need angles
            target = self.approach(ball, aim) if fancy else ball - unit(flat(aim - ball)) * CONTACT
            pad = self.pad_on_path(target, CONFIG["small_pad_max_delay"]) if T > 1.0 else None
            if pad is not None:
                target, want_speed = pad, 2300.0
            return self.drive_to(target, max(want_speed, 600.0), boost_ok)

        if mech == FLIP_UP:
            lift = contact[2] - me.pos[2]
            double = lift > 220
            jt = jump_time(lift, double)
            if jt is not None:
                if T <= jt + 0.03 and lined_up and abs(dist - speed_now * T) < 250:
                    # WHY: the jump takes as long as the ball needs to drop into reach: go now.
                    return self.start(self.air_routine(mech, aim, double, purpose), mech)
                return self.drive_to(under, want_speed, boost_ok)
        if self.reachable and self.aerial_ready(contact, T):
            # WHY: from here a double jump plus boost carries us to the contact point on time.
            return self.start(self.air_routine(mech, aim, True, purpose), mech)
        return self.drive_to(under, want_speed, boost_ok)

    def aerial_ready(self, target, T):
        """Would a double jump right now, plus boost, carry us to `target` in T seconds?"""
        me = self.me
        if T < 0.35 or me.boost < 20 * T:
            return False
        v = me.vel + me.up * (2 * JUMP_IMPULSE + JUMP_HOLD_ACC * JUMP_HOLD_TIME)
        delta = target - me.pos - v * T - 0.5 * GRAVITY * T * T
        need = 2 * np.linalg.norm(delta) / (T * T)
        facing = unit(flat(delta)) @ unit(flat(me.forward))
        return need < 0.8 * AIR_BOOST_ACC and (facing > 0.8 or np.linalg.norm(flat(delta)) < 300)

    def air_target(self, aim, mech, boost_ok):
        """Contact point, time, and extra acceleration needed for the soonest slice our flight can meet."""
        me, t = self.me, self.times
        shots = aim - self.path
        contacts = self.path - shots / np.maximum(np.linalg.norm(shots, axis=1), 1.0)[:, None] * CONTACT
        drift = me.pos + me.vel * t[:, None] + 0.5 * GRAVITY * t[:, None] ** 2
        need = 2 * np.linalg.norm(contacts - drift, axis=1) / t ** 2
        if mech == VOLLEY:
            need = np.where(self.path_vel[:, 2] < 0, need, np.inf)   # volleys meet the ball on its way down
        budget = 0.85 * AIR_BOOST_ACC if boost_ok and me.boost > 0 else 100.0
        ok = np.nonzero(need < budget)[0]
        i = int(ok[0]) if len(ok) else int(np.argmin(need))
        return contacts[i], t[i], need[i]

    # ------------------------------------------------------------------ mechanics (one yield per tick)
    def start(self, routine, name):
        """Lock in a timed mechanic; it runs every tick until it finishes."""
        self.routine, self.routine_name = routine, name
        return self.step_routine() or SimpleControllerState()

    def step_routine(self):
        try:
            return next(self.routine)
        except StopIteration:
            # WHY: the mechanic is over (landed, hit, or gave up): hand control back to the state machine.
            self.routine = None
            return None

    def flip_hit(self, boost):
        """Hop, then dodge into the ball: the cheapest power shot in the game."""
        start = self.time
        while self.time - start < 0.08:
            yield SimpleControllerState(jump=True, throttle=1.0, boost=boost)
        yield from self.dodge(self.ball_pos - self.me.pos)
        yield from self.recover()

    def dodge(self, direction):
        """Let go of jump for a frame, flip toward `direction`, and hold the stick so the flip isn't cancelled."""
        yield SimpleControllerState(throttle=1.0)
        local = self.me.local(direction)
        scale = max(abs(local[0]), abs(local[1]), 1e-6)
        pitch, yaw = -local[0] / scale, local[1] / scale
        start = self.time
        yield SimpleControllerState(jump=True, pitch=pitch, yaw=yaw, throttle=1.0)
        while self.time - start < 0.5 and not (self.me.on_ground and self.time - start > 0.2):
            yield SimpleControllerState(pitch=pitch, yaw=yaw, throttle=1.0)

    def air_routine(self, mech, aim, double, purpose):
        """Jump, fly to the ball with continuous air control, hit it, keep touching it while we can, land."""
        start, boost_ok = self.time, self.may_boost(purpose)
        flip_until = start + 1.25 if not double else 0.0   # a single jump keeps its dodge for 1.25 s
        try_reset = CONFIG["flip_reset_enabled"] and mech == FULL_AERIAL and purpose == "hit"
        self.air_touches = 0
        while self.time - start < JUMP_HOLD_TIME:      # hold jump for the full extra lift
            c = SimpleControllerState(jump=True, throttle=1.0)
            target, T, _ = self.air_target(aim, mech, boost_ok)
            self.fly(c, target, T, boost_ok, mech)
            yield c
        if double:
            yield SimpleControllerState(throttle=1.0)              # a frame off the button: a fresh press
            yield SimpleControllerState(jump=True, throttle=1.0)   # second jump, stick centred so it's not a dodge
        while self.time - start < 4.0 and not self.me.on_ground:
            me = self.me
            target, T, need = self.air_target(aim, mech, boost_ok)
            to_ball = self.ball_pos - me.pos
            dist = np.linalg.norm(to_ball)
            if self.touched:
                self.air_touches += 1
                if me.local(to_ball)[2] < -BALL_R * 0.5:
                    flip_until = math.inf   # the ball was under our wheels: that touch was a flip reset
            if self.air_touches >= CONFIG["max_air_touches"] or (need > 2.5 * AIR_BOOST_ACC and dist > 500):
                break  # out of taps, or the ball went somewhere we can't follow
            if self.time < flip_until and dist < 260 and me.local(to_ball)[0] > 0.6 * dist:
                yield from self.dodge(to_ball)   # flip into it; after a reset this is the reset's payoff
                flip_until = 0.0
                continue
            c = SimpleControllerState(throttle=1.0)
            if try_reset and self.time >= flip_until and dist < 500 and self.ball_pos[2] > 500 and T < 0.4:
                # Flip-reset attempt: roof away from the ball so all four wheels land on it, nose toward the aim.
                up = unit(me.pos - self.ball_pos)
                shot = unit(aim - self.ball_pos)
                self.orient(c, unit(shot - up * (shot @ up)), up)
            else:
                self.fly(c, target, T, boost_ok, mech)
            yield c
        yield from self.recover()

    def fly(self, c, target, T, boost_ok, mech):
        """Point the nose where the missing acceleration must come from; boost when lined up and needed.
        Early on that's below the ball (stay under it); once on course it's straight at the contact point."""
        me = self.me
        T = max(T, 1 / 60)
        delta = target - me.pos - me.vel * T - 0.5 * GRAVITY * T * T
        need = 2 * np.linalg.norm(delta) / (T * T)
        heading = unit(delta) if need > 100 else unit(target - me.pos)
        if mech == VOLLEY:  # one tilt: wings level, nose no more than 45° up
            pitch = clamp(math.asin(clamp(heading[2], -1.0, 1.0)), -0.3, math.radians(45))
            heading = unit(flat(heading)) * math.cos(pitch) + WORLD_UP * math.sin(pitch)
        self.orient(c, heading, WORLD_UP)
        c.boost = boost_ok and me.boost > 0 and need > 150 and me.forward @ heading > 0.85

    def orient(self, c, forward, up):
        """Continuous PD air control: nose toward `forward`, roof toward `up`."""
        me = self.me
        f, u, w = me.local(forward), me.local(up), me.local(me.angvel)
        c.pitch = pd(math.atan2(f[2], f[0]), w[1] / 4)
        c.yaw = pd(math.atan2(f[1], f[0]), -w[2] / 4)
        c.roll = pd(math.atan2(u[1], u[2]), w[0] / 2)

    def recover(self):
        """Land on our wheels with the nose along our direction of travel."""
        start = self.time
        while not self.me.on_ground and self.time - start < 3.0:
            c = SimpleControllerState(throttle=1.0)
            travel = flat(self.me.vel) if np.linalg.norm(flat(self.me.vel)) > 200 else flat(self.me.forward)
            self.orient(c, unit(travel), WORLD_UP)
            yield c

    def recover_if_airborne(self):
        me = self.me
        if me.on_ground:
            return None
        if me.pos[2] < 60 and me.up[2] < -0.5:
            # WHY: stuck on our roof: tapping jump rolls the car back onto its wheels.
            return SimpleControllerState(jump=int(self.time * 8) % 2 == 0)
        # WHY: airborne with no mechanic running (a bump, falling off a wall): land well before deciding anything.
        return self.start(self.recover(), "recover")

    # ------------------------------------------------------------------ driving
    def drive_to(self, target, speed, boost_ok, reverse=False):
        """Steer at `target` and hold `speed` (uu/s) using throttle, boost and powerslide."""
        me = self.me
        self.target_point = target
        c = SimpleControllerState()
        local = me.local(target - me.pos)
        forward_speed = me.vel @ me.forward
        if reverse:
            c.steer = clamp(3.0 * math.atan2(local[1], -local[0]), -1.0, 1.0)  # in reverse, steer right = tail right
            c.throttle = -1.0 if -forward_speed < speed else 0.0
            return c
        angle = math.atan2(local[1], local[0])
        c.steer = clamp(3.0 * angle, -1.0, 1.0)
        c.handbrake = abs(angle) > 1.8 and forward_speed > 600
        # Don't arrive faster than we can turn onto the target: at 2300 uu/s the turning circle is ~23 m across.
        speed = min(speed, max(turn_speed(2 * abs(local[1]) / max(local[0] ** 2 + local[1] ** 2, 1.0)), 500.0))
        c.throttle = clamp((speed - forward_speed) / 250.0, -1.0, 1.0)
        c.boost = (boost_ok and speed - forward_speed > 150 and abs(angle) < 0.3
                   and forward_speed < 2200 and me.on_ground)
        return c

    def hold(self, spot):
        """Get to `spot` and sit there facing the ball. The one place we don't run at full throttle:
        a keeper doing laps in his own net is the Bronze move."""
        me = self.me
        local = me.local(spot - me.pos)
        dist = np.linalg.norm(flat(spot - me.pos))
        urgent = self.on_target
        if dist > 400:
            if abs(math.atan2(local[1], local[0])) > 2.4 and dist < 1500:   # nearly straight behind: back up
                return self.drive_to(spot, 1400.0 if urgent else 1000.0, False, reverse=True)
            return self.drive_to(spot, min(2300.0, 1.5 * dist + 300), self.may_boost("save" if urgent else "position"))
        self.target_point = spot
        face = me.local(self.ball_pos - me.pos)
        angle = math.atan2(face[1], face[0])
        throttle = clamp(local[0] / 150.0, -1.0, 1.0)
        if abs(angle) > 0.4 and abs(throttle) < 0.4:   # creep so the wheels can turn us toward the ball
            throttle = math.copysign(0.4, local[0] or 1.0)
        return SimpleControllerState(throttle=throttle, steer=clamp(3.0 * angle, -1.0, 1.0) * math.copysign(1.0, throttle))

    # ------------------------------------------------------------------ debug
    def render(self):
        if not CONFIG["debug_render"]:
            return
        r = self.renderer
        r.begin_rendering()
        mech = self.routine_name if self.routine else self.mechanic
        text = f"{self.name}: {self.state} | {mech or '-'} | boost {self.me.boost} | taps {self.air_touches}"
        r.draw_string_2d(20, 20 + 30 * self.index, 1, 1, text, r.white())
        r.draw_line_3d(self.me.pos, self.target_point, r.cyan())
        r.draw_line_3d(self.ball_pos, self.path[-1], r.yellow())
        r.end_rendering()
