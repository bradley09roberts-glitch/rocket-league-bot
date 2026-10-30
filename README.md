# FourCans: an autonomous Rocket League bot (RLBot v4)

FourCans plays the whole match by itself: kickoffs, shots, saves, rotation and boost management. You
spectate. It is pure heuristics and ball physics, with no ML.

| File | What it is |
|---|---|
| `main.py` | `CONFIG` and the full `Bot` class: state machine, mechanics, driving, air control |
| `rl_physics.py` | Arena constants, the ball predictor, the car model and travel-time estimates |
| `bot.cfg`, `appearance.cfg` | What RLBot loads (name, Python file, Octane loadout) |
| `rlbot.cfg`, `run.py` | Optional manual launcher: a 1v1 against Psyonix's All-Star bot |
| `requirements.txt` | `rlbot>=1.30,<2` and `numpy` |

## Setup

1. **Rocket League.** Install it through Epic Games (or Steam, if you own it there) on Windows, launch it
   once and close it. RLBot starts the game itself. There is no "UT4 client" in this setup: the only
   game client RLBot needs is Rocket League.
2. **Python 3.10.** Install it from python.org and tick "Add python.exe to PATH".
3. **RLBot.** Pick one:
   - **RLBotGUI** (easiest): install it from [rlbot.org](https://rlbot.org), then click **Add** → **Load
     folder** and choose this `rocket-league-bot` folder. RLBotGUI installs `requirements.txt` for you.
   - **Manual**: in this folder, run `py -3.10 -m venv venv`, `venv\Scripts\activate`, then
     `pip install -r requirements.txt`.

   This is RLBot **v4** (the `rlbot` 1.x package). RLBot v5 has a different API and will not load
   this bot. The `<2` pin in `requirements.txt` keeps pip off the v5 betas.
4. **Launch a 1v1.**
   - **RLBotGUI**: put **FourCans** on Blue and **Psyonix All-Star** (or a second FourCans) on Orange,
     set the mode to Soccer, then click **Start Match**.
   - **Manual**: run `python run.py` from this folder. It uses `rlbot.cfg`: FourCans on Blue against
     an All-Star on Orange, on DFH Stadium. Set `participant_type_1 = rlbot` for a mirror match.
5. **Watch.** With `debug_render` on, each bot prints `state | mechanic | boost | taps`. A cyan line
   shows where it is driving, and a yellow line shows where the predicted ball ends up.

For 3v3, add six participants in RLBotGUI. The bot reads teammates from the packet, so no config change
is needed.

## How it decides

The state is re-chosen each tick, with priority from top to bottom. While a jump, dodge or aerial is
running, the state is frozen.

| State | When | What |
|---|---|---|
| `KICKOFF` | Ball is sitting still on the centre spot | The closest car (left goes on a tie) boosts in at a slight side angle and flips into the ball |
| `SAVE` | Shot on target, or the ball is in our third and heading at our goal | Goal-side and first to the ball: clear it. Otherwise: hold the goal line |
| `ATTACK` | Ball in their third and we're our team's first man, or a midfield ball we win | Shoot (45° approach, flip-hit, flip-up, volley or aerial, by ball height) |
| `AERIAL` | Ball > 3 m, in our half, and we're first to it | The same hit logic, with an air mechanic |
| `ROTATE` | Anyone else gets there first | Keeper in our third, goal-side and off-ball in midfield, opposite-side 15 m back in their half |

Mechanics by intercept height: under 2 m is a ground hit (flip into the ball 0.3 s out). From 2 to 6 m
and falling, it's a "flip-up" jump shot (single or double jump, timed from a jump-height table). From 3
to 7 m with the car below the ball, it's a wave volley (nose at most 45° up, wings level). Above that,
it's an aerial. Aerials keep chasing after a touch for up to `max_air_touches`. On high attacking
aerials, the bot turns its wheels toward the ball to try for a flip reset, then spends the flip on the
next touch.

## Where I interpreted the spec

- **`keeper_clamp_x` = 3.8 m.** The real posts are at ±8.93 m. The 3.8 m clamp is used for
  shadowing, as specified. When a shot is actually on target, the keeper stands where the shot will
  cross the line (inside the posts). Otherwise a 3.8 m clamp leaves half the goal open, which is how
  the first save test was conceded.
- **"Jump at 0.3 m distance."** 30 cm is already touching the ball, so this is `ground_jump_time`
  = 0.3 s before contact.
- **The "flip reset" row** (drive under the ball, flip up, hit at the apex) describes a jump shot. It is
  implemented as described, under the name `flip-up`. Actual wheels-first flip resets are attempted
  inside aerials.
- **"Bump clear to the opposite corner"** goes to the far-end corner on the ball's own side, never
  across our own box.
- **`ball.reset_time`** doesn't exist in the packet. A kickoff is detected as "ball dead still at
  centre".
- **"Always full throttle"** holds for rotation (at least 1000 uu/s, so the bot circles its spot). The
  keeper is the exception: it stops and faces the ball.
- **The ball predictor** is our own, as specified: gravity, drag, and bounces off the floor, ceiling,
  side walls, back walls (with goal mouths) and 45° corners. RLBot's built-in
  `get_ball_prediction_struct()` uses the real arena mesh, including curved wall joins, and is more
  accurate. Swapping it in is the first upgrade to try.

## What was and wasn't tested

There's no Rocket League in the environment this was written in, so **it has not been played in the
real game**. Here is what was checked:

- It loads through RLBot 1.68's own code: `bot.cfg` parsing, the class importer, the `rlbot.cfg` match
  parser, and the ctypes and flatbuffer controller conversion. This used Python 3.10.
- 20,000 random packets (demolished cars, upside-down cars, ball in the net, replays, 6 cars) produced
  no exceptions, and every stick value stayed in range.
- In a rough stand-in physics sim (simplified driving, jumps, air rotation with Rocket League's
  torque signs, sphere collisions):
  - Kickoffs connect from all five spots.
  - Straight-on shots score. Wide-angle and corner balls connect (the corner ball gets centred) but
    didn't score.
  - 30 of 30 random on-target shots were saved.
  - 25 of 25 pop-up aerials got an air touch, and 3 of 25 got a second touch.
  - Wheels-on-ball reset contact happened in 3 of 25, but the follow-up flip never connected. The bot
    arrives too fast, so the reset touch knocks the ball away. Expect flip resets to be the weakest
    mechanic until approach speed is tuned in-game.
  - Ticks average about 1.2 ms per bot in 1v1 and 1.6 ms in 3v3. The budget is 8.3 ms.

Treat the skill targets (SSL in 1v1, Diamond in 3v3) as unmeasured. Hand-tuned heuristic bots rarely
get near SSL, and this one hasn't faced a ranked player yet. The knobs most likely to need in-game
tuning are `ground_jump_time`, `CONTACT` in `rl_physics.py`, the `0.8 * AIR_BOOST_ACC` aerial threshold
in `aerial_ready`, and `prediction_time`. 1.5 s is too short to see where a sky-high ball lands, so try
3.0.
