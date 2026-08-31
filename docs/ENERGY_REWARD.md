
RolvRole icon, ML Overlord
OP
 — 14/05/2025 10:47
def jump_potential_energy(car: Car):
    local_vel = np.dot(car.physics.rotation_mtx.T, car.physics.linear_velocity)
    speed = np.linalg.norm(local_vel)
    if speed >= 1700:
        dodge_dv = CAR_MAX_SPEED - speed
    else:
        f = speed / 1700
        dodge_dv = 500 * (1 - f) + 600 * f
    double_jump_dv = 291.667
    jump_dv = 550.0

    dv1 = np.array([0, 0])
    dv2 = np.array([0, 0])
    if car.on_ground:
        dv1[1] += jump_dv
        dv2[1] += jump_dv
    if car.has_flip:
        dv1[0] += dodge_dv
        dv2[1] += double_jump_dv

    lv = np.array([
        math.sqrt(local_vel[0] ** 2 + local_vel[1] ** 2),
        local_vel[2]
    ])

    new_vel1 = local_vel + dv1
    new_vel2 = local_vel + dv2
    dv = max(np.linalg.norm(new_vel1),
             np.linalg.norm(new_vel2))
    dv = min(CAR_MAX_SPEED, dv) - speed

    return dv
Waddles — 14/05/2025 10:47
Well that certainly looks thorough...
RolvRole icon, ML Overlord
OP
 — 14/05/2025 10:48
just noticed it returns dv instead of the actual energy hehe
Waddles — 14/05/2025 10:49
Why is the linear vel dotted into the rotation matrix?
RolvRole icon, ML Overlord
OP
 — 14/05/2025 10:50
i forget
lv also not used
I probably just left it unfinished when it got too complicated
Waddles — 14/05/2025 10:55
Well, if anyone's interested in my hacky energy reward mod, I'll be trying it.  Maybe it's terrible
from typing import Any, Dict, List

import numpy as np
from rlgym.api import RewardFunction, AgentID, StateType, RewardType
from rlgym.rocket_league.api import GameState

mod_energy_reward.py
3 KB
RolvRole icon, ML Overlord
OP
 — 14/05/2025 10:55
not car.has_jumped?
Waddles — 14/05/2025 10:56
I thought has_jump didn't exist? 
RolvRole icon, ML Overlord
OP
 — 14/05/2025 10:56
I'm just not sure why you're using it 
car.on_groundis the check surely?
Waddles — 14/05/2025 10:57
Well, I'm adapting from Kaiyo, v2 didn't seem to have has_jump, but you can also have jump in the air, right?  That's the second double jump?
So I was presuming has_jumped is just the negative of has_jump as I saw that somewhere else
RolvRole icon, ML Overlord
OP
 — 14/05/2025 10:58
also I don't think the boost potential scales linearly?
and it depends on current velocity
RolvRole icon, ML Overlord
OP
 — 14/05/2025 10:59
.
Waddles — 14/05/2025 11:01
Yeah, I'm definitely fudging.  Basically, if there wasn't a max_velocity, a full tank of boost could net you 3000uu/s, so I'm just dividing that up equally.
every time you double velocity, KE quadruples, so a single boost would be worth 1/4th of two boost, etc
RolvRole icon, ML Overlord
OP
 — 14/05/2025 11:04
yep
idk as mentioned earlier in this post I feel like energy might be the wrong thing to measure
encouraging all this stuff is fine but it really seems like it ends up with no basis in reality
it's just a weighting scheme
Waddles — 14/05/2025 11:06
Yeah, fair enough
As it is, height is linearly rewarded, speed is very non linearly rewarded, boost is linearly rewarded, etc.  It doesn't seem too crazy
I was just unhappy by how little impact boost and jump were having, which I suspect might have just be a unit conversion thing
RolvRole icon, ML Overlord
OP
 — 14/05/2025 11:08
right but boost and flips technically should be non linearly rewarded
they'd scale with n^2
but like, I'm using sqrt(n) for boost, so that's very counterintuitive
Waddles — 14/05/2025 11:08
Flips at least are non-linear in the same fashion
I have no idea what is up with dodge_impulse, just assuming it works, and what comes out in the logs seems reasonable
Eh, since I'm fudging things, maybe I'll decrease the boost potential back down a ways, it is sorta swamping now, lol
Waddles — 14/05/2025 11:16
(.5*CAR_MASS*((CAR_MAX_SPEED)**2))  ends up with it being closer to 2x height potential instead of 3.3.  : hands wave vigorously :
RolvRole icon, ML Overlord
OP
 — 14/05/2025 12:13
fwiw emprically the formula for boost value is more like (1-10^-x)/0.9 (made by running next goal predictor with varying boost amounts)
but sqrt(x) is fairly close
(boost amount normalized to 0-1 range)
RolvRole icon, ML Overlord
OP
 — 14/05/2025 15:05
Image
⁠ml-lounge⁠
RichardsWorld IS BACK — 17/07/2025 02:46
Is this reward good? I'm making my own, but I wonder I think I'd good enough already
Waddles — 17/07/2025 02:50
I am not a good source of information on what rewards work, unfortunately
EastvillageRole icon, Twitch Mod — 22/10/2025 20:32
Nice observation Rolv!
Looks like 1-(1-x)^2 is a slightly better approximation. And maybe slightly cheaper than sqrt idk performance 🤷‍♂️
Image
I am trying to figure out why my bot likes to spend all the boost immediately. Maybe this change will do the trick 
RolvRole icon, ML Overlord
OP
 — 22/10/2025 20:45
any of those functions probably do the trick
the main thing I noticed was that rewarding for having boost is more effective than rewarding change in boost
RolvRole icon, ML Overlord
OP
 — 22/10/2025 21:23
here's a new plot with a new NGP
Image
RolvRole icon, ML Overlord
OP
 — 22/10/2025 21:34
Image
EastvillageRole icon, Twitch Mod — 22/10/2025 21:35
A lot more linear than last
RolvRole icon, ML Overlord
OP
 — 22/10/2025 21:37
about x^(3/4)
EastvillageRole icon, Twitch Mod — 22/10/2025 21:38
(1+x-(1-x)^2)/2 ?
RolvRole icon, ML Overlord
OP
 — 22/10/2025 21:38
also pretty close
x^(3/4) overvalues low boost and yours overvalues high boost
Image
EastvillageRole icon, Twitch Mod — 22/10/2025 21:42
But this is NGP. Does this curve translate to boost energy value?
RolvRole icon, ML Overlord
OP
 — 22/10/2025 21:42
it does not
it just tells you the relative quality (next goal probability) of having x boost
RolvRole icon, ML Overlord
OP
 — 22/10/2025 21:50
this is 3v3 btw, let me see in 2s and 1s
RolvRole icon, ML Overlord
OP
 — 22/10/2025 22:01
here's the relative value of going to x boost in a state where you had y boost (in 3v3)
Image
Here's 1v1 (much less linear)
Image
RolvRole icon, ML Overlord
OP
 — 22/10/2025 22:14
and 2v2 (somewhere inbetween)
Image
Waddles — 23/10/2025 03:54
Surely with a ngp that considers the whole state it actually depends on the state of play?
RolvRole icon, ML Overlord
OP
 — 23/10/2025 08:06
Yes
This is the average
