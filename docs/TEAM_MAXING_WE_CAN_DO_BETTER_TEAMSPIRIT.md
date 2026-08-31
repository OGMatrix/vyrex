EastvillageRole icon, Twitch Mod
OP
 — 17/02/2026 15:09
I was recently asked how I got my bots to do rotations so early in learning. I think it is because, I have been trying out a different kind of team spirit formula.

TL;DR: Use max instead of mean in team spirit.
EastvillageRole icon, Twitch Mod
OP
 — 17/02/2026 15:10
TeamSpirit usually refer to a reward function wrapper, where we reward bots if their team mates does something good. The goal is to prevent stuff like double commits, if all team mates are rewarded for a bot committing to a shot, then they will eventually figure out that it is better if only one of them commits.

The standard implementation (excuse my notation) is usually a linear interpolation between the individual's reward and the team's mean reward:

reward[i] = childReward[i] * (1 - alpha) + mean(childReward[team]) * alpha


and when combined with zero-sum:

reward[i] = childReward[i] * (1 - alpha) + mean(childReward[team]) * alpha - mean(childReward[1 - team]) * beta


where alpha and beta are hyperparameters for the interpolation and the "zero-suming", respectively.

Using this formula definitely helps the bots learn 2s and 3s, but I do not think it is ideal in many scenarios. The problem is two-fold. Firstly, the team is still better off double-committing in the short term; If multiple team mates touch the ball, the final individual reward is higher, than if only one bot did it. This means the team spirit is not actually solving the problem it was intended to solve. Secondly, since the rewards are effectively distributed, the individual rewards signal are reduced. If a bot makes a good touch in 3s, it is only rewarded a third of what that shot would have been in 1s (assuming equal weights). To the bot, this means that touches are less important in 3s. You can increase the weights in 3s to counter this, but then issue one is just more prominent. While I have used touching and double-commiting in this example, the issues apply to most rewards that we team-spirit(/zero-sum). VPB, rotations, defence, holding on to boost, etc.
So what's the solutiton? Instead of taking the mean, we take the max (or min for penalties):

reward[i] = childReward[i] * (1 - alpha) + max(childReward[team]) * alpha - max(childReward[1 - team]) * beta


Now the individual bot gets the full reward, if any of the team mates are able to activate the reward signal - which is really want we want in many cases - and both the issues from before disapear. A single commit and a double commit rewards every team member the same amount, so the second bot might as well do something else, and we end up with the bots doing different things: committing, defending, collecting boost, positioning for a pass. Moreover, the reward signals are not dulled in 2s or 3s, and we can in many cases use the same weights in 1s and 3s. Not in all cases though. For instance, there are more demos and aerials in 3s than in 1s, so you probably still need to adjust for that.

In ⁠Wisp, I use averaging when I want all team members go for the reward simultaneous (e.g. speed, holding boost, or energy), and I use the max when I just want one team member to go for the reward (e.g. vpb, touches, or defensive position). I am really happy with the results.
Thank you for coming to my TED talk. 
RolvRole icon, ML Overlord — 17/02/2026 15:12
Actually pretty easy to do with tools, just make a DistributeRewardsWrapper with agg_method=max
Cryy_Stall — 17/02/2026 15:12
from TeamMeaning™ to TeamMaxing™
RolvRole icon, ML Overlord — 17/02/2026 15:18
there are also use-cases for agg_method=sum for that matter
EastvillageRole icon, Twitch Mod
OP
 — 17/02/2026 15:25
sum is the same as mean, but with a weight scaled by team size.
RolvRole icon, ML Overlord — 17/02/2026 15:25
yes
EastvillageRole icon, Twitch Mod
OP
 — 17/02/2026 15:26
mul is an interesting aggretion method. It works like a conditional (assuming the range is 0..1). They are only rewarded if everyone does it. That might be useful too, but maybe those conditions should just go in the reward logic instead. 
JPK314Role icon, Maestro — 17/02/2026 15:38
This does zero summing weirdly. It only actually does zero sum if selfishness and selflessness sums to 1 and the agg method is average and team coef = opp coef
Distribute rewards is supposed to be for zero summing as well right
Shouldn't you always subtract the average of the spirit rewards for the opposing team
Maybe modified by some weight if you don't want to do full zero summing
RolvRole icon, ML Overlord — 17/02/2026 15:40
They defaults are such that this is the case
I just tried to make it a bit more configurable
JPK314Role icon, Maestro — 17/02/2026 15:41
Sure but the implementation makes zero summing impossible if you use other values which you might want to do for other reasons
Also why is team coef even a thing
RolvRole icon, ML Overlord — 17/02/2026 15:42
What?
JPK314Role icon, Maestro — 17/02/2026 15:42
Let's say you want to use max as the agg function but you still want to zero sum
The wrapper you have makes that impossible
It won't be zero sum
RolvRole icon, ML Overlord — 17/02/2026 15:43
Ehhh
zero max
it's fine people can figure it out
JPK314Role icon, Maestro — 17/02/2026 15:43
What the fuck are you talking about
Zero max
RolvRole icon, ML Overlord — 17/02/2026 15:44
idk
JPK314Role icon, Maestro — 17/02/2026 15:45
It's not even true because the spirit reward adds the max over the team to the individuals reward
RolvRole icon, ML Overlord — 17/02/2026 15:46
I can't cover every possibility
if people want max and zero sum they can wrap the wrapper
RolvRole icon, ML Overlord — 17/02/2026 15:46
.
JPK314Role icon, Maestro — 17/02/2026 15:47
It just doesn't make sense man
RolvRole icon, ML Overlord — 17/02/2026 15:47
ok
JPK314Role icon, Maestro — 17/02/2026 15:47
Why are you even interpolating between the spirit reward and the opponent aggregate
That would make sense if you were forcing selfishness and selflessness to sum to 1
JPK314Role icon, Maestro — 17/02/2026 15:55
The dimensionality! It's collapsing!
EastvillageRole icon, Twitch Mod
OP
 — 17/02/2026 16:10
I'd recommend just making a new ZeroSumTeamMaxing reward. No need to generalize further
JeffA233 — 17/02/2026 17:52
this is very interesting and I will take notes
you'd have to try it to make sure the critic can deal with it because it likes to be temperamental
SubparNova — 17/02/2026 20:43
Have you found that the bots have a hard time learning some skills in 2v2 and 3v3 this way? I’ve seen recommendations here before to intentionally set team spirit to be less than 1 because they want the individual reward signal to come through slightly stronger than the rest of the team’s reward that aren’t performing the action giving the reward
I assume skills could still be learned but maybe it just takes a bit longer with this method for some of them? The tradeoff is likely worth it but I’m curious what you’ve seen in your experience with it 
EastvillageRole icon, Twitch Mod
OP
 — 17/02/2026 20:55
I think those observation are cases of the second issue I mention in the first message. They learn the skills slower because two thirds of the reward signal is given to the team mates. In my experience, every time I switched a team spirit to team maxing, the bot learned much faster.
But you can still use a factor less than zero with team maxing if you want it make the reward more individual 
JeffA233 — 17/02/2026 21:18
be the first to train a good bot with it and see if it starts a wave of change lol
my only concern would be that it means you don't penalize bad behavior that the other agents are doing
maybe that's okay in the end because eventually it is ruled out due to positive reinforcement
but like if the 3rd agent on a 3s team is on defense rolling around doing nothing, with team spirit you would get a boost from it grabbing boost (given a boost reward), whereas with maxing, boost gathering isn't a thing if the goal rewards are big enough eg. vbg 
in theory it seems like eventually both would line up to the same actions being reinforced (at least in my imagination) but they do it differently 
maybe one other issue is maxing sounds like it has a lot of fun to be had with balancing rewards
EastvillageRole icon, Twitch Mod
OP
 — 17/02/2026 21:25
Why would boost gathering not be a thing in that case? I am not maxing over the reward functions, I am maxing over the team mates. So if one team mate increases vbg and another picks up boost, then that's better overall for both bots. 
JeffA233 — 17/02/2026 21:26
still kinda feels like it has the same issue to me
feels like there's possible information loss of sorts
I should be clear, you literally just have to test this kind of thing, I'm just giving you a guess as to what I would expect are possible issues lol
they may or may not be actually issues
EastvillageRole icon, Twitch Mod
OP
 — 17/02/2026 21:30
Yes, tbf I do see the third player flop around in goal sometimes. I think that happens because the two other bots are doing most rewards so well that the third player cannot contribute in any way except for staying near goal. More reward signals for what third man should be doing would likely fix that.
I'll let you know what I have learned when I beat Opti
FlamingFury00 — 17/02/2026 23:44
If you want every agent to do something, just punish it to literally just exist like I did some time ago. It will figure out something to do in the meanwhile (or farm some other reward).
Cryy_Stall — 17/02/2026 23:49
Existential crisis reward
JeffA233 — 17/02/2026 23:56
lmao, true
JPK314Role icon, Maestro — 17/02/2026 23:59
This is how you encourage own goals
JeffA233 — 18/02/2026 00:02
personally I only do a penalty like that with regards to boost
anything else seems like potentially negative behavior
if it sits at zero (or well in my case a threshold) on defense then I give it a penalty over time 
JPK314Role icon, Maestro — 18/02/2026 00:03
That makes sense because as a player it is physically painful to not have boost
JeffA233 — 18/02/2026 00:03
:pepeKekw: well it pains me as the creator too
so much pain
FlamingFury00 — 18/02/2026 00:07
Exactly how I called it, "ExistentialPainReward" :information:
JeffA233 — 18/02/2026 00:10
lmao