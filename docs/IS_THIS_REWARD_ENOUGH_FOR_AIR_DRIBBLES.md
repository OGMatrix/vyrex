justin
OP
 — 16/02/2026 15:52
pretty self explanatory here’s the link https://github.com/RLGym/rlgym-tools/blob/main/rlgym_tools/rocket_league/reward_functions/aerial_distance_reward.py i wanna know if i need the air reward or jump touch reward with it or this in enough
GitHub
rlgym-tools/rlgym_tools/rocket_league/reward_functions/aerial_dista...
Extra tools for RLGym, like sb3 compatibility. Contribute to RLGym/rlgym-tools development by creating an account on GitHub.
Extra tools for RLGym, like sb3 compatibility. Contribute to RLGym/rlgym-tools development by creating an account on GitHub.
justin
OP
 — 16/02/2026 15:53
also it’s at 2b steps and it can play pretty well it’s starting to learn to flick the ball should i wait or can i put the rewards now with no issue
Martico2432

Role icon, Verified — 16/02/2026 16:16
it will teach it to aerial
justin
OP
 — 16/02/2026 16:19
yes but is it enough or should i also have air reward or jump touch reward
and is it enough for air dribbles too?
Martico2432

Role icon, Verified — 16/02/2026 16:19
Image
Martico2432

Role icon, Verified — 16/02/2026 16:20
tell me what you think
justin
OP
 — 16/02/2026 16:20
i do think it’s enough but my thinking isn’t enough because i don’t know shit
it rewards for how much time and how much touches and what height is was made so i think so yes
justin
OP
 — 16/02/2026 16:36
but i need approval i don’t wanna fuck it up
Martico2432

Role icon, Verified — 16/02/2026 16:38
Q: should i also have air reward?
A: I mean, it has to jump

Q: should i also have  jump touch reward?
A: Clearly states: First aerial touch is rewarded by height, so no

Q:  is it enough for air dribbles too?
A: Rewards for multiple touches in the air, and moving the ball towards any direction, so probably
justin
OP
 — 16/02/2026 16:48
thank you
would you also happen to know approximately what weight i should put or how to find out myself?
Martico2432

Role icon, Verified — 16/02/2026 16:52
justin
OP
 — 16/02/2026 16:54
would you be able to tell me if i should put it higher or lower than advanced touch reward
Martico2432

Role icon, Verified — 16/02/2026 16:54
i have no idea
justin
OP
 — 16/02/2026 16:55
thank you
Mahkmoud — 16/02/2026 16:56
jsut fuck around and find out
justin
OP
 — 16/02/2026 16:58
i will
SubparNova — 16/02/2026 19:07
Very high. Bots struggle to find aerial plays so you’ve gotta really crank it
Basically with all reward weights: start with a guess then let it run. Then watch your bot and adjust rewards. Just rinse and repeat over and over
Rou_Ad — 16/02/2026 19:21
nah
I tried that and the bot started abusing it like crazy
u need to have just the right weights
justin
OP
 — 16/02/2026 19:22
i think i’m going to start with a weigh of 10 and let it train for like 8h to see  i’ll adjust it after
Rou_Ad — 16/02/2026 19:25
ok
dont forget backupos
justin
OP
 — 17/02/2026 13:09
they started abusing it and dribbling
justin
OP
 — 17/02/2026 13:20
they both just dribble the same ball
HappyBavarian07

 — 17/02/2026 14:14
could help to make it zero sum after they learned to dribble
justin
OP
 — 17/02/2026 15:02
what’s 0 sum
put 0 for the reward? 
what does it do
HappyBavarian07

 — 17/02/2026 15:10
it makes it so for that reward if one person gets it the other bots on the other team get an equal or scaled penalty basically
means the total sum of that reward will be zero across teams or in that certain scale
justin
OP
 — 17/02/2026 15:15
but he’s playing 1v1 rn so he’s going to get nothing from it when they both dribble it?
and how do i do it
how will it help him learn air dribbles
Mahkmoud — 17/02/2026 15:44
player_reward = (self_reward * (1 - team_spirit)) + (avg_team_reward * team_spirit) - average_opp_reward
itll make it so that when they see an opponent air dribble they ll try to stop them which usually means 50 in the air
justin
OP
 — 17/02/2026 15:45
thank you that’s really a good tip
but will it fix them both dribbling to abuse it or should i just lower the reward
or maybe lower à part of the reward like the touch for example
Mahkmoud — 17/02/2026 15:47
fucka round and find out
Rou_Ad — 17/02/2026 15:48
justin
OP
 — 17/02/2026 15:49
why r y’all telling me to 0 sun if y’all don’t know if it’ll work you don’t know?
Rou_Ad — 17/02/2026 15:53
Zero-sum means that, for example, if an agent on the blue team receives the reward, the agent on the orange team has that reward deducted.
justin
OP
 — 17/02/2026 15:54
i know that’s really great thank you for the tip really helpful but i need tips for my dribbling problem
Mahkmoud — 17/02/2026 15:54
a good example would be velocitytowardsgoal you want to punish the bots for letting the velocity towards their goal increase
Rou_Ad — 17/02/2026 15:55
just try some stuff
Mahkmoud — 17/02/2026 15:55
yurr
justin
OP
 — 17/02/2026 15:55
😔
thanks for the help
Rou_Ad — 17/02/2026 15:56
lol
Mahkmoud — 17/02/2026 15:56
everyone who knows what theyre doing are assholes from my experience so yeah you gotta make your own path
I belive in you
not everyone but most
Rou_Ad — 17/02/2026 15:57
Im not knowing what Im doing and I'm still an asshole
the assholes are everywhere
Mahkmoud — 17/02/2026 15:57
exactly if we want to mae good bots we gotta be
justin
OP
 — 17/02/2026 15:57
huh
Mahkmoud — 17/02/2026 15:57
justin try to be more of an asshole and youll find success
justin
OP
 — 17/02/2026 15:58
fuck you
bitch
Mahkmoud — 17/02/2026 15:58
there we go, that all you got bitch
justin
OP
 — 17/02/2026 15:58
yessir
now imma beat slatter
Mahkmoud — 17/02/2026 15:58
yur
justin
OP
 — 17/02/2026 22:36
the 0 sum doesnt work i have an error
NotImplementedError
wait imma make a new question