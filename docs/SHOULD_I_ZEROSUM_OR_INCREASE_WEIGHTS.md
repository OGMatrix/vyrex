should i zerosum saveboost or increase weights ?
🌊 Foxe 🔥 🇨🇦

OP
 — 24/12/2024 06:09
i have a 3B bot, and it so wasteful in boosting, i entered a saveboost reward at 0.1 of weights at 1B and in 2B i increased to 0.5 but without succes, so could zerosumming saveboost could actually help to train faster ?

also my bot at 3B suck so i will revert to 1B (which my bot are trained in step by step so i could easiliy revert if there any problem) and doing some difference choice, which one of them is too increase my saveboostreward to avoid to make the same error
Crazyperson274

 — 24/12/2024 07:34
I would for sure zersum and probably increase.
Fake zealan — 24/12/2024 07:43
Zero sum wouldn't make it less wasteful with boost but it might encourage boost stealing
🌊 Foxe 🔥 🇨🇦

OP
 — 24/12/2024 07:58
tbh
Fredrik — 24/12/2024 11:42
Zerosumming is the best even zerosumming from the start is fine
Cryy_Stall — 24/12/2024 12:04
Fredrik, i hope your parenting methods won't be like your bot training methods :pepeKekw: 

Putting zerosum from the start makes the reward weird because it changes based on a signal the bot doesnt have, sounds like spending computing power
Fredrik — 24/12/2024 14:01
Wdym a signal the bot dosent have lol, it has the reward and and boost amount not hard to put two strings toghether:pepeKekw:  trust the process
Cryy_Stall — 24/12/2024 14:15
Well the reward is fluctuating based on a way more difficult process, it's not like you explicitly tell the bot to lower the opponent's reward
Yes, once understood and discovered, zero sum is far superior
But to use it from the start, you're rough on this poor guy :pepeKekw:
Fredrik — 24/12/2024 15:04
Umh i zerosum almost all my rewards from the start:pepeKekw:  while still improving at insane necto in 1b speeds:pepeKekw:
Just makes my life easier not having to micromanage them
Cryy_Stall — 24/12/2024 15:06
idk what's in that sauce you made, but that's illegal :pepeKekw:
Fredrik — 24/12/2024 15:06
:pepeKekw:
Tech Studio

 — 24/12/2024 19:31
So I'm trying to implement a zerosum reward for saving boost as well (this is my first time trying them out), and I have imported ZeroSumReward as well as CombinedReward. However, my implementation of zerosum reward might be a little screwed up. I'm trying to use it with zipped combined rewards as well so I have something that looks like this 
Save_Boost_Reward = ZeroSumReward(
        child_reward = SaveBoostReward(),
        team_spirit = 0.0,
        opp_scale = 1.0
        )

reward_fn = CombinedReward.from_zipped(
        (Save_Boost_Reward(), 0.8)
)

However, that throws an error saying that "Save_Boost_Reward()" is not callable. Any ideas on what I'm doing wrong? I feel like its something super simple that I'm missing @Fredrik @Cryy_Stall 
Cryy_Stall — 24/12/2024 20:41
Don't call it
You created the object above, just use it below
Tech Studio

 — 24/12/2024 20:44
I thought thats what I was doing, by using it in the reward_fn or am I understanding that wrong?
I just wanted to make that one thing a zero sum so I wasn't sure how to incorporate it while keeping everything else in the combinedreward
Cryy_Stall — 24/12/2024 20:46
You created a "Save_Boost_Reward" object that is a zerosum reward

You then create a "reward_fn" object using the result of calling your object (which makes no sense, since it's an object, not a class)
Careful, that's basic python, you just tried to call on an object, not a class
Your code structure is the right one
Tech Studio

 — 24/12/2024 20:56
Holy shit I'm so stupid
Thanks for that @Cryy_Stall
My excuse for that royal screwup is I've been up for like 36 hours
:pepeDerp:
Tech Studio

 — 24/12/2024 21:08
Can't believe I didn't realize the parenthesis