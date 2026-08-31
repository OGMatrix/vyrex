Flipping and boost
T0ddler.
OP
 — 14/02/2023 18:22
I have trained my bot (based on Impossibums tutorial) for just under 1B steps,. and it is going pretty well, it can beat the All-Star bot at least lol. I have noted that as further it goes in the training, the less it flips. Right now my bot never flips, unless it gets the ball on top and opponent really close, so it does not flip to gain speed at all. At some point earlier in training it flipped for recoveries, but it never learned to do it efficient. Neither does it seem like it picks up any boost on purpose (although I am less worried about that). 

I have some rewards for speed and for boost, but I´m afraid it will get stuck in bad habits if I does not change something, but I do not just want to increase said rewards as it could have a lot of bad side-effects. As boost pickups is kinda random and will probably mess up the whole thing if it is too large a reward e.g. For my velocity rewards I am actually decreasing it slowly right now since I want the bot to focus more on hitting the ball hard, so yeah.. tricky

So my question is, anyone that has made specific rewards for flipping for gaining speed? Or did it just come with time randomly?
Kaiyotech — 14/02/2023 18:28
Flipping for speed is very hard to get right. Your experience is very common. Often if comes back with time.
T0ddler.
OP
 — 14/02/2023 20:39
Ty for response. I was just wondering if it would be helpfull or hurtfull to e.g. make a statesetter that only lasted a few seconds maybe, and then having only velocity reward on, so no goals or stuff, just a race to the ball and then reset? Would doing an intense session of this (for a night maybe) have bad consequences maybe, like messing up the entire model? Or could it potentionally help the bot to learn flipping, ideally speedflipping for when far away from the ball?
Kaiyotech — 14/02/2023 20:41
Ideally you don't want to steer your model in only one direction. You could mix in your setter and new rewards and stuff but just need to be careful with weights basically. You'll just have to try things and see what works though
T0ddler.
OP
 — 14/02/2023 20:43
Unrelated, I have made a "RecoveryReward" that I think is working, but I don´t have a good way of checking if it is working correctly (except looking at the txt files for the avg rewards etc.)
Is there a way to see the real-time reward while watching the bot play, to make sure it works properly?
Kaiyotech — 14/02/2023 20:44
Not really. You'd need to just debug step through probably
T0ddler.
OP
 — 14/02/2023 20:45
Okay, darn it, tysm tho!