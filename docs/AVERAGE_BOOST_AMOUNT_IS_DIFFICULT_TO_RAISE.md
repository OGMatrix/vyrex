Average boost amount is difficult to raise without having a save boost reward that drowns rewards
sly
OP
 — 08/12/2025 04:21
So I'm training a 2v2 bot and I'm around 50b steps, and I'm running into an issue with boost management, really been stuck on this issue for a good 2 weeks now lol...

Bot is higher level than necto, but only by a bit, boost management and rotations have been the biggest issue for a little, the bot DOES grab boost, however not efficiently and tends to avoid boost frequently.

Current training setup:
tick skip 8 (I did some weird ass shit and modified my tick skip to 12 to increase reaction time and yes this did work, but then decided what if i change it back mid training, and it actually improved the bot a LOT)
kickoff 40%, random states (variation of on ground and in air) at 60%
zero sum reward is used on most of my rewards (besides goal and personal player movement based rewards)
~50% air ratio (could be causing boost issues, but when watching the bot it ignores a LOT of boost pads it could easily take)

The problem:
 
With save boost reward at low weights, the average boost stays at ~17%
To get boost above 25-30%, save boost reward needs to be so high that it becomes 60%+ of the total reward
At 60% dominance, my assumption is that a lot of other rewards are being drowned out

(theres also a chance my calculations on the percentage that save boost is reaching may be miscalculated? ill try re-looking into it but im pretty sure im calculating it right...? if anyone wants to give some insight on common issues when calculating reward totals like zero sums potentially messing shit up that could be an issue im overlooking)

I noticed that necto and other high level bots replace the save boost reward entirely in favor of a boost usage penalty to maintain their high boost, but that doesn't sound logically superior than teaching it to save boost directly? I feel like it would find itself on low boost frequently but then again necto doesnt seem to nearly as much as mine... really unsure on that.

Also side questions:
currently ive applied a team spirit of .5 and penalty of 1 to touches, dribbles, and kickoff related states, and i set the team spirit and penalty to 0 for individual mechanics like wavedashes, goals (.75 team spirit currently), and no wrapper for air and energy reward (i still use air reward, i found when i took it out shit got weird but maybe just me being stupid?), so what im asking, is this a good approach? is there anything i shouldnt be adding penaltys or team spirit from the ones ive mentioned?

when i had team spirit on saveboost reward, idk if im insane but one bot would hoard 100 boost while teammate had 0, i removed it and its slightly balanced, but still not great, not sure if this is all training time / placebo that i need to stop paying such close attention to or what

i also had a weird issue where the bot at around 32b steps started flipping onto its back and staying there, so i added a faceup penalty (penalty for being upside down near the ground), seems to have fixed it, but the real question: is this a common issue? is this an effective fix?

general question: for high level bots, in terms of learning rate, what is best practice? from what ive researched it seems to be very preference and observation based, if bot not improving / plateuing, lower learning rate, is this the general consensus or have i simplified it way too far?

anyways sorry for the MASSIVE yap, and my capitalization deterioted throughout the post, ive had a lot of questions that i just kept to myself trying to figure out, and finally decided i need outside insight

thanks for any help! 
sly
OP
 — 08/12/2025 04:22
also part of me is thinking i should restart with a new bot instead of pushing through all these changes i made, opinions?
sly
OP
 — 08/12/2025 06:40
current experiment = set save boost reward super high and pray 
and also ignore my reward percentage charts
sly
OP
 — 08/12/2025 06:51
horrible idea
trying out boost usage penalty
sly
OP
 — 08/12/2025 19:54
welp, saving that model for now and restarting from scratch, really stumped on this still
sly
OP
 — 08/12/2025 21:32
wait... a minute...
i still had velocity player to ball in my rewards...???
WELL NO SHIT IT DOESNT THINK TO TURN AWAY FROM BALL AND GO BOOST?? am i crazy or am i stupid
i dont know anymore... its been weeks...