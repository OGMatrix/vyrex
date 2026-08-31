Mahkmoud
OP
 — 18/02/2026 16:51
kinda lost on how to do this, maybe like checking if the last flip was backwards and giving it reward for cancelling but I have a strong feeling its going to not turn out how I expect
justin — 18/02/2026 16:52
NO CONSENT TO ROLV METH SALES

Role icon, Verified — 18/02/2026 16:53
Cryy_Stall — 18/02/2026 16:53
mh, i've never heard of a half flip reward, maybe you can try face ball reward and state set the bot facing away from the ball ?
i'm thinking about it from a bot perspective, but half flip is something that appears in human gameplay in like gold/plat, it's a result of needing to rotate faster in your net
Mahkmoud
OP
 — 18/02/2026 16:55
thats lowkey possible, I really have to start using state setters
Cryy_Stall — 18/02/2026 16:55
the thing is, if you somehow manage to make a half flip reward, i feel like it could be farmed too easily
Mahkmoud
OP
 — 18/02/2026 16:56
yes right, it would have to be a gentle reward but then I feel like it wont do it
but maybe with state setters its possible, ill try that
justin — 18/02/2026 16:57
what if you put the reward a lil higher and when he starts to do it a little you lower it progressively
Cryy_Stall — 18/02/2026 16:57
could work, but you need to be careful and have good metrics to detect when to lower
also it might lose the essentials
(driving to ball, boost, stuff like that)
Mahkmoud
OP
 — 18/02/2026 16:57
yes thats what im thinking
justin — 18/02/2026 16:57
is the half flip reward even necessary
Cryy_Stall — 18/02/2026 16:58
yeah i don't think it really is
justin — 18/02/2026 16:58
i feel like he’ll find a way to replace it or even do it
Mahkmoud
OP
 — 18/02/2026 16:58
I havent really used satte setters, whats the issues in running iterations with high reward in such a state like its turned away from the ball, where it would end after the ball was touched and then running it in real games with low reward
NO CONSENT TO ROLV METH SALES

Role icon, Verified — 18/02/2026 16:59
no
state setters are lwk not worth it
Cryy_Stall — 18/02/2026 16:59
invalid take, refused opinion
NO CONSENT TO ROLV METH SALES

Role icon, Verified — 18/02/2026 16:59
just do random state
and kickoffstate
i sauced up REAL rolv ratios
for this
25% kickoffs
75% random states
Mahkmoud
OP
 — 18/02/2026 17:00
yeah I only have kickoff state thats lowkeya  big issue, I need to get on that random state
and I need to make a training program to set my values im literally just eyeballing it rn with a couple interpolated values sometimes
crystall why do you say no point in half flip
NO CONSENT TO ROLV METH SALES

Role icon, Verified — 18/02/2026 17:06
if your bot needs to
it will learn it
justin — 18/02/2026 17:06
useless mech first of all and second of all he’ll adapt
Cryy_Stall — 18/02/2026 17:06
i didn't say no point in the mechanic, i say no point in the reward
Mahkmoud
OP
 — 18/02/2026 17:06
mm
Cryy_Stall — 18/02/2026 17:06
because it's more of a mechanic that will be learned as a consequence of face ball / faster rotations
Mahkmoud
OP
 — 18/02/2026 17:06
do good bots generally learn half flip
Cryy_Stall — 18/02/2026 17:08
i'm not sure about that, i don't remember
from what i saw, they try to do it, i sometimes see failed attempts at half flips
Mahkmoud
OP
 — 18/02/2026 17:09
makes sense, im just scared about setting my face ball too high
Cryy_Stall — 18/02/2026 17:09
eh that's the whole reward engineering reason 
Mahkmoud
OP
 — 18/02/2026 17:10
fuck it we got backups
Cryy_Stall — 18/02/2026 17:11
eh, you'll revert if it picks up on bad habits
Mahkmoud
OP
 — 18/02/2026 17:11
thats what im saying though if a good half flip reward was there I wouldnt need to make my bot ballchase to get it
PigMasta — 18/02/2026 17:23
Is this as simple as just modifying the environment creation for GGL to do exactly that? Or is there more depth to that implementation
NO CONSENT TO ROLV METH SALES

Role icon, Verified — 18/02/2026 17:28
i'm gonna stop helping you publicly
please regardé les contents of your inbox
Mahkmoud
OP
 — 18/02/2026 18:08
result.stateSetter = new CombinedState({
        {new KickoffState(), 0.5f},
        {new RandomState(true, true, false), 0.5f}
    });
SubparNova — 18/02/2026 18:38
Yes, my bots have learned half flips just fine without a reward for it
Mahkmoud
OP
 — 18/02/2026 18:45
good to know
justin — 18/02/2026 19:36
tu parles français?
c’est clair y’a aucun anglais qui va écrire « regardé » avec é en plus
y’a les aussi
NO CONSENT TO ROLV METH SALES

Role icon, Verified — 18/02/2026 20:20
@🌊 Foxe 🔥 🇨🇦
traducter svp
Zealan

 — 18/02/2026 20:21
🌊 Foxe 🔥 🇨🇦

 — 18/02/2026 20:21
you speak french ?
it's clear that there not english person that will write ''regardé'' with added é
---------------------------------------------
he said that @NO CONSENT TO ROLV METH SALES 
justin — 18/02/2026 20:41
how r you not french but you wrote « regardé » and « les » while not speaking french
justin — 18/02/2026 21:09
« svp »
NO CONSENT TO ROLV METH SALES

Role icon, Verified — 18/02/2026 21:41
j'suis geekd en tbrnk
Zealan

 — 18/02/2026 21:41
man i wish i studied french in high school lol
itd be so sick to know it atp
im jealous of u :Madge:
Mahkmoud
OP
 — 18/02/2026 21:58
I start getting nervous when theres too many french people around
justin — 18/02/2026 22:07
lol jsavais
justin — 18/02/2026 22:07
why
Mahkmoud
OP
 — 18/02/2026 22:08
I feel like theyre gonna surrender
makes me nervous
justin — 18/02/2026 22:10
i don’t understand