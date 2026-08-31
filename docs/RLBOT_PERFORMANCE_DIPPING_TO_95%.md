
JeffA233 — 20/02/2026 06:39
https://docs.pytorch.org/docs/stable/threading_environment_variables.html there's these as well
I'm not positive if/how libtorch works with that function (set_num_threads()) but it seems like it should do the same thing so I'm not sure 
PigMasta
OP
 — 20/02/2026 06:41
setting it to 1 thread felt better for my experience in-game and my cpu usage was like 10%
but the bot was no longer staying at 100%
JeffA233 — 20/02/2026 06:43
:pepeShrug: just seems odd to me, you have a theoretically more powerful CPU than I do with a policy that is only double what I've ran and I get like max a few percent of CPU usage per agent
the set_num_thread part was really important though lol, otherwise one agent/bot used like 30%
and yes they run at 100% unless I forget to call that function to set threading to 1 lol 
PigMasta
OP
 — 20/02/2026 06:45
i've gotta be doing something wrong
JeffA233 — 20/02/2026 06:46
and even better yet I'm using PyTorch :Kek:
so yeah idk
PigMasta
OP
 — 20/02/2026 06:46
like if i'm not playing the bot i dont really care for the fps instability, it's whatever
but when i play it idk, it just irks me
PigMasta
OP
 — 20/02/2026 07:12
I'm too tired to try it tonight but what I'm gonna try as a sanity check is to make a model of the same size in the python version and try running that on one thread
That would at least tell me if it's something with my PC or if it's just something with CPP/GGLBot/GigaLearn somehow
Cryy_Stall — 20/02/2026 11:27
It could be that there is something with nova's wrapper too, it's fairly new, so maybe you've discovered a bug
I would believe that c++ is more than fast enough to compute in less than 1/120 seconds
NO CONSENT TO ROLV METH SALES

Role icon, Verified — 20/02/2026 17:13
❌
My bots do this too
In the Zealan approved style
PigMasta
OP
 — 20/02/2026 17:35
Do you have a different way of making the bot compatible with rlbot?
NO CONSENT TO ROLV METH SALES

Role icon, Verified — 20/02/2026 17:43
Slightly
PigMasta
OP
 — 20/02/2026 17:44
:kannainspect:
I doubt it'd help me at all especially since you're getting the same issue lol
NO CONSENT TO ROLV METH SALES

Role icon, Verified — 20/02/2026 17:46
It’s not really an issue
PigMasta
OP
 — 20/02/2026 17:48
It'd be nice to have confidence in it though, I don't know for sure if that 5% loss is actually impacting game performance
PigMasta
OP
 — 20/02/2026 17:56
What if I just somehow get the bot to be hosted on my server a cook that CPU while my PC is smooth sailing
Genius idea more like overcomplicating and avoiding the problem
Another actual thing I could try is porting the bot to python then from there making it compatible with rlbot but I'd wanna test the same size model made in python first
PigMasta
OP
 — Yesterday at 02:26
I have failed at this
I was able to get the checkpoint converter script provided with GigaLearn to work, at least in the sense that it executed without errors, but trying to run the bot I get errors relating to the state dict and I'm assuming it's something to do with the fact that I have no clue how to replicate GigaLearn's obs in python and I'm not even sure the action parser is the same, they sound the same: lookuptable and default, for python and gigalearn respectively, both 90 actions 
i know it shouldnt but this small issue is kinda killing my motivation lol, what's the point of training a bot if i cant fully enjoy playing against it
NO CONSENT TO ROLV METH SALES

Role icon, Verified — Yesterday at 03:37
@PigMasta
it's like... a 5% difference
i promise you it has no bearing
don't bother
python rlbot bots do this too
literally just noise
Cryy_Stall — Yesterday at 12:00
No? What you on about
Always got my bots running at 100 and sometimes 99
NO CONSENT TO ROLV METH SALES

Role icon, Verified — Yesterday at 14:28
My Python bots do this
The RLBot pack bots do this
Cryy_Stall — Yesterday at 14:28
i never got any of my bots to reach under 98, and it was because i was printing loads of shit
NO CONSENT TO ROLV METH SALES

Role icon, Verified — Yesterday at 14:28
Well the RLBot pack bots don’t have any extra fluff
And it still just happens
Cryy_Stall — Yesterday at 14:29
you're not running the game at 500fps aren't you ? :pepeKekw:
NO CONSENT TO ROLV METH SALES

Role icon, Verified — Yesterday at 14:30
240 usually
120 in RLBot
Cryy_Stall — Yesterday at 14:30
no vsync?
NO CONSENT TO ROLV METH SALES

Role icon, Verified — Yesterday at 14:31
No VSync
Cryy_Stall — Yesterday at 14:31
i personally run no vsync 120 fps, lower res and never got any issue 
granted i didn't try the botpack
fwiw when i run astra in rlbot, which is the classic sb3 bot, it runs just fine 
eh well it's not really classic because i removed the sb3 stuff
yeah i'm running a nexto 3v3, they are all at 100%
sometimes drop down to 99.2%, but goes back up to 100 after
NO CONSENT TO ROLV METH SALES

Role icon, Verified — Yesterday at 14:38
V5 mang
Cryy_Stall — Yesterday at 14:38
yea i'm running v5 :pepeKekw:
PigMasta
OP
 — Yesterday at 20:58
I was having issues even testing a rlgymv2 bot of the same layer sizes, like i could get it in-game but it just wouldnt move, it was throwing only invalid actions so it crashed initially but when i had it just repeat the last valid action instead they didnt move
my guess is because the bot was only up to like 300k steps? but it should still just do random actions basically right?
but if it was actually running the second time then it was staying at 100% running on a single thread