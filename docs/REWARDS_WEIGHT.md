justin
OP
 — 16/02/2026 17:35
i wanna implement the aerial distance reward on my bot, is there a way to know approximately what weight i should put other than fucking around
Rou_Ad — 16/02/2026 17:36
no
Martico2432

Role icon, Verified — 16/02/2026 17:36
no
Rou_Ad — 16/02/2026 17:36
Im struggling with that too lol
Martico2432

Role icon, Verified — 16/02/2026 17:36
always FAFO
Rou_Ad — 16/02/2026 17:36
I just cant figure out the right weights. or my jumptouch reward is just shit
Martico2432

Role icon, Verified — 16/02/2026 17:37
just calculate avg reward when it will get it, and then think
Rou_Ad — 16/02/2026 17:37
lol easier said then done
Martico2432

Role icon, Verified — 16/02/2026 17:37
i mean, tru
Rou_Ad — 16/02/2026 17:37
my bot just started airdribbling
I ned to restart because it completely ignored scoring opportunitys
Martico2432

Role icon, Verified — 16/02/2026 17:38
roll back
Rou_Ad — 16/02/2026 17:38
bro I was at 9b steps and it started at like 3 b
sry
justin
OP
 — 16/02/2026 17:39
may i know what weights you put?
Martico2432

Role icon, Verified — 16/02/2026 17:39
fafo
Rou_Ad — 16/02/2026 17:39
dont forget to save ur checkpoints
justin
OP
 — 16/02/2026 17:39
oof that’s tough
Rou_Ad — 16/02/2026 17:39
lol
justin
OP
 — 16/02/2026 17:39
dw i always do backups
choccy milk — 16/02/2026 17:47
seems to be farely dependant on how you set it up
I was able to get my guy sorted in around 80M steps by just spawning a ball in the sky and letting it try and hit it over and over again until it learned how
then let it try it in games later
you'll probably be able to do that with rocketsim
justin
OP
 — 16/02/2026 17:48
so i change the parameters so the ball spawns high and then i let it cook?
choccy milk — 16/02/2026 17:48
pretty much
Martico2432

Role icon, Verified — 16/02/2026 17:48
state setter
justin
OP
 — 16/02/2026 17:48
then put the old settings back?
choccy milk — 16/02/2026 17:49
yup once it learns how just change it back to how you had it before doing self play or whatever
justin
OP
 — 16/02/2026 17:49
thank you
choccy milk — 16/02/2026 17:49
nws