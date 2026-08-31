jthn
OP
 — 20/02/2026 23:53
I can put energy at 20,000 and critic doesnt care at all
juan diego — 20/02/2026 23:54
umm are u 100% sure ur reward isnt bugged and is actually running?
nullptr — Yesterday at 00:23
Sounds like you have a bug in your code. Follow the logic of your code and talk it out to yourself to see if you can find why the bot isn't responding to a reward change.

What do your graphs show?
jthn
OP
 — Yesterday at 03:00
Ive done exstensive testing. my critic is completely and utterly fucked. ive tried to kind of soft reset it
jthn
OP
 — Yesterday at 03:01
My graphs indicate the rew is firing, the bot is improving yes but wont adapt to rewards, i can put a continuous on 5k for standing still and the bot will not adapt to that
juan diego — Yesterday at 03:07
what mfs say will happen if u dont use leaky relu 
I would lowk reset ur critic maybe use leaky relu if ur not already and then just keep re-pasting the policy so that ur policy model doesnt get cooked
(if thats actually whats happening) although cant imagine why thatd be the case
juan diego — Yesterday at 03:10
maybe some other cooked hyper params??
jthn
OP
 — Yesterday at 03:55
unironically, i dont use leaky..
