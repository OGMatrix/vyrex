Boost pad locations in obs: Keep them as static numbers straight from common_values.BOOST_LOCATIONS?
stylis
OP
 — 17/07/2025 20:16
When adding boost pad locations in the obs, should I make the locations be relative to the player or keep them as static numbers copy and pasted from common_values.BOOST_LOCATIONS? Also, should I invert them or no? I am currently using RelativeDefaultObs with relative ball predictions.
NO CONSENT TO ROLV METH SALES

Role icon, Verified — 17/07/2025 20:25
when inverting pad positions it turns out it doesn't matter
i think if you invert and mirror the boost pads you just get the same array back
NO CONSENT TO ROLV METH SALES

Role icon, Verified — 17/07/2025 20:34
i wouldn't put locations in obs though
it doesn't help
if you must then have it be relative
but again
it's a constant of the world
it'll just figure it out
stylis
OP
 — 17/07/2025 20:34
Ah got it
I’m was thinking that for my bcm, having relative boost locations can possibly help
NO CONSENT TO ROLV METH SALES

Role icon, Verified — 17/07/2025 20:35
if you really want to add them
make them fully local 