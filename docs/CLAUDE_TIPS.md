Latest RLGym/RLBot Techniques Worth Researching
Since you're training a bot right now, here's what the current state of the art looks like (so you know what to point OpenClaw at):
RLGym v2 + RocketSim is the current recommended stack. RocketSim runs training much faster than the actual game, and PPO collects experience where each timestep advances 8 physics ticks Rlgym. The key components:

LookupTableAction as the action parser (discrete action space with a pre-built lookup table of all meaningful button combinations)
CombinedReward for reward composition — the community consensus is that gentle, general rewards produce the best long-term behavior
Nexto mostly used very general and gentle rewards, and its passive dribble-flick playstyle with mostly forward flicks seems to be a natural evolution of basic ballchasing behavior GitHub

Key reward design insights from ZealanL's guide:

Using sqrt() for boost-related rewards makes boost more important the less you have, reflecting actual Rocket League dynamics GitHub
Bots tend to ignore small pads, so making the small pad pickup reward much stronger than 12% of the big boost pickup reward is recommended GitHub
Passiveness is a common flaw — bots prefer to wait to save rather than fight to score, partly because faster reaction times make being passive more viable GitHub. Aggressive play rewards help counter this.

State mutators are important for training diversity — using KickoffMutator combined with random state setups from high-level replays exposes the bot to more scenarios. The Necto project also explored using replay files to learn from human gameplay (inspired by Video PreTraining), seeing years of gameplay before training with RL GitHub.
Self-play and ELO tracking are used to measure improvement — the bot training framework rocket-learn uses an ELO-like rating system to track the skill of the bot against previous versions GitHub.