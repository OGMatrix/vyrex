meciah1017
OP
 — 13/02/2026 14:02
Trained until all bots were able to consistently move / chase the ball around the map, training any further just made it so 1 bot pushed the ball and rest farmed air reward.

 Rolled back to a good checkpoint where it doesn't do that and i lowered the rewards and added a goal reward, stronger touches, and some boost management stuff. 

Issue is after 100m more steps the bot is becoming retarded and spam jumping / not touching the ball anymore. I don't even have an air reward set so im not sure why its doing it. Ive included a before and after video. 

Video where its nicely pushing ball is 100m steps before the video of it just spam jumping 
meciah1017
OP
 — 13/02/2026 14:19
Does the default VelocityBallToGoalReward punish the player when ball moves the wrong way?
Rou_Ad — 13/02/2026 14:21
do u use ggl?
meciah1017
OP
 — 13/02/2026 14:25
yes
Rou_Ad — 13/02/2026 14:25
U can check in CommonRewards
meciah1017
OP
 — 13/02/2026 14:26
i checked from my understanding it does punish. Do i just need to clamp the negative to 0 to prevent it from punishing?
Rou_Ad — 13/02/2026 14:26
its under  C:\Users.....\GigaLearnCPP-Leak\GigaLearnCPP\RLGymCPP\src\RLGymCPP\Rewards
hm
meciah1017
OP
 — 13/02/2026 14:27
im right now testing adding just goal reward and strong touches then ill add velocity ball to goal if it gets that down
Rou_Ad — 13/02/2026 14:28
I just switched to ggl but that sounds like ur bot just farms a reward.
what rewartds do u have?
meciah1017
OP
 — 13/02/2026 14:29
Samples (player-steps): 16318

Arena Mix (envs sampled this window):
1v1: 33.35%  2v2: 34.11%  3v3: 32.54%

Touch Rate (player-steps): 0.05%
Touch Rate by mode: 1v1=0.18%  2v2=0.04%  3v3=0.01%

Reward                AvgRaw        AvgW       W
--------------------------------------------------
StrongTouch         0.000103    0.00123412.000000
BallToGoal         -0.001870   -0.0037402.000000
SpeedToBall         0.034964    0.0174820.500000
FaceBall            0.156610    0.0234920.150000
AirTouch            0.000020    0.0000201.000000
SaveBoost           0.125358    0.0188040.150000
PickupBoost         0.001517    0.0015171.000000
SmallPad+           0.004351    0.0032630.750000
Goal                0.000000    0.000000150.000000

========================================
Average Step Reward: 0.0795
Policy Entropy: 0.7877

Policy Update Magnitude: 0.0040
Critic Update Magnitude: 0.0031

Collection Steps/Second: 121,754.5000
Consumption Steps/Second: 226,718.2969
Overall Steps/Second: 79,214.1406

Collection Time: 0.8041
 - Inference Time: 0.2628
 - Env Step Time: 0.1021
Consumption Time: 0.4318
 - GAE Time: 0.0012
 - PPO Learn Time: 0.3153
Collected Timesteps: 97,908
Total Timesteps: 1,220,969,873
Total Iterations: 18,999
 
Rou_Ad — 13/02/2026 14:31
im not good ad stuff like this lol.
meciah1017
OP
 — 13/02/2026 14:31
its not farming anything really, most its reward is coming from face ball / speed to ball / save boost
but its not a crazy amount
watching it play its just spam jumping
and i removed air rewards
only rewards when it jumps and touches ball in air
regular jumping doesnt reward it so idk why it desolved into spam flipping
Rou_Ad — 13/02/2026 14:32
cfg.ppo.entropyScale?
meciah1017
OP
 — 13/02/2026 14:33
i usually use 0.035 but i tried a run with 0.045 as well
and after a few dozen million just started doing same thing
Rou_Ad — 13/02/2026 14:33
then ur reward is broken somehow
idk
meciah1017
OP
 — 13/02/2026 14:34
just using stuff from commonrewards so idk if one of those is broken
Rou_Ad — 13/02/2026 14:34
there is a jump touch reward in common rewards?
meciah1017
OP
 — 13/02/2026 14:36
no but i removed it and still doing it, just did only strong touch and goal reward and it started being terrified of touching ball. i think the default strong touch reward is broken
Rou_Ad — 13/02/2026 14:37
it works for me
meciah1017
OP
 — 13/02/2026 14:37
is this the same for you?

    class StrongTouchReward : public Reward {
    public:
        float minRewardedVel, maxRewardedVel;
        StrongTouchReward(float minSpeedKPH = 20, float maxSpeedKPH = 130) {
            minRewardedVel = RLGC::Math::KPHToVel(minSpeedKPH);
            maxRewardedVel = RLGC::Math::KPHToVel(maxSpeedKPH);
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev)
                return 0;

            if (player.ballTouchedStep) {
                float hitForce = (state.ball.vel - state.prev->ball.vel).Length();
                if (hitForce < minRewardedVel)
                    return 0;

                return RS_MIN(1, hitForce / maxRewardedVel);
            } else {
                return 0;
            }
        }
    };
}
ill try lowing the min speed
Rou_Ad — 13/02/2026 14:40
yes
