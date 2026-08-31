meciah1017
OP
 — 09/02/2026 20:00
Getting into training my own bots, and a question I had was when should I introduce rewards for teammate related stuff? Should I start off the bat with rewards for not stealing ball / avoiding teammate while during the teaching them to touch ball phase? Or should I just get the bot good at touching the ball then introduce teammate rewards. Thanks!
meciah1017
OP
 — 09/02/2026 20:06
Also is my training config completely wack or overkill?
I have a I9-14900F, 32gb DDR5 6000mt/s, rtx 5070
Using about 20gb of memory with CPU spikes around 60% usage and GPU around 80%

n_proc = 60

policy_layer_sizes = [1536, 1536, 1536]
    critic_layer_sizes = [1536, 1536, 1536]

    ts_per_iteration = 200_000
    ppo_batch_size = 200_000
    exp_buffer_size = 600_000

ppo_minibatch_size=50_000,
        ppo_epochs=3,

--------BEGIN ITERATION REPORT--------
Policy Reward: 1,476.07643
Policy Entropy: 3.90309
Value Function Loss: 0.26733

Mean KL Divergence: 0.20708
SB3 Clip Fraction: 0.63912
Policy Update Magnitude: 1.00005
Value Function Update Magnitude: 0.47109

Collected Steps per Second: 53,969.44578
Overall Steps per Second: 18,332.53796

Timestep Collection Time: 3.70758
Timestep Consumption Time: 7.20722
PPO Batch Consumption Time: 0.93860
Total Iteration Time: 10.91480

Cumulative Model Updates: 1,533
Cumulative Timesteps: 35,010,128

Timesteps Collected: 200,096
 
HappyBavarian07

 — 09/02/2026 21:30
atleast for me what kinda works (i think but need to train more, which i will do with this run since it seems to be going very well) is first teaching teh bot to hit the ball then we can move on to scoring and then slowly introduce more stuff and at some point i introduce team spirit which basically gives the teammate a bit of the reward of the bot that got it making it attractive for the teammate to let the player do his thing sometimes. and ofc you can introduce rewards for not going near teammates but that can be counterproductive (for example on kickoff when you may want the second man to cheat up and go for the shot or second 50). but best thing to do is just to fafo
https://github.com/ZealanL/RLGym-PPO-Guide/tree/wip
GitHub
GitHub - ZealanL/RLGym-PPO-Guide at wip
A beginner's guide to creating a Rocket League ML bot using RLGym-PPO. - GitHub - ZealanL/RLGym-PPO-Guide at wip
A beginner's guide to creating a Rocket League ML bot using RLGym-PPO. - GitHub - ZealanL/RLGym-PPO-Guide at wip
HappyBavarian07

 — 09/02/2026 21:34
for the team spirit part. either use a zero sum reward implementation or make a small wrapper that just does the team spirit calculations from the zerosum reward and then slap that around each reward that you want to have team spirit in
but thats just my way of going about it
meciah1017
OP
 — 09/02/2026 22:25
are you doing 2v2 the whole time or starting as 1v1?
HappyBavarian07

 — 09/02/2026 22:26
i train all modes with around equal distribution (i think 34% 1s, 33% 2s, 33% 3s) but then some arenas are training from kickoff states everytime or just kickoffstates and some just random states.
when learning to touch the ball not the most optimal since in that stage we just want to get a lot of experience and a lot of iterations in (so training only 3s would be good (i think))
meciah1017
OP
 — 09/02/2026 22:29
thanks
HappyBavarian07

 — 09/02/2026 22:29
but in stages where we wanna actually learn to play the game and score just train all modes unless you wanna make a bot just for 3s or 2s or 1s
then obviously just train that mode