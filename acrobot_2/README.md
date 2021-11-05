
### First NN Reinforced Learning model


![](https://cdn-images-1.medium.com/max/800/0*6V6O_IPAQlZXiOJS.gif)

### Introduction

This is a log of my learning from copying a Deep Q-network with pytorch

### Copying Others

_I am copying code and ideas from this_ [_tutorial_](https://towardsdatascience.com/deep-q-network-with-pytorch-and-gym-to-solve-acrobot-game-d677836bda9b) _and associated_ [_code_](https://github.com/eugeniaring/DQN/blob/main/acrobot.ipynb)

### Gym Environment

**Background**

Acrobat seems to be a university control theory exercise. Acrobot is a linked rods hinged at a hinge and connected at an elbow which can be actuated. Acrobot mimics an acrobat on a trapeze then it can be assumed that the actuation is only on the second hinge.

**Goal**

There is a height above the top hinge where the Acrobot’s ‘feet’ should cross.

**State**

The state consists of the sin() and cos() of two rotational joint angles and joint angular velocities:

[cos(θ1),sin(θ1),cos(θ2),sin(θ2),v1,v2]

Top hinge downwards is an angle of 0. State of [1, 0, 1, 0, …, …] means that both links point downwards.

**Inputs**

There are 3 inputs `+1`, `0` or `-1`. Since the Acrobot mimics an acrobat on a trapeze then it can be assumed that the actuation is only on the second hinge. and there is actuation in either direction.

**Reward**

At each timestep (not episode), reward is `+1` if feet reach height otherwise `-1`

**Terminal Condition**

when done or total reward -500

### Approach

[_[source]_](https://www.henrypan.com/blog/reinforcement-learning/2019/12/03/acrobot.html)

Model used _actor-critic_ with Deep Q learning to train. Assumed an infinite time horizon when deciding on the algorithm, since this problem requires the agent to build momentum to actually swing up to the top, which means the policy can’t be short-sighted and needs to consider all the actions it took to build momentum and finally reach the goal.

**Key considerations:**

-   The state space is continuous, it was be inefficient to represent the state-action values (q-values) in traditional tabular form. A sample of Q-Tabling is used to manage this
-   This problem does not require a global optimal solution, we consider the problem solved after reaching a reward of larger than -100. This means that we can trade-off the accuracy of the algorithm in exchange for a more efficient training process while still finding a near-global optimal policy.
-   Dense rewards. There is no final reward, rather the reward is given at every time step, and represents how far/close the agent is from the goal, so the feedback is very “real-time”.

### Baseline

Ten ‘Random Agent’s which choose actions at random are put in the environment and None of the agents reach the objective.

### Memory

I’m curious on how the use of memory can affect learning. In this case, memory is random episodes with Q-values related to each episode. This leaves chance to pick all the worst episodes and the actor may just always choose the worst choices given these scenarios.

How is the memory managed? is it flushed when the capacity is reached? Is always the lowest Q-values removed?

However in human memory there may a value related to each memory and higher valued memories, sometimes good and bad are easier to remember. Whereas other less striking memories are forgotten. In addition, memories tend to mix and merge in some cases maybe taking the best experiences and knitting them into one sequence.

### Scores

![](https://cdn-images-1.medium.com/max/800/1*CjZZc82ftJovUtE62WkZoQ.png)

Reward in article

![](https://cdn-images-1.medium.com/max/800/1*aTFGiixx0n490U1imuUR3g.png)

My reward curve

I added a reward if ‘done’ where the reward of 100 is given for completion. I see this as important to signal the objective of the game with a reward. This may cause overfitting by giving a premium to solution it already knows. If I compare the reward graph in the [article](https://towardsdatascience.com/deep-q-network-with-pytorch-and-gym-to-solve-acrobot-game-d677836bda9b), my graph starts to increase much earlier. Also has much less unstable growth.

We also share the same random seed so at episodes ~560 and 700 we both see a dip in score.

there seems to be a natural ceiling at a score of 62–63. I do not know if that is the best possible performance of it is actually a limitation with the code.

### Functions

def **choose_action_softmax(net, state, epsilon):** ...  
 return **action, net_out.cpu().numpy()**

_This function runs current state and epsilon (or temperature) through the neural net . Compiling? the recommended action through softmax (scaling it by some temperature) to get a resultant action and q-table._ Temperature here is a type of uncertainty.

def **update_step(policy_net, target_net, replay_mem, gamma, optimiser, loss_fn, batch_size)  
    ...** return **None.**

Most of the work in the program comes from the update, if I can understand this I can find out how the nn stores experiences and uses the optimiser to amend the weights in both policy and target net.

> Written with [StackEdit](https://stackedit.io/).
