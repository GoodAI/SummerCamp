# Reinforcement learning module using Monte Carlo AIXI CTW node.
![](http://i.imgur.com/aOGxfr2.png)
This node is able to do reinforcement learning in a wide variety of environments, including  POMDP.

It will work in *worst case scenario*:
* unknown environment
* noisy observations
* stochastic effects of actions
* aliasing of perceptions
* no explicit notion of state
* rewards only sparsely distributed

...but it will take quite a long time.

[kt]: https://en.wikipedia.org/wiki/Krichevsky%E2%80%93Trofimov_estimator
[AIXI]: http://www.hutter1.net/ai/aixigentle.pdf
[pUCT]: http://jveness.info/publications/nips2010%20-%20pomcp.pdf
[UCT]: http://www.sztaki.hu/~szcsaba/papers/ecml06.pdf

## Short introduction to algorithm

From point-of-view of the algorithm, the world looks like a sequence of bits, in which are encoded past rewards, observations, and actions. Algorithm has set from beginning what is a number of bits representing observations/rewards/actions. It interacts with the environment by giving bits that encode chosen an action and receive bits that encode observation and reward.

![interaction](http://i.imgur.com/vxavfqf.jpg)

This algorithm tries to approximate general [AIXI], which is formalism showing the (incomputable) notion of what is an optimal action for an agent in the unknown environment.

The internal model of environment for original AIXI is a probability distribution over all possible programs that could represent the environment, where the apriori probability of one possibility (ie: one particular program) depends on its complexity.

The internal model of environment MC-AIXI-CTW is a probability distribution over all predictive suffix trees (PSTs). Where apriori probability of one possibility (ie: one particular PST) depends on its complexity. This is done by context-tree weighting (CTW)
algorithm.

### PSTs and CTW
Bellow is example of one simple PST:

![Example of PST](http://i.imgur.com/jNDnV3p.png)

When we want to use this PST to predict probability of next bit in interaction (eg: what will be first bit of our observation in next round?), when we saw sequence (for example) 01000*10* so far, it can be done this way:

- We start in the root and look at last bit we saw: it was 0. We, therefore, go to a child under edge with a label "0". 
- We are not in leaf, so far, so we continue in the same way. Second-to-last bit was 1, so we end up in node  θ <sub>01</sub>.
- When we are in leaf, we find there the probability of next bit being 1, in this case, we predict that next bit will be 1 with probability 0.3.

CTW part of MC-AIXI-CTW uses context tree weighting to compute a mixture of predictions of all PST of a certain depth, where each has a weight corresponding to its complexity. Thetas of each PST are computed by [KT-estimator][kt]. Importance of this algorithm is that it is able to compute mixture of predictions of all 2<sup>D</sup> possible models, which would naively took 2<sup>2<sup>D</sup></sup> steps, in O(D).

CTW was originally used for compression, but reusing it for prediction is natural since these two problems are closely related. When we are compressing, we can forget everything we can predict. Therefore, having a good predictor means having a good compressor. Original AIXI achieves the best action possible, by using best compressor possible: Kolmogorov complexity.

### Monte Carlo
So far, we have rough idea about how to model (ie: being able to predict reactions of) environment. When we have this model, it is necessary to plan what action to take.  Most well-known algorithms for this (eg: α-β-pruning) are not usable because:

- we do not have the heuristic estimation of quality (goodness/badness) of a state. Our algorithm runs in the unknown environment.
- There is no explicit "state" - The state of the environment is hidden, algorithm sees just some output. From a point of view of MC-AIXI-CTW state we are in is whole history of interaction.

To solve this, Monte Carlo Tree Search (MCTS) algorithm, widely used in bots playing GO, is used. When we consider what to do in some state, instead of considering possible next states from "state space" like in minimax-like algorithms, we try to play several games starting in our current state against our model of the environment while choosing our actions randomly.

How MCTS works:
![pUCT](http://i.imgur.com/1h3Funl.png)

This way, an algorithm can make a tree model, like on the image above. With information about to what situation will various actions lead to, and compute average expected reward of various actions. To pick what action to take it uses [ρUCT] (modified [UCT]) to keep a balance between exploration and exploitation.

# What is in this package
## AIXINode
This node has three inputs and onr output. Two of inputs are reward and observation and output is action. The third input is EnvironmentalData.

AIXINode needs to have some information about the environment, like a number of actions, etc. This can be configured in settings of the node, or via this input directly from the environment (world AIXIEnvironmentWorld provides line with this information).

Note: In this version, maximal size of each observation, reward and action is just 23b, and the sum of a number of bits needed for observation and reward has to be less than 32b.

#### Parameters. 
![](http://i.imgur.com/gsAgnUo.png)

It is possible to set several parameters of AIXI Node:
- The depth of Context tree - That is how far back (in bits) should agent look while deciding about what to do next. Time complexity grows O(N). Maximal space usage is O(2<sup>N</sup>), but this is not achieved in practice. (common values: 4-96).
- Agent horizon = How many rounds into the future look while planning. (Common values: 3-6)
- Initial Exploration and Exploration Decay: Probability of doing random action, instead of planning using monte-carlo, is (initial-exploration)*(exploration decay)<sup>(# of round)</sup>.
- MC Simulations. How many times do Monte Carlo sampling. Common values are 100-1000
- Experimental period: when set to non-zero positive number N, agent will start do random actions in first N rounds.

Agent also has to know something about the environment it is in. This can be set in AIXI Node properties or via input EnvironmentalData (described bellow).

Note: a large number of possible observations and rewards has little impact on performance. Actions are searched in linear order, thus there should be smaller number of actions.

![](http://i.imgur.com/AgrJuWa.png)


## World AIXIEnvironmentWorld
In the package is included world AIXIEnvironmentWorld that contains several standard games from literature about genral reinforcement learning. References can be found in [Vennes (2011)][jvPhD]. It also provides a vector of 10 numbers Environment Data, that can be connected to AIXINode, that will use it to automatically set parameters of an environment, that has to be set by hand otherwise.

There are these values in Environment Data vecotor:

0. action_bits
1. reward_bits
2. observation_bits
3. perception_bits (sum of two above)
4. min_action
5. min_reward
6. min_observation
7. max_action
8. max_reward
9. max_observation

## Available games
There is example brain for each of these games:

#### CoinFlip
This is an extremely simple environment. In each round, environment flips a biased coin that has probability 0.9 of being tails (coded as 0). Agent try to guess it, and environment says what it was and what real result. The reward is 0 for incorrect guess and 1 point for correct guess.

#### Tiger
In this environment, there are two doors. Behind one is a pot of gold. And behind the second one is the tiger. Agent can open one of the doors (10 points for the pot of gold, -100 for tiger), or listen (-1 point), which will reveal the position of the tiger with probability 0.85. After opening the door, the game starts over with the random position of gold and tiger.

#### Extended Tiger
This game is similar to Tiger, but Agent starts sitting. Available actions are: open left door, open right door, listen, stand up. Listening works only while sitting. Reward are similar like in Tiger, but 30 points for finding the pot of gold. This environment require planning more actions ahead than others.

#### Biased Rock Paper Scissors
Here agent plays rock-paper-scissors against an opponent who will repeat rock if it won the game using it.

#### Cheese Maze
Agent is in the maze (bellow) and has actions: left, right, up, down. And rewards are: -10 for bumping into a wall, 10 for finding cheese, -1 for moving to free cell. The agent is observing 4 bits corresponding to seeing walls around cell he is in. That means he has ambiguous observations. 

![Cheese Maze](http://i.imgur.com/0iZwU8T.png)

#### Tic-Tac-Toe
Here agent plays repeated games of tic-tac-toe against oponent playing randomly. Nine actions are "place mark to field no. N". Observation is whole playing field. Rewards are 2 for winning, -2 for losing, -3 for an ilegal move (field is already taken). There is much more possible observations here, than in other worlds.

### Overview of games
| Game           | # of actions | # of observations | Perceptual aliasing | Noisy observation |
|----------------|--------------|-------------------|---------------------|-------------------|
| Coin flip      | 2            | 2                 | No                  | Yes               |
| Tiger          | 3            | 3                 | Yes                 | Yes               |
| Extended Tiger | 4            | 3                 | Yes                 | Yes               |
| Biased RPS     | 3            | 3                 | No                  | Yes               |
| Cheese Maze    | 4            | 16                | Yes                 | No                |
| Tic-Tac-Toe    | 9            | 19683             | No                  | No                |


## AIXILibrary
This is a Brain Simulator-independent C# library, implementing MC-AIXI-CTW.

Example of usage:
```c#
// Description of these options is in AIXIStandalone/AIXIStandalone/Program.cs
var options = new Dictionary<string, string>();
options["ctw-model"] = "ctf";
options["exploration"] = "0.1";
options["explore-decay"] = "0.99";
options["ct-depth"] = "8";
options["agent-horizon"] = "4";
options["mc-simulations"] = "200";

var env = new CoinFlip(options);

var agent = new MC_AIXI_CTW(env, options);

// interaction loop
while (True){
      //give observation and reward to agent, so it can update its model of environment
      agent.ModelUpdatePercept(env.Observation, env.Reward);

      // Let agent pick action according to UCT
      int action = agent.Search();

      env.PerformAction(action);

      // Let the agent know that we really want to do this action.
      agent.ModelUpdateAction(action);
}
```
There are included two Visual Studio solutions: AIXIStandalone and AIXIModule. Former is for using AIXILibrary without Brain Simulator. Later is for using Brain Simulator. 

[jvPhD]: http://jveness.info/publications/veness_phd_thesis_final.pdf
[mcaixictw]: https://www.jair.org/media/3125/live-3125-5397-jair.pdf
[rockiThesis]: http://olab.is.s.u-tokyo.ac.jp/~kamil.rocki/phd_thesis.pdf
[parmcts]: https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf
[rockipaper]: http://olab.is.s.u-tokyo.ac.jp/~kamil.rocki/rocki_scai11.pdf
[mcsurvey]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6145622&tag=1

##

## Credits
This project was created for [GoodAI](http://goodai.com), during [Summer Camp](http://datalab.fit.cvut.cz/events/52-summer-camp-2015) at  [FIT ČVUT](http://fit.cvut.cz/). Some images used are by Joel Vennes. It was heavily inspired by [pyaixi](https://github.com/gkassel/pyaixi) by Geoff Kassel and [mc-aixi](https://github.com/moridinamael/mc-aixi) by Daniel Visentin and Marcus Hutter.

## Literature

- Introduction to MC-AIXI-CTW is in Ph.D. thesis of [Joel Veness][jvPhd].
- Main paper on this is [A Monte-Carlo AIXI Approximation][mcaixictw] by J Veness et al.
- Some ideas on parallelisation of monte-carlo tree search are in:
    - [Ph.D. thesis][rockiThesis] of Kamil Rocki
    - [Parallel Monte-Carlo Tree Search][parmcts] by G Chaslot, et al.
    - [Parallel Monte Carlo Tree Search on GPU][rockipaper] by K Rocki and R Suda
- A survey of techniques related to Monte Carlo Tree Search is in:
  - [A Survey of Monte Carlo Tree Search Methods][mcsurvey]
