# Reinforcement learning module using Monte Carlo AIXI CTW node.

This node is able to do reinforcement learning in wide variety of environments, including  POMDP.

It will work in *worst case scenario*:
* unknown environment
* noisy observations
* stochastic effects of actions
* aliasing of perceptions
* no explicit notion of state
* rewards only sparsely distributed

...but it will takes quite long time.

[kt]: https://en.wikipedia.org/wiki/Krichevsky%E2%80%93Trofimov_estimator
[AIXI]: http://www.hutter1.net/ai/aixigentle.pdf
[pUCT]: http://jveness.info/publications/nips2010%20-%20pomcp.pdf
[UCT]: http://www.sztaki.hu/~szcsaba/papers/ecml06.pdf

## Short introduction to algorithm

From point-of-view of algorithm, the world looks like a sequence of bits, in which are encoded past rewards, observations and actions. Algorithm has set from beginning what is number of bits representing observations/rewards/actions. It interact with environment by giving bits that encodes chosen action and recieve bits that encodes observation and reward.

![interaction](http://i.imgur.com/vxavfqf.jpg)

This algorithm tries to approximate general [AIXI], which is formalism showing (incomputable) notion of what is an optimal action for agent in unknown environment.

Internal model of environment for original AIXI is probability distribution over all possible programs that could represent environment, where apriori probability of one possibility (ie: one particular program) depends on its complexity.

Internal model of environment MC-AIXI-CTW is probability distribution over all predictive suffix trees (PSTs). Where apriori probability of one possibility (ie: one particular PST) depends on its complexity. This is done by context-tree weighting (CTW)
algorithm.

### PSTs and CTW
Bellow is example of one simple PST:

![Example of PST](http://i.imgur.com/jNDnV3p.png)

When we want to use this PST to predict probability of next bit in interaction (eg: what will be first bit of our observation in next round?), when we saw sequence (for example) 01000*10* so far, it can be done this way:

- We start in the root and look at last bit we saw: it was 0. We therefore go to child under edge with label "0". 
- We are not in leaf, so far, so we continue in same way. second-to last bit was 1, so we end up in node  θ <sub>01</sub>.
- When we arive in leaf, we find there probability of next bit being 1, in this case we predict that next bit will be 1 with probability 0.3.

CTW part of MC-AIXI-CTW uses context tree weighting to compute mixture of predictions of all PST of certain depth, where each has weight corresponding to its complexity. Thetas of each PST are computed by [KT-estimator][kt]. Importance of this algorithm is that it is able to compute mixture of predictions of all 2<sup>D</sup> possible models, which would naively took 2<sup>2<sup>D</sup></sup> steps, in O(D).

CTW was originaly used for compression, but reusing it for prediction is natural since these two problems are closely related. When we are compressing, we can forget everything we can predict. Therefore, having a good predictor means having a good compressor. Original AIXI achieves best action possible, by using best compressor possible: Kolmogorov complexity.

### Monte Carlo
So far, we have rough idea about how to model (ie: being able predict reactions of) environment. When we have this model, it is necessary to plan what action to take.  Most well-known algorithms for this (eg: α-β-pruning) are not usable because:

- we do not have heuristic estimation of quality (goodness/badness) of a state. Our algorithm runs in unknown environment.
- There is no explicit "state" - state of environment is hidden, algorithm see just some output. From point of view of MC-AIXI-CTW state we are in is whole history of interaction.

To solve this, Monte Carlo Tree Search (MCTS) algorithm, widely used in Go playing bots, is used. When we consider what to do in some state, instead of considering possible next states from "state space" like in minimax-like algorithms, we try to play several games starting in our current state against our model of environment while choosing our actions randomly.

How MCTS works:
![pUCT](http://i.imgur.com/1h3Funl.png)

This way, algorithm can make a tree model, like on image above as to what situation will various actions lead to, and compute average reward of them. To pick what action to take it uses [ρUCT] (modified [UCT]) to keep balance between exploration and exploitation.



### Parallelisation
Parallelisation of this algorithm is not so straightforward since it depends on searching in trees, instead of matrix operations, etc. Basic ideas for parallelisation of general MCTS techniques were published by [Rocki][rockipaper] and [Chaslot][parmcts]. But published research in this area is rather scarse.

I decided to make algorithm paralel in two ways at once. First way is that, when we simulate game against environment (wigly lines at bottom of image bellow), we simulate many games at once in parallel, their results are then averaged.

Other way is that instead of having one BCTS tree of possible future, we make several trees that are merged at end of cycle.

![Parallelisation](http://i.imgur.com/T2AqbzE.jpg)

# Usage
[TODO]

# Performance
[TODO]

[jvPhD]: http://jveness.info/publications/veness_phd_thesis_final.pdf
[mcaixictw]: https://www.jair.org/media/3125/live-3125-5397-jair.pdf
[rockiThesis]: http://olab.is.s.u-tokyo.ac.jp/~kamil.rocki/phd_thesis.pdf
[parmcts]: https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf
[rockipaper]: http://olab.is.s.u-tokyo.ac.jp/~kamil.rocki/rocki_scai11.pdf
[mcsurvey]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6145622&tag=1


## Literature
- Introduction to MC-AIXI-CTW is in PhD thesis of [Joel Veness][jvPhd].
- Main paper on this is [A Monte-Carlo AIXI Approximation][mcaixictw] by J Veness et al.
- Some ideas on parallelisation of monte-carlo tree search are in:
    - [PhD thesis][rockiThesis] of Kamil Rocki
    - [Parallel Monte-Carlo Tree Search][parmcts] by G Chaslot, et al.
    - [Parallel Monte Carlo Tree Search on GPU][rockipaper] by K Rocki and R Suda
- A survey of techniques related to Monte Carlo Tree Search is in:
  - [A Survey of Monte Carlo Tree Search Methods][mcsurvey]

