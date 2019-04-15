# Learning Reinforcement Learning:
Algorithms from DQN to Rainbow, GAE, DDPG with unified interface.

* **Theory overview**: [Theory Overview](https://github.com/FortsAndMills/Learning-Reinforcement-Learning/tree/master/Theory%20Overview) - conspect of articles about implemented algorithms.
* **LRL Demo**: [LRL demo.ipynb](https://github.com/FortsAndMills/Learning-Reinforcement-Learning/blob/master/LearningRL%20-%20Demo.ipynb) Library interface guide.
* **LRL Cartpole**: [LRL Playing with cartpole.ipynb](https://github.com/FortsAndMills/Learning-Reinforcement-Learning/blob/master/LRL%20Playing%20with%20cartpole.ipynb) - launching a bunch of algorithms on the simplest environment Cartpole.

## Requirements:
* gym
* PyTorch
* matplotlib
* pickle
* JSAnimation

## Sources of code and inspiration:
* [RL Adventure](https://github.com/higgsfield/RL-Adventure), [RL Adventure pt.2](https://github.com/higgsfield/RL-Adventure-2) - beware of bugs!
* [DeepRL Tutorials](https://github.com/qfettes/DeepRL-Tutorials)

# UPDATES
specially for my beloved scientific advisor:
* **23/09/18** Added detailed review of basic policy gradient algorithms: baselines for REINFORCE, Actor-Critic, Advantage Actor-Critic. Minor updates on rubik's cube solution attempts.
* **03/10/18** Several more HER experiments failed :( HER code for Rubik cleared, bug fixes;
* **24/10/18** Finally achieved first results with multiprocessing policy gradient algorithm! First attempt to incorporate policy gradient interface into library + support for multiprocessing environments. A lot of work to do yet...
* **02/11/18** Theory update: DDPG, GAE, HER, bugfixes
* **08/11/18** Code bugfixes; interface polished, A2C and Rainbow both working.
* **13/11/18** Rainbow turned out to be working very slow due to some poor coding. Fixed; multiprocessed DQN tested (without achieving performance improvement :( ). Also rainbow works 3x slower than vanilla DQN, that's not good, but corresponds with other sources.
* **13/11/18** [Ptan accelerations](https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55) (except #3) were tested. Acceleration #4 was "to use newest environment wrappers from deepmind", but turned out they were already used in my code.

| | My current code| Launching PTAN's code | PTAN's promises |
| ------------ | ------------ | ------------- | ------------- |
| Initial implementation (classic DQN) | ? | ~54 FPS | ~154 FPS |
| With "new wrappers" (improvement 4) | ~56 FPS | ? | ~182 FPS |
| Improvement 4+1 (larger batch_size and steps) | ~120 FPS | ? | ~268 FPS |
| Improvement 4+2 (async playing and optimizing) | ? | ? | ~316 FPS |
| Improvement 4+3 (async cuda transfer) | ~60 FPS | ? | ~188 FPS |
| All 4 improvements | ? | ? | ~484 FPS |

* **22/11/18** Found a simple way of slightly accelerating double DQN and A2C by simply reducing amount of forward passes through the network. Haven't seen this trick in other implementations, funny. Uploaded the optimized code.
* **22/11/18** Experiment: backward experience replay. Works with Categorical DQN (which corresponds to theory), but doesn't provide any significant advancement to vanilla Categorical DQN. Added Backwards Replay to the library.
* **22/11/18** Implemented: GAE.
* **28/11/18** Epic fail of the week: [this wonder](https://github.com/Unity-Technologies/ml-agents) failed to launch due to some TCP-sockets conflict... Days of fightings wasted :( :{ :\[
* **29/11/18** Implemented: DDPG, continuous control support, Factorized Gaussian policy for continuous PG algorithms. Library code revisited, fat bug in Categorical+Prioritized Replay catched!
* **01/12/18** Theory update: QR-DQN.
* **04/12/18** Implemented: QR-DQN.
* **16/12/18** Some ideas tested (code in "Experimental" folder), nothing works as always.
* **24/02/19** Back to the game: theory on TRPO and underlying natural gradients foundations.
* **26/02/19** Attempt to rewrite the library in different paradigm failed... :/ Code revision. Implemented: Twin DQN.
* **27/02/19** Theory update: PPO (short version: throw all TRPO theory away)
* **09/03/19** Implemented: PPO (sneaky bug was finally detected!). Also uploaded code for Quantile Regression Actor-Critic, which is an experiment on combining improvements of DQN with critic in policy gradient methods.
* **28/03/19** [Draft of theoretical part of course work](https://github.com/FortsAndMills/Learning-Reinforcement-Learning/blob/master/Modern_DRL_Algorithms.pdf) is written. 
* **15/04/19** TRPO implementation does not work :o(. Prioritized experience replay rewritten with SumTree structure for substantial acceleration. NoisyNetwork has initialisation issues; switched implementations. Minor changes in code structure and refactoring.
* **15/04/19** Uploaded "LRL Pong.ipynb" with launches of all compared algorithms (with 1 000 000 samples limitation): vanilla DQN, c51 (categorical DQN), quantile regression (QR-DQN), rainbow, A2C (with GAE), PPO. 

### PLANS:
* Find out how to properly fix seeds for experiments (setting seeds in numpy, torch, torch.cuda and in environments... didn't help! Even when no asynchronity is used!..)
* Check other environments.
* Continue theory exploration (minimal program: great list of articles by https://blog.openai.com/spinning-up-in-deep-rl/ ... Still waiting for me: DQRN, ACKTR, ACER, SAC, TD3, REACTOR, APE-X, R2D2)
* Continue implementing algorithms to the library (next: TRPO?)
* Check code from other sources (Yandex Practical RL, OpenAI baselines, ...).

