# Learning Reinforcement Learning:
Algorithms from DQN to Rainbow with unified interface.

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

### PLANS:
* Find out how to properly fix seeds for experiments (setting seeds in numpy, torch, torch.cuda and in environments... didn't help! Even when no asynchronity is used!..)
* Check other environments.
* Continue theory exploration (minimal program: great list of articles by https://blog.openai.com/spinning-up-in-deep-rl/)
* Continue implementing algorithms to the library (next: DDPG?.. after all bugfixes, meh)
* Check code from other sources (Yandex Practical RL, OpenAI baselines, ...).

