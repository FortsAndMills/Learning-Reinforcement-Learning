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

### PLANS:
* Find out how to properly fix seeds for experiments (setting seeds in numpy, torch, torch.cuda and in environments... didn't help!)
* Acceleration of Rainbow still possible.
* Multiprocessing environments support for value-based algorithms.
* Check other environments.
* Continue theory exploration (minimal program: R2D2, TRPO, PPO, Quantile Regression, DRQN, model-based stuff)
* Continue implementing algorithms to the library (next: DDPG?)
* Check code from other sources (Yandex Practical RL, OpenAI baselines)

