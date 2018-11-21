# Udacity Deep Reinforcement Learning Nanodegree Project 1: Navigation

## Synopsis

This is my solution to the first project of the deep reinforcement learning nanodegree. The project goal is to write an agent that goes around collecting bananas. 

## Requirements

* python 3
* pytorch (Instructions: https://pytorch.org/)
* unityagent (Instructions: https://github.com/Unity-Technologies/ml-agents)
* Jupyter
* Numpy
* Matplotlib
* The banana world environment (Downloadable here https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)

## Environment 

The environment is square world filled with yellow and blue bananas. The goal is to pick up the yellow bananas and avoid the blue ones. The task is episodic and the end goal is to reach an average score of +13 over 100 consecutive episodes. 

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.


## How to run

1. Download the environment and unzip it into the directory of the project. Ensure that he Banana.exe is at the same level as the Repor.ipynb
2. Use jupyter to run the Report.ipynb notebook: `jupyter notebook Report.ipynb`
3. To train the agent run the cells in order. They will initialize the environment and train until it reaches the goal condition of +13
4. A graph of the scores during training will be displayed after training. 

## Code

There are two main files in the project, agent.py which implements the DQN agent. It initializes the target and local q-networks. The networks are located in the model.py file. It's a simple network composed of 3 fully connected layers and using rectefied linear units as the activation function. A third file, replay_buffer contains a implementation of a replay buffer that will contain experience tuples and allow sampling during training. 

