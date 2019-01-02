![trained agent](trained_agent.gif)



## Introduction
This project uses a Deep Deterministic Policy Gradient (DDPG) method to train an agent to control a double-jointed arm to move to a target location and stay within the target location. The target location continuously moves. The environment is a pre-built variant of the [Unity Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. This project is being done as part of the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).



## Environment
This environment is a pre-built version of the [Unity Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) example one. It is therefore not necessary to install Unity itself. There is a choice of two different environments actually:

- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.

We've focused on solving the 20 agent case, although both versions can use our code to train with (possibly with adjustment to hyper-parameters).

We describe the single agent case, just for the sake of easy explanation, but it applies to the multi(20)- agent version in the same way. The 20 agent version is really just observing 20 of the single agent versions simultaneously - allowing more observations quicker, which in turn translates to quicker learning.

We observe a double-jointed arm, which can be controlled at both joints. The arm is fixed at the shoulder joint, extends to the elbow joint and ends in the hand. Control is enabled by applying torque (in 2 perpendicular directions) at each joint - so 4 torque values in total, with each torque value standardised to be in the range [-1, 1]. This allows unrestricted movements, except that an arm cannot move onto and through itself.

A 3D target locataion in the shape of a sphere, moves around the fixed shoulder joint. We want to have the agent's hand within this target location for as many of the fixed 1001 timesteps as possible.

#### Goal
The goal is to have the agent's hand within this target location for as many of the timesteps as possible. The hand does not start within the target location and needs to move towards it by applying the needed combination of torques and to then keep it there as long as possible.

The task is episodic, and in order to solve the environment, we must get an average score of +30 per episode per agent over 100 consecutive episodes. Each timestep that the hand is within the target location, a score of +0.04 is awarded. Given that there are 1001 timesteps per episode, this implies that solving the environment requires the hand to be in the target location for very close to 3/4 of the timesteps.

#### States
The state space has 33 dimensions per agent corresponding to position, rotation, velocity, and angular velocities of the arm as well as the position of the target location. Given this information, the agent has to learn how to best select actions.

#### Actions
Four continuous actions are available, corresponding to torque applied at the joints. 2 torque values for each of the 2 joints, so 4 torque values in total. Each torque value is standardized and has to be in the rage [-1, 1].

#### Rewards
A reward of +0.04 is provided per timestep that the hand is within the target location sphere. This is the only reward observed. The project instruction states that a +0.1 reward is given per timestep, which is definitely not what is observed.



## Installation

#### Operating system and context
The code was only run in Ubuntu 18.04. It may be possible to get it working on other operating system, but this is untested.

The project came with Jupyter Notebook files, but I've decided not to use these are I am more comfortable with PyCharm and have therefore decided to do the project in PyCharm. Therefore, all files with python code will be `.py` files.

The recommended method to work with the project is to run the main.py from the command line (in a terminal) with the intended parameters. More details on this below.

#### Pre-requisites
Make sure you having a working version of [Miniconda](https://conda.io/miniconda.html) (the one I used) or [Anaconda](https://www.anaconda.com/download/) (should also work, but untested) on your system.

#### Step 1: Clone the repo
Clone this repo using `git clone https://github.com/ErnstDinkelmann/udacity_deeprl_reacher_continious.git`.

#### Step 2: Install Dependencies
Create an anaconda environment that contains all the required dependencies to run the project.

Linux:
```
conda create --name drlnd python=3.6
source activate drlnd
conda install -y pytorch -c pytorch
pip install unityagents
```

#### Step 3: Download Reacher environment
You will also need to install the pre-built Unity environment, but you will NOT need to install Unity itself (this is really important, as setting up Unity itself is a lot more involved than what is required for this project).

Select the appropriate compressed file for your operating system from the choices below.

Single agent version:

- Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

Multi-agent version:

- Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Download the file into the top level directory of this repo and extract it. The code as it is assumes that the environment file (`Reacher.x86` for 32-bit systems or `Reacher.x86_64` for 64-bit systems) is located with a directory in the root of the repo (`/Reacher_Multi/` for the multi-agent version of the environment or `/Reacher_Single/` for the single-agent version of the environment).



## Files in the repo
This is a short description of the files in the repo (that comes with the repo) or may be generated as output by the code when running:

* `main.py`: the main file containing high-level training function as well as a function for viewing a trained agent. Note that the directory location of this file is detected and used as a parameter in the code. If you are not executing from the command line, you may need to make adjustments to the code. This is the file that will be run from the command line as follows:

    * To train the agent: `python main.py --mode train` or, since training is the default behaviour, `python main.py` will also work just fine.
    * To view a trained agent: `python main.py --mode view`. You will need a `actor_checkpoint.pth` file in the root for this to work.

* `parameters.py`: the file with all the hyper-parameters that we chose to expose (essentially the ones we felt were important during some stage of the development and understanding). Actual parameters are merely global python constants.

* `agent.py`: contains the DdpgAgent class, which controls our agents interaction and learning with/from the environment. This agent specifically learns the mapping of states -> actions with the help of a critic that learns states + actions -> action value. These mappings are learnt as neural networks specified in the networks.py file.

* `networks.py`: contains the nueral networks set up in pytorch. There are two classes specified: the ActorNetwork and the CriticNetwork, both of which are used by the agent for learning.

* `replay_buffer.py`: contains the ReplayBuffer class, which serves as the memory of the agent. It is a fixed length list that stores experiences of the agent within the environment. The agent then samples from these past experiences to learn from, as opposed to learning directly from the environment as it's experienced.

* `noise.py`: contains two classes which generates noise, only one of which is used, chosen by a parameter. They work differently and details are within the file. The noise is added to the torque values (actions) so that we explore the environment.

* `actor_checkpoint.pth` and `critic_checkpoint.pth`: contains the saved weights of the specified neural network, for an already trained agent right after it achieved the requisite performance to "solve" the environment. There are also _number versions where the number is the episode at which the save was done - at episode 100, 200, 300, etc. There are also _tmp versions that are saved after each episode, but overwritten.



## Train the agent
To train the agent: `python main.py --mode train` or, since training is the default behaviour, `python main.py` will also work just fine.

This will start the Unity environment and output live training statistics to the command line. Training continues for up to 1001 episodes. However, with the hyper-parameters as they are set in the repo, the required average is achieved before 100 episodes are done. It will continue to save checkpoint files with the weights of the networks at episode 100, 200, 300, etc thereafter. You may interrupt the code at any time.

Feel free to experiment with modifying the hyper-parameters to see how it affects training.

I found DDPG (at least in the form we've created it) extremely sensitive to the hyper-parameters and very unstable. Making a slight change to one the hyper-parameters, could suddenly cause erratic behaviour and also unable to train at all. This was very disappointed, but it may just be that not enough stabalising enhancements had been introduced or that the DDPG is not really suitable for this specific task. More specific details will be provided in the Report file, linked below.



## View the trained agent perform
To view a trained agent: `python main.py --mode view`.
This will load the saved weights from a checkpoint file (`actor_checkpoint.pth`).  A previously trained model is already included in this repo.



## Report
See the [report](Report.md) for more insight on the technical modelling aspects of the project and how we arrived at a solution.
