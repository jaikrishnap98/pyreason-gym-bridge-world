# pyreason-gym-bridge-world
# PyReason Gym for Bridge World


## Table of Contents
  
* [Getting Started](#getting-started)
    * [The Setting](#the-setting)
    * [The Actions](#the-actions)
    * [The Objective](#the-objective)
    * [Rewards](#rewards)
* [Installation](#installation)
* [Usage](#usage)


## Getting Started
This is an OpenAI Gym environment for reinforcement learning in a block world setting using [PyReason](https://github.com/lab-v2/pyreason) as a default simulator but there is an option 
to use simulator other than PyReason.

### The Setting
1. There are 5 blocks of certain color, shape and cost kept on a table. 
2. Possible colors: [Red, Green, Blue], Possible shapes: [vertical, horizontal]
3. Cost of block is dependent on the color -> {Red: 2, Green: 4, Blue: 6}
4. RL policy is independent of the cost.
5. Robot arm will work with these blocks.

### The Actions
There are 3 types of actions taken step by step by the simulator:

#### 1. Reset: 
Restarts the simulator by bringing it to a state where environment is not aware of the properties of 5 blocks.

#### 2. Initalize environemnt
Environment is provided with properties of the available 5 blocks i.e. color, shape.



### The Objective
The objecive of the game is to build a legal bridge structure using the blocks on the table in such a way that it satisfies certain constraints:
1. There are 2 base slots: [h1], [h2]
2. There is 1 top slots:  [h3]
3. Base slots can be only filled by vertical blocks.
4. Top slot can be only filled by horizontal block.
7. Green block cannot touch blue block, i.e. h3 touches h1 and h2. h1 and h2 does not touch each other.
### Rewards
A reward is calculated as follows:

1. If block is placed in a legal slot, then reward of `+1` is given.
2. If block is placed in a illegal slot, then reward of `-2` is given.
4. If house structure is built completely then reward of `+10` is given.
5. If house structure is incomplete even if number of available blocks is less than number of remaining slots, then reward of `-10` is given.
## Installation
Make sure  latest version `pyreason==2.0.1` has been installed using the instructions found [here](https://github.com/lab-v2/pyreason#21-install-as-a-python-library)

Clone the repository, and install:
```bash
git clone https://github.com/jaikrishnap98/pyreason-gym-bridge-world
```
## Usage
To run the environment and get a feel for things you can run the [`test_legal_bridge.py`](./test_legal_bridge.py) file which will perform actions in the bridge world. 
You can change the actions in the script according to your requirement. 

If you want to use PyReason as a simulator, set macro `PYREASON_SIMULATOR`= True otherwise set it to False and provide your own implementation in the [`other_simulator.py`](./pyreason_gym/simulator_other/other_simulator.py)
```bash
python test_legal_bridge.py

This Bridge World scenario needs a graph in GraphML format to run. A graph file has **already been generated** in the [graphs folder](pyreason_gym/pyreason_bridge_world/graph/bridge_world_graph.graphml/). 

This is an OpenAI Gym custom environment. More on OpenAI Gym:

1. [Documentation](https://www.gymlibrary.dev/)
2. [GitHub Repo](https://github.com/openai/gym)

The interface is just like a normal Gym environment. To create an environment and start using it, insert the following into your Python script. Make sure you've [Installed](#installation) this package before this.

A Tutorial on how to interact with gym environments can be found [here](https://www.gymlibrary.dev/)

