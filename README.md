# [[Re]: Value Alignment Verification](https://openreview.net/forum?id=BFLM3nMmhCt)

The following codebase is an attempt to re-implement the ideas proposed in the paper titled [Value Alignment Verification](http://proceedings.mlr.press/v139/brown21a/brown21a.pdf) and conduct experiments to verify the claims with respect to the alignment of a robot with a human agent. 

## Citation
For citing this work use:  
```
@inproceedings{panigrahi2022re,
  title           = {[Re]: Value Alignment Verification},
  author          = {Siba Smarak Panigrahi and Sohan Patnaik},
  booktitle       = {ML Reproducibility Challenge 2021 (Fall Edition)},
  year            = 2022,
  url             = {https://openreview.net/forum?id=BFLM3nMmhCt}
}
```

### Report Description
It is critical to be able to efficiently evaluate an agent's performance and correctness when humans engage with autonomous agents to execute increasingly complex, sometimes dangerous activities. The paper formulates and conceptually investigates the problem of efficient value alignment verification: how to test whether another agent's behaviour is aligned with a human's values in an efficient manner. The goal is to create a _driver's test_ that a human may give to any agent to check value alignment with the least number of queries possible. In a wide range of gridworlds, we investigate the verification of exact value alignment for rational agents and verify claims made on the heuristics. We investigate and reproduce the claims with respect to _explicit human, explicit robot_ and _explicit human, implicit robot_ settings. 

### File Description

This repository contains a directory ```vav-reproduce```. The following files are present in the directory.
- [environment.py](./vav-reproduce/environment.py): Script to create a discrete environment for _explicit human, explicit robot_ and _explicit human, implicit robot_ setting. The environment has the following functionalities.
  - Run value iteration
  - Print the optimal policy
  - Obtain the next state given an action
  - Obtain the state-value and value function
- [gridworld.py](./vav-reproduce/gridworld.py): Script to create a gridworld of dicrete environment type. The functionalities of the gridworld are as follows.
  - Design features for each state of the gridworld
  - Get the reward for each state in the gridworld
  - Check ambiguity of the gridworld
- [tester.py](./vav-reproduce/tester.py): Script to create the testers for the following algorithms and heuristics.
  - Reward weight queries
  - Reward queries
  - Value function queries
  - Trajectory preference queries
  - Critical states heuristic
  - Machine teaching - Set cover optimal teaching heuristic
  - Aligned reward polytope - black box heuristic
- [main.py](./vav-reproduce/main.py): Script to create the human and robot agent, initlialise their rewards, policies and check for alignment of the robot agent with respect to the human agent.

### Installation

Follow the steps mentioned below: 
```
  conda env create -f vavrc21.yml
  conda activate vavrc21
```

### Experiments

To run the code with default arguments: ```python ./main.py``` 

To run the code with user-defined arguments: ```python ./main.py --cols 8 --feature_type linear --num_actions 4 --num_humans 100 --num_robots 100 --rows 4 --size 5 --tester rwt --visualise False```

We carry out the following experiments. 
- **Algorithms and Heuristics**: In different gridworlds, different algorithms and heuristics are compared. With varied gridworld widths ranging from 4 to 8 and feature sizes ranging from 3 to 8, we tabulate tester performance (accuracy, false positive rate, false negative rate, and number of queries provided to the robot for verification). The number of features or feature size is the dimension of a feature for a state. These state features are only one-hot vectors in our tests.
- **Diagonal Actions**: In gridworlds with an extended action space, several algorithms and heuristics are compared. Between conventional movements, we offer diagonal movement. The normal four actions (left, up, right, and down) are now increased to eight (left-up-diagonal, up-right-diagonal, right-down-diagonal, and down-left-diagonal). We tabulate the results of testers for various gridworld widths once more.
- **Non-linear reward and state-feature relationships**: Comparison of non-linear (_cubic_ and _exponential_) reward R and state-feature relationships using various algorithms and heuristics. When w<sup>T</sup>&#981;(s) for state s is approximately equal to zero, we approach the linear behaviour in _cubic_; otherwise, we do not. In _exponential_, we ignore the linear relationship between R and state feature. In all situations, we tabulate tester performance in relation to various gridworld widths.
- **Critical States Tester for different thresholds**: Comparison of Critical States Tester performance with different threshold values (0.8, 0.2 and 0.0001).

### Acknowledgement

We contacted the authors of the original paper several times to get clarifications and further information about the implementation. The authors were quite responsive to our questions, and we appreciate their detailed and timely responses.
