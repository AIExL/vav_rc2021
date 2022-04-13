"""
    Following codebase is the re-implementation to conduct experiments for 
    the paper titled 'Value Alignment Verification (http://proceedings.mlr.press/v139/brown21a/brown21a.pdf)'
    The cases of (explicit human, explicit robot) and (explicit human implicit robot) are implemented
"""

# Import the required libraries
import os
import sys
import copy
import math
import scipy
import pickle
import random
import argparse
import numpy as np
from gridworld import GridWorld
from tester import RewardWeightTester, RewardTester, ValueFunctionTester, PreferenceTrajectoryTester, CriticalStatesTester, SetCoverOptimalTeachingTester, ARPBBTester

import warnings
warnings.filterwarnings("ignore")

# Function to generate new weights for a robot by
# adding gaussian noise to the existing weights
def add_gaussian_noise(weights, mean = 0, sigma = 1):

    s = np.random.normal(mean, sigma, (weights.shape[0], weights.shape[1]))
    weights = weights + s
    weights /= np.linalg.norm(weights)
    return np.round(weights, decimals=4)

# Function to specify the human weights to a random vector
def get_human_weights(size, mean = 0, sigma = 1):

    weights = np.random.normal(mean, sigma, size)
    weights /= np.linalg.norm(weights)
    return np.round(weights, decimals=4)

# Check alignment of a robot with human
# A robot is aligned with human if the actions of robot 
# at all states is optimal under the policy of the human
def check_oracle_alignment(human, robot):
    
    for state, actions in list(human.policy.items()):
        robot_actions = set(robot.policy[state])
        actions_set = set(actions)

        if not robot_actions.issubset(actions_set):
            return False
    
    return True

# Function to get a class instance given the name of the class as string
def str_to_class(str):
    return getattr(sys.modules[__name__], str)

# Main function: Assemble all function calls
def main(args):

    if args['visualize']:
        # store the accuracy, false positive rate and false negative rate for visualization
        acc = []
        fpr = []
        fnr = []
        
        # create a directory to store the results
        store_dir = os.path.join('experiments', args['tester'], f"gw_{args['rows']}_{args['cols']}",
                                    f"actions_{args['num_actions']}", f"features_{args['size']}", f"type_{args['feature_type']}") 
        # experiments/rwt/gw_{row}_{col}/actions_{#actions}/features_{#features}/type_{feature_type}/{acc, fpr, fnr}.pkl, metadata.json
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)


    print("\n\n############################ VALUE ALIGNMENT VERIFICATION ############################\n\n")
    # Tester class mappings to store tester class names as strings
    tester_class_mappings = {"rwt": "RewardWeightTester", "rt": "RewardTester", "vft": "ValueFunctionTester",
                            "ptt": "PreferenceTrajectoryTester", "cst": "CriticalStatesTester", 
                            "scott": "SetCoverOptimalTeachingTester", "arpbbt": "ARPBBTester"}
    # Iterate over the number of humans
    for i in range(args['num_humans']):

        # Variable to track the accuracy
        # Increased if the tester verdict matches the oracle alignment
        accuracy = 0

        # Generate the human weights
        human_weights = get_human_weights((args['size'], 1))

        # Create the gridworld associated with the human
        human = GridWorld((args['rows'], args['cols']), args['size'], human_weights, gw_features = None, 
                            gw_diagonal = False, gw_actions = args['num_actions'], gw_terminal_state = None, gw_feature_type = args['feature_type'])

        # Obtain the value function and policy associated with the human by running value iteration
        human.value_function, human.policy = human.value_iteration()

        tester = str_to_class(tester_class_mappings[args['tester']])(human) # create the tester
        if(args['tester'] != 'cst'):
            tester.create_ARPMatrix() # create the ARP Matrix
        
        # Initialise the number of non-ambiguous robots as the total number of robots for each human
        non_ambiguous_robots = args['num_robots']

        # Iterate over the number of robots
        for j in range(args['num_robots']):

            # Get the robot weights by adding gaussian noise to human weights
            robot_weights =  add_gaussian_noise(human_weights) 

            # Create the robot gridworld
            # NOTE: The features and terminal state for the robot gridworld are the same as that of the human
            robot = GridWorld((args['rows'], args['cols']), args['size'], robot_weights, gw_features = human.state_feature_matrix, 
                        gw_diagonal = False, gw_actions = args['num_actions'], gw_terminal_state = human.terminal_state, gw_feature_type = args['feature_type'])
            
            # Obtain the value function and policy associated with the robot by running value iteration
            robot.value_function, robot.policy = robot.value_iteration()

            # If the tester is Reward Tester or Value Function Tester
            # Definer a temporary robot
            if args['tester'] == 'rt' or args['tester'] == 'vft':
                tester.define_temp_robot(robot)
                robot = tester.robot_temp
                
            # Check alignment of the robot with the human to get the tester verdict
            tester.check_alignment(robot)
            num_queries = tester.num_queries

            # Obtain the oracle verdict
            oracle_verdict = check_oracle_alignment(human, robot)

            # The following code segment updates the accuracy and 
            # segregates the cases of false positives and negatives
            if oracle_verdict:
                if tester.aligned:
                    print(f"============== Tester Verdict: {tester.verdict}")
                    accuracy += 1
                    if args['visualize']:
                        # append 1 -> acc, 0 -> fpr, 0 -> fnr
                        acc.append(1)
                        fpr.append(0)
                        fnr.append(0)
                    
                else:
                    print(f"============== (False Negative) Oracle Verdict: {oracle_verdict}")
                    print(f"============== (False Negative) Tester Verdict: {tester.verdict}")
                    if args['visualize']:
                        # append 0 -> acc, 0 -> fpr, 1 -> fnr
                        acc.append(0)
                        fpr.append(0)
                        fnr.append(1)
                    
            else:
                if not tester.aligned:
                    print(f"============== Tester Verdict: {tester.verdict}")
                    accuracy += 1
                    if args['visualize']:
                        # append 1 -> acc, 0 -> fpr, 0 -> fnr
                        acc.append(1)
                        fpr.append(0)
                        fnr.append(0)

                else:
                    print(f"============== (False Positive) Oracle Verdict: {oracle_verdict}")
                    print(f"============== (False Positive) Tester Verdict: {tester.verdict}")
                    if args['visualize']:
                        # append 0 -> acc, 1 -> fpr, 0 -> fnr
                        acc.append(0)
                        fpr.append(1)
                        fnr.append(0)
                    # NOTE: uncomment the following line to stop the execution if FP arises (but it is most likely to arise)
                    # assert args['tester'] not in ['rwt', 'rt', 'vft'], "Exact Value Alignment Verification: This should not happen"

        # If number of non-ambiguous robots is greater than zero, compute the final accuracy, else continue
        if non_ambiguous_robots > 0:
            print(f"\n\nHuman {i + 1} Num robots {args['num_robots']}: Verifier ({args['tester']}) Accuracy {accuracy/non_ambiguous_robots}\n\n")
        else:
            print(f"\n\nHuman {i + 1} Num robots {args['num_robots']}: Verifier ({args['tester']}) Accuracy Not Applicable! \n\n")

    if args['visualize']:
        # store the acc, fpr, fnr lists
        acc = np.array(acc)
        fpr = np.array(fpr)
        fnr = np.array(fnr)

        pickle.dump(acc, open(os.path.join(store_dir, 'acc.pkl'), 'wb'))
        pickle.dump(fpr, open(os.path.join(store_dir, 'fpr.pkl'), 'wb'))
        pickle.dump(fnr, open(os.path.join(store_dir, 'fnr.pkl'), 'wb'))

    args['num_queries'] = num_queries
    # dump the arguments, which includes the num_queries for this specific scenario
    if args['visualize']:
        pickle.dump(args, open(os.path.join(store_dir, 'args.pkl'), 'wb'))

if __name__ == "__main__":

    """ Standard argument parsing
            cols: Width of the gridworld
            feature_type: Type of reward, weight and state-feature mapping
            num_actions: Number of allowed actions in the action space
            num_humans: Number of human agents to be created
            num_robots: Number of robots for each human agent
            rows: Number of rows in the gridworld
            size: Size of the weight vector
            tester: Type of tester to be used
            visualize: indicator to store the results and specifications for the gridworld
    """

    ap = argparse.ArgumentParser()
    ap.add_argument('--cols', default = 6, type = int)
    ap.add_argument('--feature_type', default = "linear", type = str, choices = ["linear", "cubic", "exponential"])
    ap.add_argument('--num_actions', default = 4, type = int)
    ap.add_argument('--num_humans', default = 100, type = int)
    ap.add_argument('--num_robots', default = 100, type = int)
    ap.add_argument('--rows', default = 4, type = int)
    ap.add_argument('--size', default = 3, type = int)
    ap.add_argument('--tester', default = 'ptt', type = str, choices=['rwt', 'rt', 'vft', 'ptt', 'cst', 'scott', 'arpbbt'])
    ap.add_argument('--visualize', default = True, type = bool)
    args = vars(ap.parse_args())

    main(args)