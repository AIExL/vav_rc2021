"""
    Following codebase is the re-implementation to conduct experiments for 
    the paper titled 'Value Alignment Verification (http://proceedings.mlr.press/v139/brown21a/brown21a.pdf)'
    The cases of (explicit human, explicit robot) and (explicit human implicit robot) are implemented
"""

# Import the required libraries
import copy
import scipy
import random
import numpy as np 
import math, os, sys
from environment import DiscreteEnvironment 

# Class to create a gridworld of discrete environment type (inherits DiscreteEnvironment)
class GridWorld(DiscreteEnvironment):

	def __init__(self, 
				gw_size,
				gw_num_states,
				gw_weights,
				gw_terminal_state = None,
				gw_features = None,
				gw_diagonal = False,
				gw_actions = 4,
				gw_policy = None,
				gw_dtype = np.float32,
				gw_feature_type = "linear"):
		super(GridWorld, self).__init__()

		"""
			gw_size: Size of the weight vector
			gw_num_states: Number of unique states in the gridworld
			gw_weights: Weight vector associated with the gridworld
			gw_terminal_state: Terminal state for the gridworld
			gw_features: State feature matrix for gridworld
			gw_diagonal: True if diagonal action in the gridworld is possible
			gw_actions: Set of possible actions 
			gw_policy: Policy for an agent working in the gridworld
			gw_dtype: Data type used
		"""

		self.size = gw_size
		self.num_actions = gw_actions
		self.num_states = gw_num_states
		self.terminal_state = gw_terminal_state
		self.weights = gw_weights
		self.features = gw_features
		self.dtype = gw_dtype
		self.policy = gw_policy
		self.diagonal = gw_diagonal
		self.value_function = None
		self.Q_value_matrix = None
		self.value_function_ns = None
		self.feature_type = gw_feature_type
		
		# Design the state_feature_matrix 
		# depending upon presence of terminal_state, change its feature vector
		self.state_feature_matrix = np.zeros((self.size[0], self.size[1], self.num_states), dtype = self.dtype)

		# action to vec contains the action in clockwise order - starting from left movement
		if self.num_actions == 8:
			# Diagonal movements allowed
			self.action_to_vec = np.array([[0, -1], [-1, -1], [-1, 0], 
											[-1, 1], [0, 1], [1, 1],
											[1, 0], [1, -1]], dtype = self.dtype)
			self.action_to_text = {0: '(L)', 1: '(LUD)', 2:'(U)', 3:'(RUD)', 4:'(R)',
									5:'(RDD)', 6:'(D)', 7:'(LDD)', 'None': '(T)'}

		else:
			self.action_to_vec = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]], dtype = self.dtype)
			self.action_to_text = {0:'(L)', 1:'(U)', 2:'(R)', 3:'(D)', 'None':'(T)'}

		self.design_features()
		self.reward()
	
	# Function to design the state feature matrix
	def design_features(self, feature_type = 'one-hot'):
		
		if self.features is not None:
			# Feature matrix is already available
			self.state_feature_matrix = copy.deepcopy(self.features)

		elif feature_type == 'one-hot':
			# one hot features for all states
			max_weight_index = np.argmax(self.weights) # terminal state should have the highest reward

			for i in range(self.size[0]):
				for j in range(self.size[1]):
					# Ensure the max_weight_index is not assigned to any state
					# This will be assigned later to the terminal state
					random_index = np.random.randint(0, self.num_states)
					if random_index == max_weight_index:
						random_index -= 1

					self.state_feature_matrix[i][j][random_index] = 1
		
			if self.terminal_state is not None:
				# Overwrite the pre-assigned terminal state feature vector
				self.state_feature_matrix[self.terminal_state[0]][self.terminal_state[1]] = np.zeros(self.num_states) 
				self.state_feature_matrix[self.terminal_state[0]][self.terminal_state[1]][max_weight_index] = 1
			else:
				# Randomly create a terminal state 
				row_index = random.randint(0, sys.maxsize) % self.size[0]
				col_index = random.randint(0, sys.maxsize) % self.size[1]
				self.state_feature_matrix[row_index][col_index] = np.zeros(self.num_states)
				self.state_feature_matrix[row_index][col_index][max_weight_index] = 1
				self.terminal_state = (row_index, col_index)
		
		elif feature_type == 'random':
			# Create a random 3D numpy array
			self.state_feature_matrix = np.random.random((self.size[0], self.size[1], self.num_states))
			self.state_feature_matrix = self.state_feature_matrix / np.sum(self.state_feature_matrix, axis = 2)

			# Randomly create a terminal state 
			row_index = random.randint(0, sys.maxsize) % self.size[0]
			col_index = random.randint(0, sys.maxsize) % self.size[1]
			self.state_feature_matrix[row_index][col_index] = np.zeros(self.num_states)
			self.state_feature_matrix[row_index][col_index][max_weight_index] = 1
			self.terminal_state = (row_index, col_index)

	def reward(self):
		# Return reward for all states 
		if self.feature_type == "linear":
			self.reward = np.dot(self.state_feature_matrix, self.weights)
		elif self.feature_type == "cubic":
			self.reward = np.dot(self.state_feature_matrix, self.weights) ** 3 \
							+ 10 * np.dot(self.state_feature_matrix, self.weights)
		elif self.feature_type == "exponential":
			self.reward = np.exp(np.dot(self.state_feature_matrix, self.weights))

	def get_reward(self, state):
		# Return reward for a particular state
		return self.reward[state[0]][state[1]]

	def get_unique_reward(self):
		# NOTE: Current implementation allows only one-hot feature type
		unique_reward = np.zeros((self.num_states, 1), dtype = self.dtype)
		unique_state_feature_matrix = np.array([], dtype = self.dtype).reshape(0, self.num_states)
		for i in range(self.num_states):
			state_feature = np.zeros((self.num_states, 1))
			state_feature[i] = 1
			unique_state_feature_matrix = np.concatenate((unique_state_feature_matrix, state_feature.reshape(1, -1)), axis = 0)
			if self.feature_type == "linear":
				unique_reward[i] = np.dot(state_feature.T, self.weights)
			elif self.feature_type == "cubic":
				unique_reward[i] = np.dot(state_feature.T, self.weights) ** 3 \
									+ 10 * np.dot(state_feature.T, self.weights)
			elif self.feature_type == "exponential":
				unique_reward[i] = np.exp(np.dot(state_feature.T, self.weights))
		
		return unique_reward, unique_state_feature_matrix

	# Function to check the ambiguity of an agent with respect to its policy (optional)
	def check_ambiguity(self):
		if self.policy is None:
			print('Please initialize a policy OR run value iteration!')
			return True
		
		for i in range(self.size[0]):
			for j in range(self.size[1]):
				s = (i,j)
				actions = self.policy[s]
				if s == self.terminal_state or len(actions) != 1:
					continue

				for act in actions:
					ns = self.next_state(s, act)
					ns_actions = self.policy[ns]
					for ns_act in ns_actions:
						temp_s = self.next_state(ns, ns_act)
						if temp_s == s and len(ns_actions) == 1:
							return True
		return False