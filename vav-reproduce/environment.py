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

# Class to create Discrete Environment for (explicit human, explicit robot), (explicit human, implicit robot) setting
class DiscreteEnvironment():

	def __init__(self, env_tolerance = 1e-9, env_gamma = 0.9, env_dtype = np.float32):

		"""
			env_tolerance: Tolerance value while computing the optimal value function in value iteration
			env_gamma: Discount factor for value iteration
			env_dtype: Data type used
		"""

		self.tolerance = env_tolerance
		self.gamma = env_gamma
		self.dtype = env_dtype

	# Function to print the policy of an agent
	def print_policy(self, reverse=False):
		if self.policy is None:
			print('Please initialize a policy OR run value iteration!')
			return
		
		print("\nPrinting the Policy:\n")
		self.policy = dict(sorted(self.policy.items()))
		for state, actions in list(self.policy.items()):
			print_length = 0
			for a in actions:
				if a is None:
					print(self.action_to_text['None'], end = "")
					print_length += 3
					continue
				print(self.action_to_text[a], end="")
				print_length += len(self.action_to_text[a])
			if self.diagonal:
				print(" " * (24 - print_length), end="")
			else:
				print(" " * (15 - print_length), end="")

			if state[1] % self.size[1] == self.size[1] - 1:
				print("\n")

	# Function to obtain the next state given the current state and action
	def next_state(self, state, action):

		if action == None:
			return state

		next_state = tuple(np.array(state) + self.action_to_vec[action])		
		next_state = (int(next_state[0]), int(next_state[1]))	
			
		if next_state[0] < 0 or next_state[0] >= self.size[0] or next_state[1] < 0 or next_state[1] >= self.size[1]:
			return state
		return next_state 	  
	
	# Function for value iteration and obtaining the policy of an agent
	def value_iteration(self):
		value_function = np.random.rand(self.size[0], self.size[1])
		value_function[self.terminal_state[0]][self.terminal_state[1]] = 0
		policy = {} 

		# Value iteration algorithm
		while True:
			delta = 0
			for row in range(self.size[0]):
				for col in range(self.size[1]):
					s = (row, col)
					v = value_function[row][col]
					qvalue = np.zeros(self.num_actions)
					for a in range(self.num_actions):
						prob = 1
						ns = self.next_state(s, a)
						r = self.get_reward(s)
						qvalue[a] +=  prob * (r + self.gamma * value_function[ns[0]][ns[1]])

					value_function[row][col] = np.max(qvalue)
					delta = max(delta, abs(v - value_function[row][col]))
						
			if delta < self.tolerance:
				break
		
		# Finding the optimal policy 
		for row in range(self.size[0]):
			for col in range(self.size[1]):
				s = (row, col)
				if self.terminal_state != None:
					if s == self.terminal_state:
						policy[s] = [None]
						continue
					
				qvalue = np.zeros(self.num_actions)
				for a in range(self.num_actions):
					prob = 1
					ns = self.next_state(s, a)
					r = self.get_reward(s)
					qvalue[a] += prob * (r + self.gamma * value_function[ns[0]][ns[1]])
					
				opt_qvalue = np.max(qvalue)
				policy[s] = [x for x in np.where(qvalue == opt_qvalue)[0]]
					
		return value_function, policy

	# Function to compute the Q value function matrix
	def Q_value_function_matrix(self):

		# NOTE: It is sufficiecnt to get Q values only for the optimal action 
		Q_value_matrix = np.zeros((self.size[0], self.size[1]), dtype = self.dtype)
		value_function_ns = np.zeros((self.size[0], self.size[1]), dtype = self.dtype)
		for row in range(self.size[0]):
			for col in range(self.size[1]):
				s = (row, col)
				opt_action = self.policy[s][0]
				prob = 1
				ns = self.next_state(s, opt_action)
				r = self.get_reward(s)
				qvalue = prob * (r +  self.gamma * self.value_function[ns[0]][ns[1]])
				Q_value_matrix[row][col] = qvalue
				value_function_ns[row][col] = self.value_function[ns[0]][ns[1]]
		
		return Q_value_matrix, value_function_ns