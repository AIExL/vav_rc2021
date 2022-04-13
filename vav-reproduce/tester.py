"""
    Following codebase is the re-implementation to conduct experiments for 
    the paper titled 'Value Alignment Verification (http://proceedings.mlr.press/v139/brown21a/brown21a.pdf)'
    The cases of (explicit human, explicit robot) and (explicit human implicit robot) are implemented
"""

# Import the required libraries
import scipy
import numpy as np 
import math, os, sys
import random
import copy, time
from scipy.spatial import distance
from scipy.optimize import linprog

# Tester class to implement reward weight tester
# NOTE: For this type of tester, both weight vectors for both human and robot are known
class RewardWeightTester():

	def __init__(self, human, robot=None, horizon = 10, dtype = np.float32, precision=1e-4, linprog_bounds = None):
		
		"""
			human: Human gridworld
			robot: Robot gridworld
			horizon: Maximum number of states until which a trajectory is generated
			dtype: Data type used
			precision: Small value used for removal of duplicate queries
			linprog_bounds: Bounds on the variable used removing redundant queries
		"""
		
		self.human = human
		self.robot = robot
		self.horizon = horizon
		self.dtype = dtype
		self.precision = precision
		self.linprog_bounds = linprog_bounds
		
		# Initialise delta matrix, aligned, verdict to None
		self.delta_matrix = None
		self.aligned = None
		self.verdict = None

	def get_num_queries(self):
		# only one query to robot vector is needed
		self.num_queries = 1
	
	# Create the ARP Matrix containing all possible delta vectors
	def all_delta_vectors(self):
		
		# Initialise delta matrix as an empty numpy array
		# NOTE: Length of delta vector same as number of unique states
		self.delta_matrix = np.array([], dtype = self.dtype).reshape(0, self.human.num_states)
		
		# Lists to track the trajectories 
		# all_trajectories_a: Trajectory generated when the action in first state is optimal
		# all_trajectories_b: Trajectory generated when the action in first state is sub-optimal
		self.all_trajectories_a, self.all_trajectories_b = [], []

		# Iterate over all the states of the gridworld
		for row in range(self.human.size[0]):
			for col in range(self.human.size[1]):

				# s is the current state (start state)
				s = (row, col)
				
				assert self.human.policy is not None, "Please initialize a policy OR run value iteration!"
				
				# Define the set of optimal action for the current state
				# Define the set of all actions from the action space
				# Define the set of non optimal actions for the current state
				opt_actions = self.human.policy[s]
				set_opt_actions = set(opt_actions)
				set_all_actions = set(range(self.human.num_actions))
				non_opt_actions = list(set_all_actions - set_opt_actions)

				# If the current state is terminal state, ignore that state
				if s == self.human.terminal_state:
					continue

				# Iterate over each action in the set of optimal action
				for a in opt_actions:

					# Iterate over each action in the set of non-optimal actions
					for b in non_opt_actions:
						
						# For action a
						t, a_, s = 0, a, (row, col)
						phi_a = np.zeros(self.human.num_states)
						traj_a = []

						# Iterate till horizon number of states
						# NOTE: We need not stop once the agent reaches terminal state. 
						# NOTE: It will keep on collecting the rewards in the terminal state for the remaining number of steps.
						while t < self.horizon:
							traj_a.append((s, a_))
							# Obtain the feature count for the current state 
							phi_a = phi_a + self.human.gamma**t * (self.human.state_feature_matrix[s[0]][s[1]])
							# Update the current state to the next state
							s = self.human.next_state(s, a_) 
							# Obtain the set of optimal actions
							opt_actions = self.human.policy[s]
							# Pick a random action to be taken in the next state
							idx = random.randint(0, len(opt_actions) - 1)
							a_ = opt_actions[idx]
							t += 1
							
						# for action b
						t, b_, s = 0, b, (row, col)
						phi_b = np.zeros(self.human.num_states)
						traj_b = []

						# Iterate till horizon number of states
						# NOTE: We need not stop once the agent reaches terminal state. 
						# NOTE: It will keep on collecting the rewards in the terminal state for the remaining number of steps.
						while t < self.horizon:
							traj_b.append((s, b_))
							# Obtain the feature count for the current state 
							phi_b = phi_b + self.human.gamma**t * (self.human.state_feature_matrix[s[0]][s[1]])
							# Update the current state to the next state
							s = self.human.next_state(s, b_)
							# Obtain the set of optimal actions
							opt_actions= self.human.policy[s]
							# Pick a random action to be taken in the next state
							idx = random.randint(0, len(opt_actions) - 1)
							b_ = opt_actions[idx]
							t += 1

						# Assert: Ambiguous gridworld contains a few states which result in a loop when using optimal action. 
						# Such a grid world might contain two adjacent states in which the agent can loop forever (up & down).
						# This will lead to the agent getting higher rewards for trajectory B (first non opt action, the opt action)
						# than trajectory A (all opt actions).

						try:
							self.delta_matrix = np.concatenate((self.delta_matrix, (phi_a - phi_b).reshape(1, -1)), axis = 0)
						except:
							print('An ambiguous gridworld is created. Please re-run the code.')
							print('Exiting')
							exit(0)

						# Append the trajectories thus generated
						self.all_trajectories_a.append(traj_a)
						self.all_trajectories_b.append(traj_b)
		
		print("(all_delta_vectors) Total number of questions: ", self.delta_matrix.shape[0])

	# Function to remove the duplicate query vectors from the ARP Matrix
	def remove_duplicate_vectors(self, constraints = None):
		
		# NOTE: constraints is not None for Set Cover Optimal Teaching Tester
		if constraints is None:
			copy_delta_matrix = copy.deepcopy(self.delta_matrix)
		else:
			copy_delta_matrix = copy.deepcopy(constraints)

		self.traj = list(range(copy_delta_matrix.shape[0]))
		
		# Initialise delta matrix to an empty numpy array
		delta_matrix = np.array([], dtype = self.dtype).reshape(0, self.human.num_states)
		i = 0

		# Iterate till the shape of copy delta matrix is zero i.e.,
		# Until all the query vectors are checked for duplication
		while copy_delta_matrix.shape[0] > 0:
			row = copy_delta_matrix[0]
			if delta_matrix.shape[0] == 0:
				delta_matrix = np.concatenate((delta_matrix, row.reshape(1, -1)), axis = 0)
				copy_delta_matrix = np.delete(copy_delta_matrix, 0, 0)
				i+=1
				continue
			# If the cosine distance between the new delta vector and the existing delta vectors
			# is greater than a precision value, add this vector to set of non-duplicate delta vectors 
			cosine_distances = np.dot(delta_matrix, row) / ((np.linalg.norm(delta_matrix, axis = 1) * np.linalg.norm(row)) + 1e-9)
			if np.sum((1 - cosine_distances) > self.precision) == cosine_distances.shape[0]:
				delta_matrix = np.concatenate((delta_matrix, row.reshape(1, -1)), axis = 0)
			else:
				self.traj.remove(i)
				pass
			i += 1
			copy_delta_matrix = np.delete(copy_delta_matrix, 0, 0)

		# Return delta matrix for Set Cover Optimal Teaching Tester		
		if constraints is not None:
			return delta_matrix
		else:
			self.delta_matrix = copy.deepcopy(delta_matrix)

		print("(remove_duplicate_vectors) Total number of questions: ", self.delta_matrix.shape[0])
	
	# Function to remove the redundant query vectors from delta matrix
	# NOTE: A redundant constraint is one that can be removed without changing the interior of the intersection of half-spaces.
	# To check if a constraint a^T x <= b is binding we can removethat constraint and solve the linear program with
	# max (a^T x) as the objective. If the optimal solution is still constrained to be less than or equal to b
	# even when the constraint is removed, then the constraint can be removed. 
	def remove_redundant_vectors(self):
		
		# Create a copy of the delta matrix and empty the original delta matrix
		copy_delta_matrix = copy.deepcopy(self.delta_matrix)
		self.delta_matrix = np.array([], dtype = self.dtype).reshape(0, self.human.num_states)
		
		i = 0
		self.traj_deep_copy = copy.deepcopy(self.traj)

		# Iterate till all the vectors in delta matrix are checked for redundancy.
		while copy_delta_matrix.shape[0] > 0:

			row = copy_delta_matrix[0]
			copy_delta_matrix = np.delete(copy_delta_matrix, 0, 0)

			if self.delta_matrix.shape[0] == 0:
				self.delta_matrix = np.concatenate((self.delta_matrix, row.reshape(1, -1)), axis = 0)
				i+=1
				continue

			temporary_delta_matrix = np.vstack((self.delta_matrix, copy_delta_matrix))
			flag = True	# Assuming that the row is redundant (True)
			b = np.zeros(temporary_delta_matrix.shape[0])
			
			# Obtain the solution for the linear programming problem using revised simplex method
			sol = linprog(row.reshape(1, -1), A_ub = -temporary_delta_matrix, b_ub = b, bounds = self.linprog_bounds, method = 'revised simplex') 

			# If no solution is obtained using revised simplex, use the default algorithm i.e., interior point
			if sol['status'] != 0:
				sol = linprog(row.reshape(1, -1), A_ub = -temporary_delta_matrix, b_ub = b, bounds = self.linprog_bounds)
			
			# NOTE: Original authors have set a small value -epsilon(-0.0001) instead of 0
			if sol['fun'] < -0.0001: 
				flag = False
			elif sol['status'] != 0: 
				flag = False
			else: 
				flag = True

			# If the vector is not redundant, concatenate the vector to the set of non redundant delta vectors
			if not flag:
				self.delta_matrix = np.concatenate((self.delta_matrix, row.reshape(1, -1)), axis = 0)
			else:
				ele = self.traj_deep_copy[i]
				self.traj.remove(ele)
				pass
			
			i += 1

		print("(remove_redundant_vectors) Total number of questions: ", self.delta_matrix.shape[0])
		
	# Create the final ARP Matrix by removing the duplicate and redundant vectors
	def create_ARPMatrix(self):

		print("-------------------- Generating all delta vectors --------------------")
		self.all_delta_vectors()
		print("\n-------------------- Remove duplicate vectors ------------------------")
		self.remove_duplicate_vectors()
		print("\n-------------------- Remove redundant vectors ------------------------")
		self.remove_redundant_vectors()
		print("\n")

		"""
			NOTE: uncomment the following lines to print the trajectories (valid for 'rwt' and 'ptt')
			print("------------------------- (trajectory a) --------------------------: ")
			print([self.all_trajectories_a[x] for x in self.traj])
			print("------------------------- (trajectory b) --------------------------: ")
			print([self.all_trajectories_b[x] for x in self.traj])
		"""

		# We have obtained the succint delta matrix

		self.get_num_queries()

	# Function to check alignment of a robot with human
	def check_alignment(self, robot = None):

		# Assert that the robot is not None
		assert robot != None or self.robot != None, "Please provide an robot!"

		self.robot = robot

		# If the number if actions in the action space of robot does not match to that of human
		# The robot is not aligned to the human
		if self.robot.num_actions != self.human.num_actions:
			self.aligned = False
			self.verdict = "The robot is not aligned"

		# Obtain the product of delta matrix and weight vector
		if robot == None:
			if self.human.feature_type == "linear":
				delta_w = np.dot(self.delta_matrix, self.robot.weights)
			elif self.human.feature_type == "cubic":
				delta_w = np.dot(self.delta_matrix, self.robot.weights) ** 3 \
							+ 10 * np.dot(self.delta_matrix, self.robot.weights)
			elif self.human.feature_type == "exponential":
				delta_w = np.exp(np.dot(self.delta_matrix, self.robot.weights))
		else:
			if self.human.feature_type == "linear":
				delta_w = np.dot(self.delta_matrix, robot.weights)
			elif self.human.feature_type == "cubic":
				delta_w = np.dot(self.delta_matrix, robot.weights) ** 3 \
							+ 10 * np.dot(self.delta_matrix, robot.weights)
			elif self.human.feature_type == "exponential":
				delta_w = np.exp(np.dot(self.delta_matrix, robot.weights))
		
		# If all the elements is the delta vector are greater than zero, the robot is aligned
		if np.sum(delta_w > 0) == delta_w.shape[0]:
			self.aligned = True
			self.verdict = "The robot is aligned"
		else:
			self.aligned = False
			self.verdict = "The robot is not aligned"

# Tester class to implement reward tester (inherits RewardWeightTester class)
# NOTE: For this type of tester, the weight vector for human and reward function for robot are known
# Using the robot reward function, we can solve to obtain the robot weights and call RewardWeightTester subsequently
class RewardTester(RewardWeightTester):

	def __init__(self, human, robot=None, horizon = 10, dtype = np.float32, precision = 1e-4, linprog_bounds = None):
		super(RewardWeightTester, self).__init__()

		"""
			human: Human gridworld
			robot: Robot gridworld
			horizon: Maximum number of states until which a trajectory is generated
			dtype: Data type used
			precision: Small value used for removal of duplicate queries
			linprog_bounds: Bounds on the variable used removing redundant queries
		"""

		self.human = human
		self.robot = robot
		self.horizon = horizon
		self.dtype = dtype
		self.precision = precision
		self.linprog_bounds = linprog_bounds
		self.delta_matrix = None
		self.aligned = None
		self.verdict = None

	def get_num_queries(self):
		# only num_states to robot vector is needed
		self.num_queries = self.human.num_states

	# Function to solve for robot weights using the robot reward function
	def solve_weights(self):
		unique_reward, unique_state_feature_matrix = self.robot.get_unique_reward()
		robot_temp = copy.deepcopy(self.robot)
		robot_temp.weights = np.linalg.pinv(unique_state_feature_matrix).dot(unique_reward)
		robot_temp.weights = np.round(robot_temp.weights, decimals=4)
		return robot_temp

	# Define a temporary robot instance
	def define_temp_robot(self, robot):
		self.robot = robot
		self.robot_temp = self.solve_weights() # updates robot's weights, but obtains directly R
		self.robot_temp.value_iteration()

# Tester class to implement value function tester (inherits RewardWeightTester class)
# NOTE: For this type of tester, the weight vector for human and value function for robot are known
# Using the robot value function, we can solve to obtain the robot weights and call RewardWeightTester subsequently
class ValueFunctionTester(RewardWeightTester):
	
	def __init__(self, human, robot=None, horizon = 10, dtype = np.float32, precision = 1e-4, linprog_bounds = None):
		super(RewardWeightTester, self).__init__()
		
		"""
			human: Human gridworld
			robot: Robot gridworld
			horizon: Maximum number of states until which a trajectory is generated
			dtype: Data type used
			precision: Small value used for removal of duplicate queries
			linprog_bounds: Bounds on the variable used removing redundant queries
		"""
		
		self.human = human
		self.robot = robot
		self.horizon = horizon
		self.dtype = dtype
		self.precision = precision
		self.linprog_bounds = linprog_bounds
		self.delta_matrix = None
		self.aligned = None
		self.verdict = None

	def get_num_queries(self):
		# only num_states * (num_actions + 1) to robot vector is needed
		self.num_queries = self.human.num_states * (self.human.num_actions + 1)

	# Get the reward vector for the robot from its Q value matrix and value function over next states
	def get_reward(self):
		return self.robot.Q_value_matrix - self.robot.gamma * self.robot.value_function_ns

	# Solve for weights from the obtained reward vector
	def solve_weights(self):

		# Iterate over each state
		for row in range(self.human.size[0]):
			for col in range(self.human.size[1]):
				s = (row, col)
				feat = self.robot.state_feature_matrix[row][col]
				index = np.argmax(feat)
				self.unique_reward[index] = self.state_reward[row][col] 

		self.robot_temp = copy.deepcopy(self.robot)
		self.robot_temp.weights = np.round(
						np.linalg.pinv(self.unique_state_feature_matrix).dot(self.unique_reward),
						decimals = 4)

	# Define a temporary robot instance
	def define_temp_robot(self, robot):
		self.robot = robot
		self.robot.Q_value_matrix, self.robot.value_function_ns = self.robot.Q_value_function_matrix()
		self.state_reward = self.get_reward()

		self.unique_state_feature_matrix = np.eye(self.human.num_states)
		self.unique_reward = np.zeros((self.human.num_states, 1))
		self.solve_weights() # updates robot's weights, but obtains directly R
		self.robot.value_function, self.robot.policy = self.robot_temp.value_iteration()

	# Function to check alignment of a robot with a human
	def check_alignment(self, robot):

		assert robot != None or self.robot != None, "Please provide an robot!"

		self.robot = robot
		# NOTE: Value function ns gives the expectation of the value function of 
		# next state after taking optimal action at the given state

		# If the number of action in the action space of the robot is not equal to that human
		# The robot is not aligned to the human
		if self.robot.num_actions != self.human.num_actions:
			self.aligned = False
			self.verdict = "The robot is not aligned"

		# Obtain the delta w vector
		if robot == None:
			delta_w = np.dot(self.delta_matrix, self.robot.weights)
		else:
			delta_w = np.dot(self.delta_matrix, robot.weights)
		
		# If each element of the delta w vector is greater than zero, the robot is aligned
		if np.sum(delta_w > 0) == delta_w.shape[0]:
			self.aligned = True
			self.verdict = "The robot is aligned"
		else:
			self.aligned = False
			self.verdict = "The robot is not aligned"

# Tester class to implement preference trajectory function tester (inherits RewardWeightTester class)
# NOTE: For this type of tester, the weight vector for human and preference over actions at a state for robot are known
# Using the preferences given by the robot, alignment is to checked.
class PreferenceTrajectoryTester(RewardWeightTester):

	def __init__(self, human, robot=None, horizon = 10, dtype = np.float32, precision = 1e-4, linprog_bounds = None):
		super(RewardWeightTester, self).__init__()
		
		"""
			human: Human gridworld
			robot: Robot gridworld
			horizon: Maximum number of states until which a trajectory is generated
			dtype: Data type used
			precision: Small value used for removal of duplicate queries
			linprog_bounds: Bounds on the variable used removing redundant queries
		"""

		self.human = human
		self.robot = robot
		self.horizon = horizon
		self.dtype = dtype
		self.precision = precision
		self.linprog_bounds = linprog_bounds
		
		# Initialise delta matrix, aligned and verdict to None
		self.delta_matrix = None
		self.aligned = None
		self.verdict = None

		# Lists to track the trajectories 
		# all_trajectories_a: Trajectory generated for action a
		# all_trajectories_b: Trajectory generated for action b
		self.all_trajectories_a = []
		self.all_trajectories_b = []

	# Function to check for preference over two trajectories
	def check_preference(self, phi_a, phi_b):

		# If the feature counts for two trajectories are same, return None
		# Else if (phi_a - phi_b) * w > 0, trajectory A is preferred and vice versa
		if np.array_equal(phi_a, phi_b):
			return None, 'A and B are similar'
		if np.dot(phi_a - phi_b, self.human.weights) > 0:
			return phi_a - phi_b, 'A is better'
		else:
			return phi_b - phi_a, 'B is better'

	def get_num_queries(self):
		# only num_states * (num_actions + 1) to robot vector is needed
		self.num_queries = self.delta_matrix.shape[0]

	# Function to create ARP matrix exhaustively with all the delta vectors
	def all_delta_vectors(self):

		# Create an empty delta matrix
		self.delta_matrix = np.array([], dtype = self.dtype).reshape(0, self.human.num_states)

		# Lists to track the trajectories 
		# all_trajectories_a: Trajectory generated for action a
		# all_trajectories_b: Trajectory generated for action b
		# NOTE: Length of delta vector same as number of unique states
		self.all_trajectories_a, self.all_trajectories_b = [], []
		
		# Iterate over all the states in gridworld
		for row in range(self.human.size[0]):
			for col in range(self.human.size[1]):

				# s stores the current state
				s = (row, col)

				assert self.human.policy is not None, "Please initialize a policy OR run value iteration!"
				
				# Obtain the set of all actions for the human
				set_all_actions = set(range(self.human.num_actions))
				
				# Iterate over all the actions 
				for a in set_all_actions:
					for b in set_all_actions:
						traj_a, traj_b = [], []

						# for action a
						t, a_, s = 0, a, (row, col)
						phi_a = np.zeros(self.human.num_states)
						while t < self.horizon:
							traj_a.append(s)
							# Update the feature count for a
							phi_a = phi_a + self.human.gamma**t * (self.human.state_feature_matrix[s[0]][s[1]])
							# If the human reaches the terminal state, the trajectory gets over
							if s == self.human.terminal_state:
								break
							# Get the next state and update the current state to the next state
							ns = self.human.next_state(s, a_)
							s = ns
							# Randomly pick an optimal action in the next state							
							opt_actions = self.human.policy[s]
							idx = random.randint(0, len(opt_actions) - 1)
							a_ = opt_actions[idx]
							t += 1

						# for action b
						t, b_, s = 0, b, (row, col)
						phi_b = np.zeros(self.human.num_states)
						while t < self.horizon:
							traj_b.append(s)
							# Update the feature count for b
							phi_b = phi_b + self.human.gamma**t * (self.human.state_feature_matrix[s[0]][s[1]])
							# If the human reaches the terminal state, the trajectory gets over
							if s == self.human.terminal_state:
								break
							# Get the next state and update the current state to the next state
							ns = self.human.next_state(s, b_)
							s = ns
							# Randomly pick an optimal action in the next state
							opt_actions= self.human.policy[s]
							idx = random.randint(0, len(opt_actions) - 1)
							b_ = opt_actions[idx]
							t += 1

						# Get the preference over the trajectories
						pref_query, A = self.check_preference(phi_a, phi_b)
						if pref_query is None:
							continue

						# Concatenate the preference query to delta matrix
						assert np.dot(pref_query, self.human.weights) > 0, f"state: {(row, col)} action a: {a} action b: {b}; human is not aligned!"
						self.delta_matrix = np.concatenate((self.delta_matrix, pref_query.reshape(1, -1)), axis = 0)
							
						self.all_trajectories_a.append([traj_a, A])
						self.all_trajectories_b.append([traj_b, A])

		print("(all_delta_vectors) Total number of questions: ", self.delta_matrix.shape[0])
		# Once the delta matrix is computed, duplicate and redundant vectors are removed, subsequently the robot is check for aligned.

# Tester class to implement critical states tester
# NOTE: The robot is queried for an optimal action at particular states known as critical states
class CriticalStatesTester():

	def __init__(self, human, robot = None, precision = 1e-4, threshold = 0.8):
		
		"""
			human: Human gridworld
			robot: Robot gridworld
			precision: Small value used for removal of duplicate queries
			threshold: Threshold for a state to be critical
		"""

		self.human = human
		self.robot = robot
		self.precision = precision
		self.threshold = threshold
		self.critical_state_actions = {}
		self.verdict = "The robot is aligned"
		self.aligned = True

		# Obtain the list of critical states
		self.construct_critical_state_actions()

	# Obtain the Q value for a particular state
	def get_q_value(self, state, action):

		ns = self.human.next_state(state, action)
		r = self.human.get_reward(state)
		qvalue = r +  self.human.gamma * self.human.value_function[ns[0]][ns[1]]

		return qvalue

	# Function to create the list of critical states at which the robot is tested
	def construct_critical_state_actions(self):
		
		# Iterate over all the states of the human gridworld
		for row in range(self.human.size[0]):
			for col in range(self.human.size[1]):

				# s is the current state
				s = (row, col)

				# Pick a random optimal action at the current state
				opt_actions = self.human.policy[s]
				idx = random.randint(0, len(opt_actions) - 1)
				random_opt_action = opt_actions[idx]

				# Obtain the Q value for the sampled action
				opt_q_value = self.get_q_value(s, random_opt_action)
				
				# Compute the average Q value for the state with all action in the action space
				average_q_value = 0
				for a in range(self.human.num_actions):
					average_q_value += self.get_q_value(s, a)
				average_q_value = average_q_value / self.human.num_actions

				# If the difference between optimal Q value and average Q value is greater than a threshold,
				# The current state is a critical state
				if opt_q_value - average_q_value > self.threshold:
					self.critical_state_actions[s] = opt_actions

	# Function to check alignment of a robot 
	def check_alignment(self, robot):

		self.robot = robot
		self.num_queries = len(self.critical_state_actions)
		
		# Iterate over the critical states, obtain a random optimal action of the robot,
		# If the action is optimal under human reward for all the states, the robot is aligned 
		for state, action_list in self.critical_state_actions.items():
			robot_actions = self.robot.policy[state]

			idx = random.randint(0, len(robot_actions) - 1)
			random_robot_action = robot_actions[idx]
			
			if random_robot_action not in action_list:
				self.verdict = "The robot is not aligned"
				self.aligned = False
				break

# Tester class to implement Set Cover Optimal Teaching Tester (inherits RewardWeightTester)
# NOTE: A minimal set of maximally informative state action trajectories is generated and the 
# robot is queried for action over those generated states
class SetCoverOptimalTeachingTester(RewardWeightTester):

	def __init__(self, human, robot = None, horizon = 10, precision = 1e-4, approx_horizon = 5):
		super(SetCoverOptimalTeachingTester, self).__init__(human, robot = robot, horizon = horizon, 
												dtype = np.float32, precision = precision)

		"""
			human: Human gridworld
			robot: Robot gridworld
			horizon: Maximum number of states until which a trajectory is generated
			precision: Small value used for removal of duplicate queries
			approx_horizon = Maximum number of states until which a candidate trajectory is generated
		"""

		self.human = human
		self.robot = robot
		self.precision = precision
		self.horizon = horizon
		self.approx_horizon = approx_horizon
		self.verdict = "The robot is aligned"
		self.aligned = True

		# List to store all trajectories
		self.all_trajectories = []

	# Function to generate the set of candidate trajectories
	def generate_candidate_trajectories(self):
		
		# Initialise candidate trajectories as an empty numpy array
		# NOTE: Length of delta vector same as number of unique states
		self.candidate_trajectories = np.array([], dtype = self.dtype).reshape(0, self.human.num_states)

		# Iterate over all the states
		for row in range(self.human.size[0]):
			for col in range(self.human.size[1]):
				
				# s stores the current state
				s = (row, col)
				
				assert self.human.policy is not None, "Please initialize a policy OR run value iteration!"
				
				# Obtain the set of optimal actions
				opt_actions = self.human.policy[s]

				# If s is terminal state, continue
				if s == self.human.terminal_state:
					continue

				# Iterate over all the action in the set of optimal actions corresponding to first state
				for a in opt_actions:
						
					# for action a
					t, a_, s = 0, a, (row, col)
					phi_a = np.zeros(self.human.num_states)
					traj_a = []

					while t < self.approx_horizon:
						traj_a.append((s, a_))
						# Obtain the next state and update current state to next state
						# Randomly sample an optimal action for the next state
						s = self.human.next_state(s, a_) 
						opt_actions = self.human.policy[s]
						idx = random.randint(0, len(opt_actions) - 1)
						a_ = opt_actions[idx]
						t += 1

					self.all_trajectories.append(traj_a)

	# Function to obtain all the constraints corresponding to the trajectories
	def get_all_constraints_traj(self, traj):
		
		# Initialise constraints as an empty numpy array
		# NOTE: Length of delta vector same as number of unique states
		constraints = np.array([], dtype = self.dtype).reshape(0, self.human.num_states)
		self.all_trajectories_a, self.all_trajectories_b = [], []
		
		# Iterate over the state-action pairs for a trajectory
		for s, a in traj:

			# Obtain the set of all actions 
			set_all_actions = set(range(self.human.num_actions))

			# If s is terminal state, continue
			if s == self.human.terminal_state:
				continue
			
			# Obtain the feature count for action a (which is optimal)
			phi_a = self.human.state_feature_matrix[s[0]][s[1]]
			ns = self.human.next_state(s, a)
			phi_a = phi_a + self.human.gamma * (self.human.state_feature_matrix[ns[0]][ns[1]])

			# Iterate over the set of all action in the action space of human
			for b in set_all_actions:
				
				# Obtain the feature count for action b
				phi_b = self.human.state_feature_matrix[s[0]][s[1]]
				ns = self.human.next_state(s, b)
				phi_b = phi_b + self.human.gamma * (self.human.state_feature_matrix[ns[0]][ns[1]])
				
				# If the norm of difference between feature counts of a and b is greater than some precision, 
				# add the difference vector as a constraint
				if np.linalg.norm(phi_a - phi_b) > self.precision:
					constraints = np.concatenate((constraints, (phi_a - phi_b).reshape((1, -1))), axis = 0)	

		# Remove the duplicate constraints from the set of constraints
		constraints = self.remove_duplicate_vectors(constraints)

		return constraints

	# Function to count new set covers
	def count_new_covers(self, constraints_new, constraint_set, covered):

		count = 0
		for c_new in constraints_new:
			for i,c in enumerate(constraint_set):
				# Check if equal via cosine distance
				if distance.cosine(c_new, c) < self.precision:
					# Check if not covered yet
					if not covered[i]:
						count += 1
		return count

	# Function to update the set of covered constraints
	def update_covered_constraints(self, constraints_to_add, constraint_set, covered):
		for c_new in constraints_to_add:
			for i,c in enumerate(constraint_set):
				# Check if equal via cosine distance
				if distance.cosine(c_new, c) < self.precision:
					# Check if not covered yet
					if not covered[i]:
						covered[i] = True
		return covered

	# Function to obatin the maximally informative trajectories	
	def get_maximally_informative_trajectories(self):
		
		self.constraint_set = copy.deepcopy(self.delta_matrix)
		self.generate_candidate_trajectories()
		
		# List to track whether a constraint is covered or not 
		covered = [False] * len(self.constraint_set)

		total_covered = 0

		# opt_demos stores the maximally informative trajectories
		opt_demos = []

		temp = 0
		while total_covered < len(self.constraint_set):

			if temp > len(self.all_trajectories):
				break

			temp += 1
			constraints_to_add = []
			best_traj = None
			max_count = -1
			
			# Iterate over all the trajectories
			for traj in self.all_trajectories:
				
				constraints_new = self.get_all_constraints_traj(traj)
				count = self.count_new_covers(constraints_new, self.constraint_set, covered)
				if count >= max_count:
					max_count = count
					constraints_to_add = constraints_new
					best_traj = traj

			# Update covered flags and add best_traj to demo
			opt_demos.append(best_traj)
			covered = self.update_covered_constraints(constraints_to_add, self.constraint_set, covered)
			total_covered += max_count

		return opt_demos

	def check_alignment(self, robot):

		self.robot = robot

		self.all_trajectories = []
		self.opt_demos = self.get_maximally_informative_trajectories()

		# NOTE: The covered boolean in the above function may not reach to all true values
		# Let's say it reaches [True, False, False], then len(opt_demos) can also be greater than zero.
		# So this may result either in false positive or false negative case, 
		# even though SCOT has not completely failed

		self.num_queries = len(self.opt_demos[0])

		if len(self.opt_demos) == 0:
			self.aligned = False
			self.verdict = "SCOT Tester failed"
		
		# For each state-action pair in opt_demos, query for a robot action and 
		# check whether it is optimal under human reward function
		# If the condition holds for all the states, the robot is aligned.
		for state, action in self.opt_demos[0]:
			
			opt_actions = self.human.policy[state]
			robot_actions = self.robot.policy[state]

			idx = random.randint(0, len(robot_actions) - 1)
			random_robot_action = robot_actions[idx]
			
			if random_robot_action not in opt_actions:
				self.verdict = "The robot is not aligned"
				self.aligned = False
				break

# Tester class to implement ARP Black Box Tester (inherits RewardWeightTester)
# NOTE: ARP-bb first computes delta matrix, removes redundant half-space constraints via linear
# programming, and then only queries for robot actions from the states corresponding 
# to the non-redundant constraints (rows) in delta matrix. 
class ARPBBTester(RewardWeightTester):

	def __init__(self, human, robot = None, horizon = 10, dtype = np.float32, precision = 1e-4, linprog_bounds = None):
		super(ARPBBTester, self).__init__(human, robot = robot, horizon = horizon, dtype = dtype, precision = precision, linprog_bounds = linprog_bounds)

		"""
			human: Human gridworld
			robot: Robot gridworld
			horizon: Maximum number of states until which a trajectory is generated
			dtype: Data type used
			precision: Small value used for removal of duplicate queries
			linprog_bounds: Bounds on the variable used removing redundant queries
		"""

		self.human = human
		self.robot = robot
		self.horizon = horizon
		self.dtype = dtype
		self.precision = precision

		self.create_ARPMatrix()

	# Function to check alignment of a robot with human
	def check_alignment(self, robot):
		
		# Iterate over all the trajectories corresponding to non redundant constraints
		self.num_queries = len(self.traj)
		for traj_num in self.traj:

			# state stores the current state
			state = self.all_trajectories_a[traj_num][0][0]

			# Obtain set of optimal action for human and robot
			opt_actions = self.human.policy[state]
			robot_actions = robot.policy[state]

			# Pick a random action for robot
			idx = random.randint(0, len(robot_actions) - 1)
			random_robot_action = robot_actions[idx]
			
			# If robot action is not optimal under the human policy, robot is not aligned
			if random_robot_action not in opt_actions:
				self.verdict = "The robot is not aligned"
				self.aligned = False
				break
			else:
				self.verdict = "The robot is aligned"
				self.aligned = True