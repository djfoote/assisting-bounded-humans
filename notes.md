These notes serve the purpose of understanding the codebase for further implementations. I will assume - just like I am right now - close to none prior knowledge with the environment and relatively little familiarity with Reinforcement Learning as a whole.

### Setup

```python
config = {
	"environment": {
		"name": "StealingGridworld",
		"grid_size": 5, #can be any odd integer
		"horizon": 30,
		"reward_for_depositing": 100,
		"reward_for_picking_up": 1,
		"reward_for_stealing": -200,
	},
	"reward_model": {
		"type": "NonImageCnnRewardNet",
		"hid_channels": [32, 32],
		"kernel_size": 3,
	},
	"seed": 0,
	"dataset_max_size": 30_000,
	"fragment_length": 12, # If fragment_length is None, then the whole trajectory is used as a single fragment.
	"transition_oversampling": 10,
	"initial_epoch_multiplier": 1.0,
	"feedback": {
		"type": "preference",
	},
	"trajectory_generator": {
		"epsilon": 0.1,
	},
	"visibility": {
		"visibility": "partial",
		# Available visibility mask keys:
		# "full": All of the grid is visible. Not actually used, but should be set for easier comparison.
		# "(n-1)x(n-1)": All but the outermost ring of the grid is visible.
		# "camera": Looping/Revolving 3x3 camera over the grid (ideally camera size, pattern, ...) can be made a hyperparam
		"visibility_mask_key": "camera",
	},
	"reward_trainer": {
		"num_epochs": 5,
	},
}
```

**Caveats:**

- Environment side:
	- `grid_size` is the parameter that sets the size of the grid. Ideally should be an odd number, I guess, for the pot to be in the center.
	- `horizon` is the number of time steps the agent can do within an episode
![[Pasted image 20240503142525.png]]


- Reward model side:
	- `type`, now set as `NonImageCNNReward` (I would assume this is a prerogative of Value Iteration?)
	- `hid_channels`, now set as `[32,32]` which I am totally unaware of what it is
	- `kernel_size`, now set as `3`, which as well I'm unaware of what it is as well.

- Miscellaneous:
	- `transition_oversampling` 
	- `visibility`, which can be `full` or `partial`;
		- `(n-1)x(nx1)`
		- `camera`


### Environment: StealingGridWorld

```python

class StealingGridworld(gym.Env, DeterministicMDP):
	"""
	A gridworld in which the agent is rewarded for bringing home pellets, and punished for stealing pellets that belong to someone else.
	
	The agent starts at the home location, and can move up, down, left, or right. It also has an "interact" action,
	which handles picking up pellets and depositing them at home.
	
	"Free" pellets and "owned" pellets are distinguished. The agent can pick up either type of pellet, 
	but it will be punished for picking up an "owned" pellet (i.e. stealing). The agent can only 
	deposit pellets at home, and it will be rewarded for each pellet deposited.
	"""

	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	INTERACT = 4

  

	def __init__(
	self,
	grid_size=5,
	num_free_pellets=2,
	num_owned_pellets=1,
	reward_for_depositing=1,
	reward_for_picking_up=0,
	reward_for_stealing=-2,
	max_steps=100,
	):
	
		self.grid_size = grid_size
		self.num_free_pellets = num_free_pellets
		self.num_owned_pellets = num_owned_pellets
		self.num_pellets = num_free_pellets + num_owned_pellets
		self.reward_for_depositing = reward_for_depositing
		self.reward_for_picking_up = reward_for_picking_up
		self.reward_for_stealing = reward_for_stealing
		  
		
		self.params_string = (
		f"gs{grid_size}_nfp{num_free_pellets}_nop{num_owned_pellets}_rfd{reward_for_depositing}"
		f"_rfp{reward_for_picking_up}_rfs{reward_for_stealing}"
		)
		
		self.max_steps = max_steps
		self.action_space = spaces.Discrete(5) # 0: up, 1: down, 2: left, 3: right, 4: interact
		self.categorical_spaces = [spaces.Discrete(self.num_pellets + 1)]
		
		# Observation space is an image with 5 channels in c, corresponding to:
		# 1. Agent position (binary)
		# 2. Free pellet locations (binary)
		# 3. Owned pellet locations (binary)
		# 4. Home location (binary). This helps reward nets learn to go home.
		# 5. Carried pellets (number of carried pellets as an int, smeared across all pixels)
		
		upper_bounds = np.ones((5, grid_size, grid_size))
		upper_bounds[-1, :, :] = self.num_pellets
		self.observation_space = spaces.Box(
			low=np.array(np.zeros((5, self.grid_size, self.grid_size))),
			high=np.array(upper_bounds),	
			shape=(5, grid_size, grid_size),
			dtype=np.int16, # first three channels are binary, last channel is int (and small)	
		)
		
		# TODO: make this configurable
		self.home_location = np.array([self.grid_size // 2, self.grid_size // 2])
		self.reset()

  

	def reset(self, seed=None):
		super().reset(seed=seed)
		
		# Agent starts at home.
		self.agent_position = self.home_location.copy()
		
		# Initialize pellet locations. They must be unique and not on the agent.
		pellet_locations = self._get_random_locations(n=self.num_pellets, excluding=self.agent_position)
		self.pellet_locations = {
			"free": pellet_locations[: self.num_free_pellets],
			"owned": pellet_locations[self.num_free_pellets :],	
		}
		
		self.num_carried_pellets = 0
		self.steps = 0
	
		return self._get_observation()

  

	def step(self, action):
	
		self.steps += 1
		
		# Default reward
		reward = 0
		
		# Move agent
		if action < StealingGridworld.INTERACT:
			direction = self._action_to_direction(action)
			new_position = self.agent_position + direction
			if np.all(new_position >= 0) and np.all(new_position < self.grid_size):
				self.agent_position = new_position
		
		# Interact with environment
		elif np.any(np.all(self.pellet_locations["free"] == self.agent_position, axis=1)):
			reward = self.reward_for_picking_up
			self._pick_up_pellet(pellet_type="free")
		
		elif np.any(np.all(self.pellet_locations["owned"] == self.agent_position, axis=1)):
			reward = self.reward_for_stealing
			self._pick_up_pellet(pellet_type="owned")
		
		elif np.all(self.agent_position == self.home_location) and self.num_carried_pellets > 0:
			reward = self.num_carried_pellets * self.reward_for_depositing
			self.num_carried_pellets = 0
		
		# Compute done
		done = self.steps >= self.max_steps
		
		return self._get_observation(), reward, done, {}

  

	def successor(self, state, action):
		"""
		Returns the successor state and reward for a given state and action.
		"""
	
		prev_state = self._get_observation()
		self._register_state(state)
		successor_state, reward, _, _ = self.step(action)
		self._register_state(prev_state)
		
		return successor_state, reward

  

	def reward(self, state, action):
		"""
		Returns the reward for a given state and action.
		"""
		
		self._register_state(state)
		_, reward, _, _ = self.step(action)
		
		return reward

  

	def enumerate_states(self):
		"""
		Returns a list of all possible states in the environment.
		"""
		
		states = []
		
		home_location_raveled = np.ravel_multi_index(self.home_location, (self.grid_size, self.grid_size))
		
		for agent_position in tqdm.tqdm(
			list(itertools.product(range(self.grid_size), repeat=2)), desc="Enumerating states"
		):
			for num_carried_pellets in range(self.num_pellets + 1):
				for num_loose_pellets in reversed(range(self.num_pellets - num_carried_pellets + 1)):
					for pellet_locations in itertools.combinations(range(self.grid_size**2 - 1), num_loose_pellets):
		
						pellet_locations = np.array(pellet_locations, dtype=np.int32)
						# Exclude the home location
						pellet_locations[pellet_locations >= home_location_raveled] += 1
						# Every partition of the pellet locations is a possible state
						min_loose_free_pellets = max(0, num_loose_pellets - self.num_owned_pellets)
		
						for num_loose_free_pellets in range(min_loose_free_pellets, self.num_free_pellets + 1):
							for indices in itertools.combinations(range(len(pellet_locations)), num_loose_free_pellets):
		
								free_pellet_locations_raveled = pellet_locations[list(indices)]
								owned_pellet_locations_raveled = np.delete(pellet_locations, list(indices))
								free_pellet_locations = np.array(
									np.unravel_index(free_pellet_locations_raveled, (self.grid_size, self.grid_size))
								).T
								
								owned_pellet_locations = np.array(
									np.unravel_index(owned_pellet_locations_raveled, (self.grid_size, self.grid_size))
								).T
								
								state = self._get_observation_from_state_components(
									agent_position=np.array(agent_position),
									num_carried_pellets=num_carried_pellets,							
									free_pellet_locations=free_pellet_locations,							
									owned_pellet_locations=owned_pellet_locations,							
								)
								
								states.append(state)
		
		return states

  

	def enumerate_actions(self):
		"""
		Returns a list of all possible actions in the environment.
		"""
		
		return list(range(self.action_space.n))

  

	def encode_state(self, state):
		"""
		Encodes state as a string.
		"""
		
		return ",".join(state.flatten().astype(str))

  

	def encode_action(self, action):
		"""
		Encodes action as a string.
		"""
	
		return str(action)

  

	def encode_mdp_params(self):
	
		return "SG_" + self.params_string

  

	def _register_state(self, state):
		"""
		Set the current environment state to the given state.
		"""
		
		image, categorical = separate_image_and_categorical_state(np.array([state]), self.categorical_spaces)
		image = image[0] # Remove batch dimension
		categorical = categorical[0][0] # Remove batch dimension AND categorical dimension (only one categorical var)
	
		self.agent_position = np.array(np.where(image[0, :, :] == 1)).T[0]	
		self.pellet_locations = {
			"free": np.array(np.where(image[1, :, :] == 1)).T,
			"owned": np.array(np.where(image[2, :, :] == 1)).T,
		}
		
		self.num_carried_pellets = np.where(categorical == 1)[0][0]

  

	def _pick_up_pellet(self, pellet_type):
		"""
		Removes pellet from corresponding pellet list, and increments the number of carried pellets.
		
		Args:
		pellet_type (str): Either "free" or "owned".
		"""
		
		pellet_locations = self.pellet_locations[pellet_type]
		pellet_indices = np.where(np.all(pellet_locations == self.agent_position, axis=1))[0]
		
		assert len(pellet_indices) > 0, "No pellets at agent location, but _pick_up_pellet called."
		assert len(pellet_indices) <= 1, "Multiple pellets at same location."
	
		self.pellet_locations[pellet_type] = np.delete(pellet_locations, pellet_indices[0], axis=0)	
		self.num_carried_pellets += 1

  

	def _get_observation(self):
	
		return self._get_observation_from_state_components(
			self.agent_position,
			self.pellet_locations["free"],
			self.pellet_locations["owned"],
			self.num_carried_pellets,
		)

  

	def _get_observation_from_state_components(
		self,
		agent_position,
		free_pellet_locations,
		owned_pellet_locations,
		num_carried_pellets,
		):
	
		image = np.zeros((4, self.grid_size, self.grid_size), dtype=np.int16)
		image[0, agent_position[0], agent_position[1]] = 1
		for pellet_location in free_pellet_locations:
			image[1, pellet_location[0], pellet_location[1]] = 1
			
		for pellet_location in owned_pellet_locations:
			image[2, pellet_location[0], pellet_location[1]] = 1
			
		image[3, self.home_location[0], self.home_location[1]] = 1
		categorical = np.full((1, self.grid_size, self.grid_size), num_carried_pellets, dtype=np.int16)
			
		return np.concatenate([image, categorical], axis=0)

  

	def _get_random_locations(self, n=1, excluding=None):
		"""
		Returns n random locations in the grid, excluding the given locations.		  
		
		Args:
		n (int): Number of locations to return. Defaults to 1.
		excluding (np.ndarray): Locations to exclude. If None, no locations are excluded. If a 1D array, it is
		interpreted as a single location. If a 2D array, it is interpreted as a list of locations. Defaults to None.
		
		Returns:
		np.ndarray: Array of shape (n, 2) containing the locations.
		"""
		
		# Create array of all grid locations
		grid_locations = np.array(np.meshgrid(range(self.grid_size), range(self.grid_size))).T.reshape(-1, 2)
		
		# Remove excluded locations
		if excluding is not None:
			if excluding.ndim == 1:
				excluding = excluding[None, :]
		
			grid_locations = grid_locations[~np.all(grid_locations[:, None, :] == excluding, axis=2).any(axis=1)]
		
		  
		# Sample n random locations
		return grid_locations[np.random.choice(np.arange(len(grid_locations)), size=n, replace=False)]

  

	@staticmethod
	def _action_to_direction(action):
	
		return {
			StealingGridworld.UP: np.array([-1, 0]),
			StealingGridworld.DOWN: np.array([1, 0]),
			StealingGridworld.LEFT: np.array([0, -1]),
			StealingGridworld.RIGHT: np.array([0, 1]),
		}[action]


	@staticmethod
	def _string_to_action(string):
	
		return {
			"u": StealingGridworld.UP,
			"d": StealingGridworld.DOWN,
			"l": StealingGridworld.LEFT,
			"r": StealingGridworld.RIGHT,
			"i": StealingGridworld.INTERACT,
		}[string.lower()[0]]


	@staticmethod
	def _action_to_string(action):
	
		return {
			StealingGridworld.UP: "UP",
			StealingGridworld.DOWN: "DOWN",
			StealingGridworld.LEFT: "LEFT",
			StealingGridworld.RIGHT: "RIGHT",
			StealingGridworld.INTERACT: "INTERACT",
		}[action]

  

	def render(self, state=None):
		"""Simple ASCII rendering of the environment."""
		
		if state is not None:
			prev_state = self._get_observation()
			self._register_state(state)
			self.render()
			self._register_state(prev_state)
			return
	
	  
	
		HOME = "H"
		OWNED_PELLET = "x"
		FREE_PELLET = "."
		AGENT = str(self.num_carried_pellets)
		
		grid = np.full((self.grid_size, self.grid_size), " ")
		grid[self.home_location[0], self.home_location[1]] = HOME
		grid[self.pellet_locations["owned"][:, 0], self.pellet_locations["owned"][:, 1]] = OWNED_PELLET
		grid[self.pellet_locations["free"][:, 0], self.pellet_locations["free"][:, 1]] = FREE_PELLET
				
		print("+" + "---+" * self.grid_size)
	
		for i in range(self.grid_size):
			print("|", end="")
		
			for j in range(self.grid_size):
				if self.agent_position[0] == i and self.agent_position[1] == j:
				print("{}{} |".format(AGENT, grid[i, j]), end="")
				else:
				print(" {} |".format(grid[i, j]), end="")
				
			print("\n+" + "---+" * self.grid_size)

  

	def render_rollout(self, rollout: TrajectoryWithRew):
	"""Render a rollout."""
	
		for state, action, reward in zip(rollout.obs, rollout.acts, rollout.rews):
			self._register_state(state)
			self.render()
			print("Action: {}".format(self._action_to_string(action)))
			print("Reward: {}".format(reward))
			print("")

  

	def repl(self, policy=None, optimal_qs=None):
	
		"""Simple REPL for the environment. Can optionally take a tabular policy and use it to select actions."""
		
		print("Welcome to the StealingGridworld REPL.")
		print("Use the following commands:")
		print(" u: Move up")
		print(" d: Move down")
		print(" l: Move left")
		print(" r: Move right")
		print(" i: Interact with environment")
		print(" q: Quit")
		print(" (no input): Ask policy for action, if policy vector is given")
		print("")
	
		self.reset()	
		total_reward = 0
		
		self.render()

		while True:
		
			if policy is not None:
				print("Policy action: {}".format(self._action_to_string(policy.predict(self._get_observation()))))
			
			if optimal_qs is not None:
				for action in range(self.action_space.n):
			
					print(
						"Q({}) = {}".format(
							self._action_to_string(action),
							optimal_qs[0][self.get_state_index(self._get_observation())][action],
							)
						)
			
			action = input("Action: ")
					
			if action.lower() == "q":
				break
			else:	
				try:
					action = StealingGridworld._string_to_action(action)
			
				except KeyError:
					print("Invalid action.")
				continue
			
				except IndexError:
					if policy is None:
						print("No policy given.")
						continue
					else:
						action = policy.predict(self._get_observation())
			
			obs, reward, done, _ = self.step(action)
			total_reward += reward
			print("Reward: {}".format(reward))
			print()
			self.render()


			if done:
				print("Episode done.")
				break

	print("Total reward: {}".format(total_reward))
```


#### Class initialization

```python
class StealingGridworld(gym.Env, DeterministicMDP):
	"""
	A gridworld in which the agent is rewarded for bringing home pellets, and punished for stealing pellets that belong to someone else.
	
	The agent starts at the home location, and can move up, down, left, or right. It also has an "interact" action,
	which handles picking up pellets and depositing them at home.
	
	"Free" pellets and "owned" pellets are distinguished. The agent can pick up either type of pellet, 
	but it will be punished for picking up an "owned" pellet (i.e. stealing). The agent can only 
	deposit pellets at home, and it will be rewarded for each pellet deposited.
	"""

	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	INTERACT = 4

  

	def __init__(
	self,
	grid_size=5, # Size of the grid 
	num_free_pellets=2, # Number of pellets that are free to pick up 
	num_owned_pellets=1, # Number of pellets that are owned and should not be stolen 
	reward_for_depositing=1, # Reward for depositing a pellet at home 
	reward_for_picking_up=0, # Reward for picking up a pellet (0 by default) 
	reward_for_stealing=-2, # Penalty for stealing an owned pellet 
	horizon=100, # Number of timesteps the game runs before it resets
	):

		# Store input parameters as instance variables
		self.grid_size = grid_size
		self.num_free_pellets = num_free_pellets
		self.num_owned_pellets = num_owned_pellets
		self.num_pellets = num_free_pellets + num_owned_pellets
		self.reward_for_depositing = reward_for_depositing
		self.reward_for_picking_up = reward_for_picking_up
		self.reward_for_stealing = reward_for_stealing
		  
		# Debugging purposes
		self.params_string = (
		f"gs{grid_size}_nfp{num_free_pellets}_nop{num_owned_pellets}_rfd{reward_for_depositing}"
		f"_rfp{reward_for_picking_up}_rfs{reward_for_stealing}"
		)

		self.horizon = horizon
		
		# Define the action space with 5 possible actions
		self.action_space = spaces.Discrete(5) # 0: up, 1: down, 2: left, 3: right, 4: interact

		# Categorical space for pellet count (+1 to account for zero pellets)
		self.categorical_spaces = [spaces.Discrete(self.num_pellets + 1)] # Number of states for pellet counts
		
		# Observation space is an image with 5 channels in c, corresponding to:
		# 1. Agent position (binary)
		# 2. Free pellet locations (binary)
		# 3. Owned pellet locations (binary)
		# 4. Home location (binary). This helps reward nets learn to go home.
		# 5. Carried pellets (number of carried pellets as an int, smeared across all pixels)

		# Define the observation space
		upper_bounds = np.ones((5, grid_size, grid_size)) # Given that ch = [0, 1, 2, 3] are binary, the upper bound is set by 1...
		upper_bounds[-1, :, :] = self.num_pellets # ... whereas for the last channel (carried pellets), the upper bound is the max number of pellets one can carry!
		self.observation_space = spaces.Box(
			low=np.array(np.zeros((5, self.grid_size, self.grid_size))),
			high=np.array(upper_bounds),	
			shape=(5, grid_size, grid_size),
			dtype=np.int16, # first four channels are binary, last channel is int (and small)	
		)
		
		if home_location is None:
			self.home_location = np.array([self.grid_size // 2, self.grid_size // 2])
		else:
			self.home_location = home_location
			
		self.reset()

```

- `num_owned_pellets` I assume is the number of wallets, whereas `num_free_pellets` is the number of coins.  

- `self.params_string` stores all info about the env in a unique string having format:
	- gs{`grid_size`};
	- nfp{`num_free_pellets`};
	- nop{`num_owned_pellets`};
	- rfd{`reward_for_depositing`};
	- rfp{`reward_for_picking_up`};
	- rfs{`reward_for_stealing`}.

	- Example:  `'gs5_nfp2_nop1_rfd100_rfp1_rfs-200'`

- `self.action_space` is the classic set of actions the agent can take in an environment. In this case, the actions are 4 + 1 : the four direction + an interaction with the environment action (to pick up pellets).

- The `categorical_spaces` variable in the `StealingGridworld` class is defined as follows: 

```python 
  self.categorical_spaces = [spaces.Discrete(self.num_pellets + 1)]
```
- 
	- This line declares a list containing a single Gym space. The space is a discrete space, which in the context of Gym represents a set of possible integer values a variable can take. Here, the variable in question is the count of pellets an agent can carry across the grid. The discrete space is initialized to have a range from `0` to `self.num_pellets`, which is the sum of `num_free_pellets` and `num_owned_pellets`.
	 The `+1` in the discrete space definition ensures that the range includes `0`, representing scenarios where the agent carries no pellets. The categorical space can therefore represent all possible states of pellet possession, from carrying none to carrying the maximum possible number defined in the environment. 

- `gym.spaces.Box` ([link](https://www.gymlibrary.dev/api/spaces/#box)) is a class provided by the OpenAI Gym library, which is commonly used to define a continuous multidimensional space of allowable states or actions in a reinforcement learning environment. Each dimension of the space can have different limits, allowing for the creation of a "box" in a high-dimensional space where each corner and edge represent the maximum or minimum value for that dimension. 
##### Parameters of `gym.spaces.Box`

- **low**: This parameter specifies the lower boundary of the box for each dimension. This can be a scalar (applying the same lower boundary to all dimensions) or an array that specifies the lower boundary for each dimension specifically. In our case, it's a 3D numpy array (5 x `grid_size` x `grid_size`), initialized at 0. The 5 channels are explained in the snippet above, namely:
	1. Agent position ${0,1}$, i.e., on-cell/off-cell
	2. Free pellet locations ${0,1}$, i.e., on-cell/off-cell
	3. Owned pellet locations ${0,1},$ i.e., on-cell/off-cell
	4. Home location ${0,1}$, i.e., on-cell/off-cell. <u>This helps reward nets learn to go home.</u>
	5. Carried pellets, $\{0,1,2,3,...,$ `num_pellets`$\}$, i.e., `num_pellets` $+ 1$ (number of carried pellets as an int, smeared across all pixels)
    
- **high**: This parameter sets the upper boundary of the box. Like `low`, it can be a scalar or an array defining the upper boundary for each dimension. In our case, once again, it's a 3D numpy array initialized at $1, 1, 1, 1$, for the binaries $[1,4]$, and `num_pellets` for the last channel
    
- **shape**: This optional parameter describes the shape of the space, i.e., the dimensions of the array that will be used to represent the space. If `low` and `high` are already arrays, the shape is inferred from them, so specifying it explicitly is not necessary.
    
- **dtype**: This parameter specifies the data type of the elements within the space.

> [!caution] 
> The `spaces.Discrete` object is initialized as `num_pellets + 1` to include 0, whereas the number of carried pellets starts from 0 (no pellets carried) to a maximum of `num_pellets`, thus resulting in `num_pellets + 1` states.
> 

- Finally, if the user predefines a `home_location`, then the pot is set there, otherwise it is assigned to the middle of the grid.

### Helper functions

```python
def separate_image_and_categorical_state(
	state: np.ndarray,
	categorical_spaces: Iterable[spaces.Discrete],
	) -> Tuple[np.ndarray, Iterable[np.ndarray]]:

	"""
	Separate the image and categorical components of the state.
	
	Args:
	state: A preprocessed batch of states.
	categorical_spaces: The spaces of the categorical components of the state.
	
	Returns:
	A tuple of (image, categorical), where `image` is a batch of images and `categorical` is a list of batches
	of categorical data, one-hot encoded.
	"""
	
	_, total_channels, _, _ = state.shape
	image_channels = total_channels - len(categorical_spaces)
	image = state[:, :image_channels]
	categorical = []
	for i, space in enumerate(categorical_spaces):
	category_values = state[:, image_channels + i, 0, 0] # Smeared across all pixels; just take one.
	categorical.append(np.eye(space.n)[category_values])
	
	return image, categorical
```

### Actions on the environment/agent side

```python
def enumerate_actions(self):
	"""
	Returns a list of all possible actions in the environment.
	"""
	return list(range(self.action_space.n))

  

def encode_state(self, state):
	"""
	Encodes state as a string.
	"""
	return ",".join(state.flatten().astype(str))

  

def encode_action(self, action):
	"""
	Encodes action as a string.
	"""
	return str(action)

  

def encode_mdp_params(self):
	return "SG_" + self.params_string

  

def _register_state(self, state):
	"""
	Set the current environment state to the given state.
	"""
	
	image, categorical = separate_image_and_categorical_state(np.array([state]), self.categorical_spaces)
	image = image[0] # Remove batch dimension
	categorical = categorical[0][0] # Remove batch dimension AND categorical dimension (only one categorical var)
	
	self.agent_position = np.array(np.where(image[0, :, :] == 1)).T[0]
	self.pellet_locations = {
		"free": np.array(np.where(image[1, :, :] == 1)).T,
		"owned": np.array(np.where(image[2, :, :] == 1)).T,
	}
	
	self.num_carried_pellets = np.where(categorical == 1)[0][0]

  

def _pick_up_pellet(self, pellet_type):

	"""
	Removes pellet from corresponding pellet list, and increments the number of carried pellets.
	
	Args:
	pellet_type (str): Either "free" or "owned".
	"""
	
	pellet_locations = self.pellet_locations[pellet_type]
	pellet_indices = np.where(np.all(pellet_locations == self.agent_position, axis=1))[0]
	
	assert len(pellet_indices) > 0, "No pellets at agent location, but _pick_up_pellet called."
	assert len(pellet_indices) <= 1, "Multiple pellets at same location."
	
	self.pellet_locations[pellet_type] = np.delete(pellet_locations, pellet_indices[0], axis=0)
	self.num_carried_pellets += 1

  

def _get_observation(self):

	return self._get_observation_from_state_components(
	self.agent_position,
	self.pellet_locations["free"],
	self.pellet_locations["owned"],
	self.num_carried_pellets,
	)
```


```python
def reset(self, seed=None):
	super().reset(seed=seed)

	# Agent starts at home.
	self.agent_position = self.home_location.copy() # Copy the home location to set the initial position of the agent.

	# Initialize pellet locations. They must be unique and not on the agent. 
	pellet_locations = self._get_random_locations(n=self.num_pellets, excluding=self.agent_position) 
	# Generate random, unique locations for pellets, avoiding the agent's start position. 
	self.pellet_locations = { 
		"free": pellet_locations[:self.num_free_pellets], # Assign the first part of the array to 'free' pellets.
		"owned": pellet_locations[self.num_free_pellets:], # Assign the remaining part to 'owned' pellets. }
	}
	
	self.num_carried_pellets = 0 # Initialize the number of pellets the agent is carrying to zero.
	self.steps = 0 # Reset the step counter for the new episode.
	
	return self._get_observation() # Return the initial observation of the environment.
```

- **`self.agent_position`**: Stores the current position of the agent on the grid.
- **`pellet_locations`**: Temporarily holds potential positions for both free and owned pellets.
- **`self.pellet_locations`**: Dictionary storing arrays of positions for free and owned pellets separately.
- **`self.num_carried_pellets`**: Counter for the pellets currently carried by the agent.
- **`self.steps`**: Counter for the number of actions taken in the current episode.

```python
def step(self, action): 

	self.steps += 1 # Increment the step counter for each action taken. 
	# Default reward 
	reward = 0 # Initialize reward for this step to zero. 
	
	# Move agent 
	if action < StealingGridworld.INTERACT: # Check if the action is a movement action. 
		direction = self._action_to_direction(action) # Convert the action to a direction vector. 
		new_position = self.agent_position + direction # Calculate the new potential position. 
			if np.all(new_position >= 0) and np.all(new_position < self.grid_size): # Check if the new position is within grid bounds.
				self.agent_position = new_position # Update the agent's position if the move is valid. 
				
	# Interact with environment 			
	elif np.any(np.all(self.pellet_locations["free"] == self.agent_position, axis=1)): 
		reward = self.reward_for_picking_up # Set reward for picking up a free pellet. 
		self._pick_up_pellet(pellet_type="free") # Execute pellet pickup logic for free pellets. 
	
	elif np.any(np.all(self.pellet_locations["owned"] == self.agent_position, axis=1)): 
		reward = self.reward_for_stealing # Set penalty for stealing an owned pellet. 
		self._pick_up_pellet(pellet_type="owned") # Execute pellet pickup logic for owned pellets. 
	
	elif np.all(self.agent_position == self.home_location) and self.num_carried_pellets > 0:
		 reward = self.num_carried_pellets * self.reward_for_depositing # Calculate reward based on number of pellets deposited. 
		 self.num_carried_pellets = 0 # Reset the carried pellet count after depositing. 
		 
	# Compute done 
	done = self.steps >= self.horizon # Check if the episode should end based on the number of steps taken. 

	# Return the new state of the environment, the reward, whether the episode is done, and additional info (empty here).
	return self._get_observation(), reward, done, {}
```

- **`action`**: The action performed by the agent, which can be one of the directional moves or an interact action.
- **`direction`**: A vector indicating the direction of movement based on the action.
- **`new_position`**: The potential new position of the agent after the move.
- **`reward`**: The reward obtained from the current step's action.
- **`done`**: A boolean indicating whether the episode has ended.

```python
def successor(self, state, action): 
	""" Returns the successor state and reward for a given state and action. """ 
	prev_state = self._get_observation() # Get the current observation to save the current state. 
	self._register_state(state) # Set the environment's state to the specified state. 
	successor_state, reward, _, _ = self.step(action) # Perform the action to get the new state and reward. 
	self._register_state(prev_state) # Restore the original state to maintain state consistency after the operation. 
	return successor_state, reward # Return the resulting state and reward.
```

