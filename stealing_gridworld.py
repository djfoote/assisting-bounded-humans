import itertools
from typing import Iterable, Tuple

import gymnasium as gym
import numpy as np
import tqdm
from gymnasium import spaces

from imitation.data.types import TrajectoryWithRew

from deterministic_mdp import DeterministicMDP
from imitation_modules import ObservationFunction


class StealingGridworld(gym.Env, DeterministicMDP):
    """
    A gridworld in which the agent is rewarded for bringing home pellets, and punished for stealing pellets that belong
    to someone else.

    The agent starts at the home location, and can move up, down, left, or right. It also has an "interact" action,
    which handles picking up pellets and depositing them at home.

    "Free" pellets and "owned" pellets are distinguished. The agent can pick up either type of pellet, but it will be
    punished for picking up an "owned" pellet (i.e. stealing). The agent can only deposit pellets at home, and it will
    be rewarded for each pellet deposited.
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    INTERACT = 4

    def __init__(
        self,
        grid_size=3,
        num_free_pellets=2,
        num_owned_pellets=2,
        reward_for_depositing=1,
        reward_for_picking_up=0,
        reward_for_stealing=-2,
        home_location=None,
        randomize=False,
        horizon=100,
        reward_fn=None,
        seed=None,
        num_envs=1,
        **kwargs,
    ):
        self.reward_for_depositing = reward_for_depositing
        self.reward_for_picking_up = reward_for_picking_up
        self.reward_for_stealing = reward_for_stealing
        self.alt_reward_fn = reward_fn
        self.randomize = randomize
        self.seed = seed
        if self.randomize:
            print('Using randomized environment')
            assert self.seed is not None, "Randomized environment requires a seed"
            # set random grid (uneven valued) grid size, pellet count, and reward values
            self.grid_size = np.random.choice(range(3, 10, 2))
            grid_size = self.grid_size
            self.num_free_pellets = np.random.choice(range(1, 5))
            self.num_owned_pellets = np.random.choice(range(1, 5))
        else:
            self.grid_size = grid_size
            self.num_free_pellets = num_free_pellets
            self.num_owned_pellets = num_owned_pellets
        self.num_pellets = num_free_pellets + num_owned_pellets
        
        print('Using the following environment parameters:')
        print(f'Grid size: {self.grid_size}')
        print(f'Number of free pellets: {self.num_free_pellets}')
        print(f'Number of owned pellets: {self.num_owned_pellets}')
            

        self.params_string = (
            f"gs{grid_size}_nfp{num_free_pellets}_nop{num_owned_pellets}_rfd{reward_for_depositing}"
            f"_rfp{reward_for_picking_up}_rfs{reward_for_stealing}"
        )

        self.horizon = horizon

        self.action_space = spaces.Discrete(5)  # 0: up, 1: down, 2: left, 3: right, 4: interact
        

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
            dtype=np.int16,  # first four channels are binary, last channel is int (and small)
        )

        if home_location is None:
            self.home_location = np.array([self.grid_size // 2, self.grid_size // 2])
        else:  
            self.home_location = home_location
        
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.num_envs = num_envs
        self._actions = None

        self.reset()

    def reset(self, seed=None, options=None):
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

        return self._get_observation(), {}

    def alt_reward(self, state, action):
        # state = state.reshape((5, self.grid_size, self.grid_size))
        state_inputs = np.repeat(state, len(np.array([action])), axis=0)
        state_inputs = state_inputs[np.newaxis, :]
        reward = self.alt_reward_fn.predict(
                state=state_inputs,
                action=action,
                next_state=state_inputs,
                done=np.zeros_like(state, dtype=bool),
            )
        return reward[0]
    
    def step_async(self, actions):
        """Store the actions given by the vectorized environment to be executed later."""
        self._actions = [actions] if actions.ndim == 1 else actions

    def step_wait(self):
        """Execute the actions stored by step_async."""
        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait")

        observations, rewards, dones, infos = [], [], [], []
        # Assume self._actions is a list of actions for each parallel environment instance
        for action in self._actions:
            observation, reward, done, info = self.step(action)
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return observations, rewards, dones, infos

    def step(self, action, sample=True):
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
            reward = self.reward_for_picking_up if sample else self.alt_reward(self._get_observation(), action)
            self._pick_up_pellet(pellet_type="free")
        elif np.any(np.all(self.pellet_locations["owned"] == self.agent_position, axis=1)):
            reward = self.reward_for_stealing if sample else self.alt_reward(self._get_observation(), action)
            self._pick_up_pellet(pellet_type="owned")
        elif np.all(self.agent_position == self.home_location) and self.num_carried_pellets > 0:
            reward = (self.num_carried_pellets * self.reward_for_depositing) if sample else self.alt_reward(self._get_observation(), action)
            self.num_carried_pellets = 0
        # Compute done
        done = self.steps >= self.horizon
        # sb3 requires both truncated and done
        # truncated is typically for if the goal is reached so...
        # TODO: implement a goal?
        truncated = done
        return self._get_observation(), reward, truncated, done, {}


    def successor(self, state, action):
        """
        Returns the successor state and reward for a given state and action.
        """
        prev_state = self._get_observation()
        self._register_state(state)
        successor_state, reward, _, _, _ = self.step(action, sample=True)
        self._register_state(prev_state)
        return successor_state, reward

    def reward(self, state, action):
        """
        Returns the reward for a given state and action.
        """
        self._register_state(state)
        _, reward, _, _, _ = self.step(action, sample=True)

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

    # def enumerate_states(self):
    #     """
    #     Returns a list of all possible states in the environment.
    #     """
    #     states = []
    #     grid_size_squared = self.grid_size ** 2
    #     home_location_raveled = np.ravel_multi_index(self.home_location, (self.grid_size, self.grid_size))

    #     agent_positions = np.array(list(itertools.product(range(self.grid_size), repeat=2)))
    #     num_carried_pellets_range = np.arange(self.num_pellets + 1)
    #     num_loose_pellets_range = np.arange(self.num_pellets + 1)

    #     # Precompute all possible combinations of pellet locations
    #     all_pellet_combinations = []
    #     for num_loose_pellets in reversed(num_loose_pellets_range):
    #         all_pellet_combinations.append(np.array(list(itertools.combinations(range(grid_size_squared - 1), num_loose_pellets)), dtype=np.int32))

    #     for agent_position in tqdm.tqdm(agent_positions, desc="Enumerating states"):
    #         for num_carried_pellets in num_carried_pellets_range:
    #             for num_loose_pellets in reversed(range(self.num_pellets - num_carried_pellets + 1)):
    #                 pellet_combinations = all_pellet_combinations[num_loose_pellets]

    #                 # Exclude home location
    #                 pellet_combinations[pellet_combinations >= home_location_raveled] += 1

    #                 min_loose_free_pellets = max(0, num_loose_pellets - self.num_owned_pellets)
    #                 for num_loose_free_pellets in range(min_loose_free_pellets, self.num_free_pellets + 1):
    #                     indices_combinations = np.array(list(itertools.combinations(range(pellet_combinations.shape[1]), num_loose_free_pellets)))

    #                     for pellet_locations in pellet_combinations:
    #                         for indices in indices_combinations:
    #                             free_pellet_locations_raveled = pellet_locations[list(indices)]
    #                             owned_pellet_locations_raveled = np.delete(pellet_locations, list(indices))
    #                             free_pellet_locations = np.array(np.unravel_index(free_pellet_locations_raveled, (self.grid_size, self.grid_size))).T
    #                             owned_pellet_locations = np.array(np.unravel_index(owned_pellet_locations_raveled, (self.grid_size, self.grid_size))).T

    #                             state = self._get_observation_from_state_components(
    #                                 agent_position=np.array(agent_position),
    #                                 num_carried_pellets=num_carried_pellets,
    #                                 free_pellet_locations=free_pellet_locations,
    #                                 owned_pellet_locations=owned_pellet_locations,
    #                             )
    #                             states.append(state)

    #     return states


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
        """
        Encodes MDP parameters as a string.
        """
        return "SG_" + self.params_string

    def _register_state(self, state):
        """
        Set the current environment state to the given state.
        """
        image, categorical = separate_image_and_categorical_state(np.array([state]), self.categorical_spaces)
        image = image[0]  # Remove batch dimension
        categorical = categorical[0][0]  # Remove batch dimension AND categorical dimension (only one categorical var)

        #self.agent_position = np.array(np.where(image[0, :, :] == 1)).T[0]
        self.agent_position = np.array(np.where(image[0, :, :] == 1)).T

        if self.agent_position.size == 0:
            self.agent_position = None  # Or a default position if appropriate, e.g., [-1, -1]
        else:
            self.agent_position = self.agent_position[0]

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
        image[0, agent_position[0], agent_position[1]] = np.array([1], dtype=np.int16)
        for pellet_location in free_pellet_locations:
            image[1, pellet_location[0], pellet_location[1]] = np.array([1], dtype=np.int16)
        for pellet_location in owned_pellet_locations:
            image[2, pellet_location[0], pellet_location[1]] = np.array([1], dtype=np.int16)
        image[3, self.home_location[0], self.home_location[1]] = np.array([1], dtype=np.int16)
        categorical = np.full((1, self.grid_size, self.grid_size), num_carried_pellets, dtype=np.int16)
        return np.concatenate([image, categorical], axis=0)

    def _get_random_locations(self, n=1, excluding=None):
        """
        Returns n random locations in the grid, excluding the given locations.

        Args:
            n (int): Number of locations to return. Defaults to 1.
            excluding (np.ndarray): Locations to exclude. If None, no locations are excluded. If a 1D array, it is
                interpreted as a single location. If a 2D array, it is interpreted as a list of locations. Defaults to
                None.

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
            StealingGridworld.INTERACT: np.array([0, 0]),
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

    # def render(self, state=None):
    #     """Simple ASCII rendering of the environment."""
    #     if state is not None:
    #         prev_state = self._get_observation()
    #         self._register_state(state)
    #         self.render()
    #         self._register_state(prev_state)
    #         return

    #     HOME = "H"
    #     OWNED_PELLET = "x"
    #     FREE_PELLET = "."
    #     AGENT = str(self.num_carried_pellets)

    #     grid = np.full((self.grid_size, self.grid_size), " ")
    #     grid[self.home_location[0], self.home_location[1]] = HOME
    #     grid[self.pellet_locations["owned"][:, 0], self.pellet_locations["owned"][:, 1]] = OWNED_PELLET
    #     grid[self.pellet_locations["free"][:, 0], self.pellet_locations["free"][:, 1]] = FREE_PELLET

    #     print("+" + "---+" * self.grid_size)
    #     for i in range(self.grid_size):
    #         print("|", end="")
    #         for j in range(self.grid_size):
    #             if self.agent_position[0] == i and self.agent_position[1] == j:
    #                 print("{}{} |".format(AGENT, grid[i, j]), end="")
    #             else:
    #                 print(" {} |".format(grid[i, j]), end="")
    #         print("\n+" + "---+" * self.grid_size)

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
                # Check if agent_position is not None and coordinates match
                if self.agent_position is not None and self.agent_position[0] == i and self.agent_position[1] == j:
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
        print("  u: Move up")
        print("  d: Move down")
        print("  l: Move left")
        print("  r: Move right")
        print("  i: Interact with environment")
        print("  q: Quit")
        print("  (no input): Ask policy for action, if policy vector is given")
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

            obs, reward, done, _, _ = self.step(action, sample=True)
            total_reward += reward
            print("Reward: {}".format(reward))
            print()
            self.render()

            if done:
                print("Episode done.")
                break

        print("Total reward: {}".format(total_reward))


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
        category_values = state[:, image_channels + i, 0, 0]  # Smeared across all pixels; just take one.
        categorical.append(np.eye(space.n)[category_values])
    return image, categorical


# class PartialGridVisibility(ObservationFunction):
#     def __init__(self, env: StealingGridworld, visibility_mask=None ,feedback="scalar"):
#         self.env = env

#         if visibility_mask is None:
#             # Default visibility mask: everything except the outermost ring.
#             if env.grid_size < 3:
#                 raise ValueError(
#                     "Grid size must be at least 3 for default partial visibility. "
#                     "Increase grid size or specify visibility mask explicitly."
#                 )
#             visibility_mask = np.ones((env.grid_size, env.grid_size), dtype=np.bool)
#             visibility_mask[0, :] = visibility_mask[-1, :] = visibility_mask[:, 0] = visibility_mask[:, -1] = False

#         self.visibility_mask = visibility_mask
#         self.feedback = feedback

#     def __call__(self, fragment):
#         # Mask out all but the number of pellets the agent is carrying (spatial representation is fake in that case).
#         if self.feedback == "scalar":
#             masked_obs = fragment.obs[:, :-1] * self.visibility_mask[np.newaxis, np.newaxis]
#             new_obs = np.concatenate([masked_obs, fragment.obs[:, -1:]], axis=1)

#             # For timesteps where the agent is not in the visibility mask, set reward to 0.
#             agent_visible = new_obs[:-1, 0].any(axis=(1, 2))
#             new_rew = fragment.rews * agent_visible

#             return TrajectoryWithRew(new_obs, fragment.acts, fragment.infos, fragment.terminal, new_rew)
        
#         elif self.feedback == "preference":
#             fragments = []
#             for f in fragment: 
#                 masked_obs = f.obs[:, :-1] * self.visibility_mask[np.newaxis, np.newaxis]
#                 new_obs = np.concatenate([masked_obs, f.obs[:, -1:]], axis=1)
#                 agent_visible = new_obs[:-1, 0].any(axis=(1, 2))
#                 new_rew = f.rews * agent_visible
#                 fragments.append(TrajectoryWithRew(new_obs, f.acts, f.infos, f.terminal, new_rew))
            
#             return (fragments[0], fragments[1])

#         else:
#             raise ValueError(
#                     "Unsupported feedback type for Partial Observability"
#                     "Choose between 'scalar' and 'preference'."
#                 )


#             #     # For timesteps where the agent is not in the visibility mask, set reward to 0.
#             #     agent_visible = new_obs[:-1, 0].any(axis=(1, 2))
#             #     new_rew = f.rews * agent_visible


#             #     traj = self.rng.choice(
#             #     trajectories,  # type: ignore[arg-type]
#             #     p=np.array(weights) / sum(weights),
#             #     )
#             #     n = len(traj)
#             #     start = self.rng.integers(0, n - fragment_length, endpoint=True)
#             #     end = start + fragment_length
#             #     terminal = (end == n) and traj.terminal
#             #     fragment = TrajectoryWithRew(
#             #         obs=traj.obs[start : end + 1],
#             #         acts=traj.acts[start:end],
#             #         infos=traj.infos[start:end] if traj.infos is not None else None,
#             #         rews=traj.rews[start:end],
#             #         terminal=terminal,
#             #     )
#             #     fragments.append(fragment)
#             # # fragments is currently a list of single fragments. We want to pair up
#             # # fragments to get a list of (fragment1, fragment2) tuples. To do so,
#             # # we create a single iterator of the list and zip it with itself:
#             # iterator = iter(fragments)
#             # return list(zip(iterator, iterator))


class PartialGridVisibility(ObservationFunction):

    # TODO : Embed the visibility mask in the observation function
    # in order to make it more general and reusable. Also avoids having to pass it as an argument.
    # The possible visibility masks for now are the default one, and a camera-like one.

    def __init__(self, env: StealingGridworld, mask_key = None, feedback="scalar", *args, **kwargs):
        self.env = env
        self.grid_size = env.grid_size
        self.visibility_mask = self.construct_visibility_mask(mask_key)
        self.feedback = feedback

    def construct_visibility_mask(self, visibility_mask_key, center=None):
        # Any other visibility mask keys should be added here.
        if visibility_mask_key == "(n-1)x(n-1)":
            visibility_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.bool_)
            visibility_mask[1:-1, 1:-1] = True
            return visibility_mask
        else:
            raise ValueError(f"Unknown visibility mask key {visibility_mask_key}.")

    def __call__(self, fragments):
        if self.feedback == "scalar":
            return self.process_scalar_feedback(fragments)
        elif self.feedback == "preference":
            return self.process_preference_feedback(fragments)

    def process_scalar_feedback(self, fragment):
        masked_obs = fragment.obs[:, :-1] * self.visibility_mask[np.newaxis, np.newaxis] # apply the visibility mask to all channels except the last one
        new_obs = np.concatenate([masked_obs, fragment.obs[:, -1:]], axis=1) # concatenate the masked obs with the last channel
        agent_visible = new_obs[:-1, 0].any(axis=(1, 2))
        new_rew = fragment.rews * agent_visible
        return TrajectoryWithRew(new_obs, fragment.acts, fragment.infos, fragment.terminal, new_rew)

    def process_preference_feedback(self, fragment_pair):
        processed_fragments = []
        for fragment in fragment_pair:

            # reshape flattened observation to 3D if needed
            if len(fragment.obs.shape) == 2: # loophole to handle the case where the observation is flattened by PPO
                observation = fragment.obs.reshape(-1, 5, self.grid_size, self.grid_size)
                masked_obs = observation[:, :-1] * self.visibility_mask[np.newaxis, np.newaxis]
                new_obs = np.concatenate([masked_obs, observation[:, -1:]], axis=1)
            else:
                masked_obs = fragment.obs[:, :-1] * self.visibility_mask[np.newaxis, np.newaxis]
                new_obs = np.concatenate([masked_obs, fragment.obs[:, -1:]], axis=1)
            
            
            agent_visible = new_obs[:-1, 0].any(axis=(1, 2))
            new_rew = fragment.rews * agent_visible
            processed_fragments.append(TrajectoryWithRew(new_obs, fragment.acts, fragment.infos, fragment.terminal, new_rew))
        return tuple(processed_fragments)
    
    def __repr__(self):
        return f"PartialGridVisibility(grid_size={self.grid_size}, visibility_mask={self.visibility_mask}, feedback={self.feedback})"
        

class DynamicGridVisibility(ObservationFunction):
    def __init__(self, env: StealingGridworld, pattern=None, feedback="scalar"):
        super().__init__()
        self.env = env
        self.grid_size = env.grid_size
        self.feedback = feedback
        
        # Define the pattern of camera movement
        if pattern is None:
            self.pattern = self.default_pattern()
        else:
            self.pattern = pattern
        print("Pattern = ", self.pattern)
        self.pattern_index = 0  # Start at the first position in the pattern

        # Build the initial visibility mask
        self.visibility_mask = self.construct_visibility_mask()

    def default_pattern(self):
        # Create a default movement pattern for the camera
        # Example for a 5x5 grid, you may adjust as needed
        positions = []
        # HARDCODED, TODO find a way to generalize this
        if self.grid_size == 3:
            positions = [(0,0), (0,1), (1,1), (1,0)]
        elif self.grid_size == 5:
            positions = [(0,0), (0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0)]
        else:
            raise NotImplementedError("Default pattern not implemented for grid size other than 3x3 or 5x5")
        return positions

    def reset(self):
        self.pattern_index = 0
        self.visibility_mask = self.construct_visibility_mask()
    
    def construct_visibility_mask(self):
        # Build a visibility mask based on the current pattern index
        mask = np.zeros((self.grid_size, self.grid_size), dtype=np.bool_)
        left_x, left_y = self.pattern[self.pattern_index]
        camera_size = self.grid_size // 2 + self.grid_size % 2
        
        # Calculate bounds of the camera window
        end_x = min(left_x + camera_size, self.grid_size)
        end_y = min(left_y + camera_size, self.grid_size)
        #print("start_x, end_x, start_y, end_y = ", start_x, end_x, start_y, end_y)
        mask[left_x:end_x, left_y:end_y] = True
        return mask

    def update_visibility(self, t=None, limits=None):
        # Update the visibility mask for the next timestep
        if t is not None and limits is not None: # For the training case
            # collect t visibility masks and return a list of them
            visibility_masks = []
            self.reset()

            self.pattern_index = limits[0]
            for _ in range(t):
                self.pattern_index = (self.pattern_index) % len(self.pattern)
                visibility_masks.append(self.construct_visibility_mask())
                self.pattern_index += 1
            return np.array(visibility_masks)
        elif t is not None: # For the evaluation case
            visibility_masks = []
            self.reset()
            for _ in range(t):
                self.pattern_index = (self.pattern_index) % len(self.pattern)
                visibility_masks.append(self.construct_visibility_mask())
                self.pattern_index += 1
            return np.array(visibility_masks)
        else: # Not sure if this case is needed
            self.pattern_index = (self.pattern_index) % len(self.pattern)
            self.visibility_mask = self.construct_visibility_mask()
            self.pattern_index += 1

    def __call__(self, fragments, limits=None):
        if self.feedback == "scalar":
            return self.process_scalar_feedback(fragments, limits)
        elif self.feedback == "preference":
            return self.process_preference_feedback(fragments, limits)


    def __repr__(self):
        return f"DynamicGridVisibility(\n    grid_size={self.grid_size},\n    visibility_mask=\n{self.visibility_mask},\n    feedback={self.feedback}\n)"

    def process_scalar_feedback(self, fragment, limits=None):
        visibility_mask = self.update_visibility()
        masked_obs = fragment.obs[:, :-1] * visibility_mask[np.newaxis, np.newaxis]
        new_obs = np.concatenate([masked_obs, fragment.obs[:, -1:]], axis=1)
        agent_visible = new_obs[:, 0, :, :].any(axis=(1, 2))
        new_rew = fragment.rews * agent_visible
        return TrajectoryWithRew(new_obs, fragment.acts, fragment.infos, fragment.terminal, new_rew)
    
    def render_mask(self, mask):
        """ Render the visibility mask as ASCII in a more detailed and visually consistent manner. """
        print("+" + "---+" * len(mask))
        for row in mask:
            print("|", end="")
            for cell in row:
                if cell:
                    print(" # |", end="")  # Visible areas marked with '#'
                else:
                    print("   |", end="")  # Non-visible areas remain blank
            print("\n+" + "---+" * len(mask))

    def process_preference_feedback(self, fragment_pair, limits=None):
        # limits is a list of ((start,end), (start,end)) tuples for a pair of fragments
        # Extract f1 limits from the limits list
        limits_f1 = limits[0]
        limits_f2 = limits[1]

        # Put vsibility mask in a dictionary
        visibility_masks_f1 = self.update_visibility(t=len(fragment_pair[0].obs), limits=limits_f1)
        visibility_masks_f2 = self.update_visibility(t=len(fragment_pair[1].obs), limits=limits_f2)
        masks = {
            0: visibility_masks_f1,
            1: visibility_masks_f2
        }
        processed_fragments = []
        for i, fragment in enumerate(fragment_pair):
            new_obs = []
            new_rews = []
            visibility_masks = masks[i]
            # visibility_masks has shape (t, grid_size, grid_size)
            # fragment.obs has shape (t, 5, grid_size, grid_size)
            # We need to apply the visibility mask to all channels except the last one
            # The last channel is the number of carried pellets, which should not be masked

            # print("Visibility masks for fragment {}:".format(i+1), "over the time steps:")
            # for t in range(len(visibility_masks)):
            #     self.render_mask(visibility_masks[t])

            # Debug: find agent position in the observation
            agent_pos = np.where(fragment.obs[:, 0, :, :] == 1)
            time_steps = agent_pos[0]
            x_coords = agent_pos[1]
            y_coords = agent_pos[2]

            # print("Agent positions at time t:")
            # start = limits_f1[0]
            # j = 0
            # for t, x, y in zip(time_steps, x_coords, y_coords):
            #     print(f"t = {t} (l = {start + j}) (x={x}, y={y})")
            #     j += 1

            

            masked_obs = fragment.obs[:, :-1] * visibility_masks[:, np.newaxis]
        
            new_obs = np.concatenate([masked_obs, fragment.obs[:, -1:]], axis=1)

            # Debug: find agent position in the new observation
            agent_pos = np.where(new_obs[:, 0, :, :] == 1)
            time_steps = agent_pos[0]
            x_coords = agent_pos[1]
            y_coords = agent_pos[2]

            # print("Agent positions at time t:")
            # start = limits_f1[0]
            # j = 0
            # for t, x, y in zip(time_steps, x_coords, y_coords):
            #     print(f"t = {t} (l = {start + j}) (x={x}, y={y})")
            #     j += 1

            # apply the reward modification
            # agent_visible = new_obs[:-1, 0].any(axis=(1, 2))
            # new_rews = fragment.rews * agent_visible           
            
            # # Store the new observations and rewards in the trajectory data structure
            # new_fragment = TrajectoryWithRew(np.array(new_obs), fragment.acts, fragment.infos, fragment.terminal, np.array(new_rews))
            # processed_fragments.append(new_fragment)
            # Debug: Render the observations before and after applying the mask
            #print("Fragment {} Before Masking:".format(i+1))
            #self.env.render_rollout(TrajectoryWithRew(fragment.obs, fragment.acts, fragment.infos, fragment.terminal, fragment.rews))
            #print("Fragment {} After Masking:".format(i+1))
            #self.env.render_rollout(TrajectoryWithRew(new_obs, fragment.acts, fragment.infos, fragment.terminal, fragment.rews))

            agent_visible = new_obs[:-1, 0].any(axis=(1, 2))
            new_rews = fragment.rews * agent_visible
            new_fragment = TrajectoryWithRew(np.array(new_obs), fragment.acts, fragment.infos, fragment.terminal, np.array(new_rews))
            processed_fragments.append(new_fragment)

        return tuple(processed_fragments)

    
    def __repr__(self):
        return f"DynamicGridVisibility(pattern={self.pattern}, feedback={self.feedback})"