import itertools
import json

import gymnasium as gym
import numpy as np
import tqdm
from gymnasium import spaces
from imitation.data.types import TrajectoryWithRew

import utils
from deterministic_mdp import DeterministicMDP
from imitation_modules import DeterministicMDPTrajGenerator


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

    # TODO: make these configurable
    REWARD_FOR_DEPOSITING_PELLET = 1
    REWARD_FOR_STEALING = -2

    def __init__(self, grid_size=5, num_free_pellets=2, num_owned_pellets=1, max_steps=100):
        self.grid_size = grid_size
        self.num_free_pellets = num_free_pellets
        self.num_owned_pellets = num_owned_pellets
        self.num_pellets = num_free_pellets + num_owned_pellets
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(5)  # 0: up, 1: down, 2: left, 3: right, 4: interact

        self.categorical_spaces = [spaces.Discrete(self.num_pellets + 1)]

        # Observation space is an image with 4 channels in c, corresponding to:
        # 1. Agent position (binary)
        # 2. Free pellet locations (binary)
        # 3. Owned pellet locations (binary)
        # 4. Carried pellets (number of carried pellets as an int, smeared across all pixels)
        upper_bounds = np.ones((4, grid_size, grid_size))
        upper_bounds[3, :, :] = self.num_pellets
        self.observations = spaces.Box(
            low=np.array(np.zeros((4, self.grid_size, self.grid_size))),
            high=np.array(upper_bounds),
            shape=(4, grid_size, grid_size),
            dtype=np.int16,  # first three channels are binary, last channel is int (and small)
        )

        # TODO: make this configurable
        self.home_location = np.array([0, 0])

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
            self._pick_up_pellet(pellet_type="free")
        elif np.any(np.all(self.pellet_locations["owned"] == self.agent_position, axis=1)):
            reward = self.REWARD_FOR_STEALING
            self._pick_up_pellet(pellet_type="owned")
        elif np.all(self.agent_position == self.home_location) and self.num_carried_pellets > 0:
            reward = self.num_carried_pellets * self.REWARD_FOR_DEPOSITING_PELLET
            self.num_carried_pellets = 0

        # Compute done
        done = self.steps >= self.max_steps

        return self._get_observation(), reward, done, {"state": self._get_state()}

    def successor(self, state, action):
        """
        Returns the successor state and reward for a given state and action.
        """
        prev_state = self._get_state()
        self._register_state(state)
        _, reward, _, _ = self.step(action)
        successor_state = self._get_state()
        self._register_state(prev_state)
        return successor_state, reward

    def reward(self, state, action):
        """
        Returns the reward for a given state and action.
        """
        prev_state = self._get_state()
        self._register_state(state)
        _, reward, _, _ = self.step(action)
        self._register_state(prev_state)
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
                                state = {
                                    "agent_position": np.array(agent_position),
                                    "free_pellet_locations": free_pellet_locations,
                                    "owned_pellet_locations": owned_pellet_locations,
                                    "num_carried_pellets": num_carried_pellets,
                                }
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
        state_temp = {}
        for key, value in state.items():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                state_temp[key] = sorted(value.tolist())
            elif isinstance(value, np.ndarray) and value.ndim == 1:
                state_temp[key] = value.tolist()
            elif isinstance(value, list) or isinstance(value, tuple):
                state_temp[key] = sorted(value)
            else:
                state_temp[key] = int(value)
        return json.dumps(state_temp)

    def encode_action(self, action):
        """
        Encodes action as a string.
        """
        return str(action)

    def _register_state(self, state):
        """
        Set the current environment state to the given state.
        """
        self.agent_position = state["agent_position"]
        self.pellet_locations = {
            "free": state["free_pellet_locations"],
            "owned": state["owned_pellet_locations"],
        }
        self.num_carried_pellets = state["num_carried_pellets"]

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

    def _get_state(self):
        return {
            "agent_position": self.agent_position,
            "free_pellet_locations": self.pellet_locations["free"],
            "owned_pellet_locations": self.pellet_locations["owned"],
            "num_carried_pellets": self.num_carried_pellets,
        }

    def _get_observation(self):
        image = np.zeros((3, self.grid_size, self.grid_size), dtype=np.int16)
        image[0, self.agent_position[0], self.agent_position[1]] = 1
        for pellet_location in self.pellet_locations["free"]:
            image[1, pellet_location[0], pellet_location[1]] = 1
        for pellet_location in self.pellet_locations["owned"]:
            image[2, pellet_location[0], pellet_location[1]] = 1
        categorical = np.full((1, self.grid_size, self.grid_size), self.num_carried_pellets, dtype=np.int16)
        return np.concatenate([image, categorical], axis=0)

    def obs_to_state(self, obs):
        image, categorical = utils.separate_image_and_categorical_state(np.array([obs]), self.categorical_spaces)
        image, categorical = image[0], categorical[0]
        return {
            "agent_position": np.array(np.where(image[0, :, :] == 1)).T[0],
            "free_pellet_locations": np.array(np.where(image[1, :, :] == 1)).T,
            "owned_pellet_locations": np.array(np.where(image[2, :, :] == 1)).T,
            "num_carried_pellets": np.where(categorical == 1)[0][0],
        }

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

    def render(self):
        """Simple ASCII rendering of the environment."""
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
                print("Policy action: {}".format(self._action_to_string(policy.predict(self._get_state()))))
            if optimal_qs is not None:
                for action in range(self.action_space.n):
                    print(
                        "Q({}) = {}".format(
                            self._action_to_string(action),
                            optimal_qs[0][self.get_state_index(self._get_state())][action],
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
                        action = policy.predict(self._get_state())

            _, reward, done, _ = self.step(action)
            total_reward += reward
            print("Reward: {}".format(reward))
            print()
            self.render()

            if done:
                print("Episode done.")
                break

        print("Total reward: {}".format(total_reward))


if __name__ == "__main__":
    STEPS = 20
    env = StealingGridworld(grid_size=3, max_steps=STEPS)
    print(f"State space size: {len(env.states)}")

    dummy_reward = lambda states, *args: np.array([s["agent_position"][0] for s in states])
    traj_gen = DeterministicMDPTrajGenerator(dummy_reward, env, None)
    print("Before training (random policy):")
    env.render_rollout(traj_gen.sample(1)[0])
    print("=====")

    traj_gen.train(STEPS)
    print("After training (Run down policy):")
    env.render_rollout(traj_gen.sample(1)[0])
    print("=====")

    traj_gen.reward_fn = env.reward_fn_vectorized
    traj_gen.train(STEPS)
    print("After training (Optimal policy):")
    env.render_rollout(traj_gen.sample(1)[0])
