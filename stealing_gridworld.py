import gymnasium as gym
import numpy as np
import tqdm
from gymnasium import spaces
from scipy import sparse


class StealingGridworld(gym.Env):
    """
    A gridworld in which the agent is rewarded for bringing home pellets, and punished for stealing pellets that belong
    to someone else.

    The agent starts at the home location, and can move up, down, left, or right. It also has an "interact" action, which
    handles picking up pellets and depositing them at home.

    "Free" pellets and "owned" pellets are distinguished. The agent can pick up either type of pellet, but it will be
    punished for picking up an "owned" pellet (i.e. stealing). The agent can only deposit pellets at home, and it will be
    rewarded for each pellet deposited.
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

        self.observation_space = spaces.Dict(
            {
                "agent_position": spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32),
                "free_pellet_locations": spaces.Box(
                    low=0, high=grid_size - 1, shape=(num_free_pellets, 2), dtype=np.int32
                ),
                "owned_pellet_locations": spaces.Box(
                    low=0, high=grid_size - 1, shape=(num_owned_pellets, 2), dtype=np.int32
                ),
                "num_carried_pellets": spaces.Discrete(self.num_pellets + 1),
            }
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

        return self._get_observation(), {}

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

        return self._get_observation(), reward, done, {}

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
        grid[self.agent_position[0], self.agent_position[1]] = AGENT

        print("+" + "---+" * self.grid_size)
        for i in range(self.grid_size):
            print("|", end="")
            for j in range(self.grid_size):
                print(" {} |".format(grid[i, j]), end="")
            print("\n+" + "---+" * self.grid_size)

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
        return {
            "agent_position": self.agent_position,
            "free_pellet_locations": self.pellet_locations["free"],
            "owned_pellet_locations": self.pellet_locations["owned"],
            "num_carried_pellets": self.num_carried_pellets,
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

    def repl(self):
        """Simple REPL for the environment."""
        print("Welcome to the StealingGridworld REPL.")
        print("Use the following commands:")
        print("  u: Move up")
        print("  d: Move down")
        print("  l: Move left")
        print("  r: Move right")
        print("  i: Interact with environment")
        print("  q: Quit")
        print("")

        total_reward = 0

        while True:
            self.render()
            action = input("Action: ")

            if action.lower() == "q":
                break

            try:
                action = StealingGridworld._string_to_action(action)
            except KeyError:
                print("Invalid action.")
                continue

            obs, reward, done, info = self.step(action)
            total_reward += reward
            print("Reward: {}".format(reward))
            print()

            if done:
                print("Episode done.")
                break

        print("Total reward: {}".format(total_reward))


def run_value_iteration(
    sparse_transitions: sparse.csr_matrix,
    rewards_vector: np.ndarray,
    horizon: int,
    gamma: float = 1,
):
    num_state_actions, num_states = sparse_transitions.shape
    num_actions = num_state_actions // num_states

    done_q = np.zeros((num_states, num_actions), dtype=rewards_vector.dtype)
    done_v = np.zeros(num_states, dtype=rewards_vector.dtype)

    optimal_qs = [done_q]
    optimal_values = [done_v]

    for t in tqdm.tqdm(list(reversed(list(range(horizon)))), desc="Value iteration"):
        optimal_qs.insert(
            0,
            (rewards_vector + gamma * sparse_transitions @ optimal_values[0]).reshape((num_states, num_actions)),
        )
        optimal_values.insert(0, optimal_qs[0].max(axis=1))


if __name__ == "__main__":
    env = StealingGridworld()
    env.repl()
