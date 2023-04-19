import numpy as np
import tqdm
from mdptoolbox import mdp
from scipy import sparse


def run_value_iteration(
    sparse_transitions: sparse.csr_matrix,
    rewards_vector: np.ndarray,
    horizon: int,
    gamma: float,
):
    """
    Runs value iteration on the given MDP.

    Args:
        sparse_transitions: A sparse matrix of shape (num_state_actions, num_states) where
            num_state_actions = num_states * num_actions. The matrix is in CSR format.
        rewards_vector: A vector of shape (num_state_actions,) containing the rewards for each state-action pair.
        horizon: The number of steps to run value iteration for.
        gamma: The discount factor.
    """
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

    return optimal_qs[0], optimal_values[0]


def get_optimal_policy_from_qs(optimal_qs):
    """
    Returns the optimal policy given the optimal Q values.
    """
    return np.argmax(optimal_qs, axis=1)
