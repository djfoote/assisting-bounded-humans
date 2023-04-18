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
    return np.argmax(optimal_qs, axis=1)


def run_value_iteration_mdptoolbox(
    sparse_transitions: sparse.csr_matrix,
    rewards_vector: np.ndarray,
    horizon: int,
    gamma: float,
):
    vi = mdp.FiniteHorizon(
        sparse_transitions,
        rewards_vector,
        gamma,
        horizon,
    )
    vi.run()

    return vi.V[:, 0], vi.policy[:, 0]
