import collections
import functools
import itertools

import numpy as np


class MDPWithObservations:
    def __init__(
        self,
        transition_kernel,
        observation_kernel,
        human_belief_function,
        reward_function,
        state_sequences,
        observation_sequences,
    ):
        self.transition_kernel = transition_kernel
        self.observation_kernel = observation_kernel
        self.human_belief_function = human_belief_function
        self.reward_function = reward_function
        self.state_sequences = state_sequences
        self.observation_sequences = observation_sequences

        self.horizon = len(state_sequences[0])  # Assume all state sequences have the same length

        self._obs_idx = {obs_seq: i for i, obs_seq in enumerate(observation_sequences)}
        self._s_to_o_idx = {seq: self._obs_idx[obs_kernel_seq(seq, observation_kernel)] for seq in state_sequences}

        self.return_function = np.array([np.sum(reward_function[[seq]]) for seq in state_sequences])
        self._obs_return_on_o = human_belief_function @ self.return_function
        self.obs_return_function = np.array([self._obs_return_on_o[self._s_to_o_idx[seq]] for seq in state_sequences])

    def on_policy_distribution(self, policy):
        partial_sequence_probs = {(0,): 1}
        for _ in range(self.horizon - 1):
            new_partial_sequence_probs = collections.defaultdict(float)
            for sequence, p_prefix in partial_sequence_probs.items():
                prev_state = sequence[-1]
                for next_state in range(self.transition_kernel.shape[2]):
                    p_sequence = p_prefix * self.transition_kernel[prev_state, policy[prev_state], next_state]
                    if p_sequence > 0:
                        new_partial_sequence_probs[sequence + (next_state,)] += p_sequence
            partial_sequence_probs = new_partial_sequence_probs

        on_policy_distribution = np.zeros(len(self.state_sequences), dtype=float)
        for i, sequence in enumerate(self.state_sequences):
            on_policy_distribution[i] = partial_sequence_probs.get(sequence, 0)

        return on_policy_distribution

    def J(self, policy):
        return self.on_policy_distribution(policy) @ self.return_function

    def J_obs(self, policy):
        return self.on_policy_distribution(policy) @ self.obs_return_function

    def overestimation_error(self, policy):
        return self.on_policy_distribution(policy) @ np.fmax(self.obs_return_function - self.return_function, 0)

    def underestimation_error(self, policy):
        return self.on_policy_distribution(policy) @ np.fmax(self.return_function - self.obs_return_function, 0)

    def deceptive_inflation(self, policy=None, ref_policy=None):
        if policy is None:
            policy = self.rlhf_policy
        if ref_policy is None:
            ref_policy = self.optimal_policy
        overestimation_increased = self.overestimation_error(policy) > self.overestimation_error(ref_policy)
        J_obs_increased = self.J_obs(policy) > self.J_obs(ref_policy)
        return overestimation_increased and J_obs_increased

    def overjustification(self, policy=None, ref_policy=None):
        if policy is None:
            policy = self.rlhf_policy
        if ref_policy is None:
            ref_policy = self.optimal_policy
        underestimation_decreased = self.underestimation_error(policy) < self.underestimation_error(ref_policy)
        J_decreased = self.J(policy) < self.J(ref_policy)
        return underestimation_decreased and J_decreased

    @functools.cached_property
    def optimal_policy(self):
        return self.optimize_policy()

    @functools.cached_property
    def rlhf_policy(self):
        return self.optimize_policy(self.obs_return_function)

    def optimize_policy(self, return_function=None):
        if return_function is None:
            return_function = self.return_function

        # Super lazy: enumerate all deterministic policies and pick the best one
        num_states, num_actions = self.transition_kernel.shape[:2]
        all_policies = itertools.product(range(num_actions), repeat=num_states)
        best_policy = max(all_policies, key=lambda policy: self.on_policy_distribution(policy) @ return_function)
        return best_policy


def obs_kernel_seq(state_sequence, observation_kernel):
    return tuple(np.where(observation_kernel[state] == 1)[0][0] for state in state_sequence)
