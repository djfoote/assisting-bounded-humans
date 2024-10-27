import gymnasium as gym
import numpy as np
import torch as th
from imitation.data.types import TrajectoryWithRew
from scipy import sparse

from partial_observability import (
    BeliefDistribution,
    BeliefFunction,
    ObservationFunction,
    TrajectoryWithObs,
    TrajectoryWithRewAndObs,
)


class GraphMDP(gym.Env):
    """
    An MDP represented by an explicit state graph. Meant to represent the environments in the PORLHF paper.

    Link: https://arxiv.org/pdf/2402.17747

    Actions give a lottery over successor states. Rewards are defined on states.

    Actions and states are represented using strings. The graph is represented as a nested dictionary mapping:
    State -> Action -> Successor state -> Probability.

    This is implemented in a way that is only reasonable for very small environments.
    """

    def __init__(self, start_states, graph, rewards, horizon):
        self.start_states = start_states
        self.states = list(graph.keys())
        self.state_index = {state: i for i, state in enumerate(self.states)}
        self.actions = list({action for state in graph for action in graph[state]})
        self.action_index = {action: i for i, action in enumerate(self.actions)}
        self.graph = graph
        self.rewards = rewards
        self.horizon = horizon
        self.max_steps = horizon

    def get_sparse_transition_matrix_and_reward_vector(self, alt_reward_fn=None):
        rewards_per_state = [self.rewards[state] for state in self.states]
        if alt_reward_fn is not None:
            with th.no_grad():
                rewards_per_state = alt_reward_fn(th.tensor(range(len(self.states)))).numpy()

        num_states = len(self.states)
        num_actions = len(self.actions)
        num_state_actions = num_states * num_actions

        row_indices = []
        col_indices = []
        data = []

        rewards_list = []

        for state in self.states:
            state_idx = self.state_index[state]
            for action in self.actions:
                action_idx = self.action_index[action]
                if action not in self.graph[state]:
                    rewards_list.append(-float("inf"))
                else:
                    rewards_list.append(rewards_per_state[state_idx])
                successors = self.graph[state].get(action, {self.states[-1]: 1.0})
                for successor, probability in successors.items():
                    successor_idx = self.state_index[successor]
                    row_indices.append(state_idx * num_actions + action_idx)
                    col_indices.append(successor_idx)
                    data.append(probability)

        sparse_transition_matrix = sparse.csr_matrix(
            (data, (row_indices, col_indices)), shape=(num_state_actions, num_states)
        )

        return sparse_transition_matrix, np.array(rewards_list)

    def _enumerate_trajectories_from(self, state, horizon=None):
        if horizon < 0:
            return [[]]

        if horizon is not None:
            horizon -= 1
        trajectories = []
        all_successors = []
        for action, successors_probs in self.graph[state].items():
            all_successors += [(action, next_state) for next_state in successors_probs.keys()]
        for action, next_state in all_successors:
            trajectories += [
                [(state, action)] + sequence for sequence in self._enumerate_trajectories_from(next_state, horizon)
            ]
        return trajectories

    def enumerate_trajectories(self):
        sequences = []
        for state in self.start_states:
            sequences += self._enumerate_trajectories_from(state, self.horizon)
        trajectories = []
        for sequence in sequences:
            state_indices = np.array([self.get_state_index(state) for state, _ in sequence])
            action_indices = np.array([self.get_action_index(action) for _, action in sequence[:-1]])
            rewards = np.array([self.rewards[state] for state, _ in sequence[:-1]])
            trajectories.append(
                TrajectoryWithRew(
                    obs=state_indices,
                    acts=action_indices,
                    rews=rewards,
                    terminal=True,
                    infos=None,
                )
            )
        return trajectories

    def transition(self, state, action):
        """
        Samples a (next_state, reward) pair from the environment. In this environment, reward is deterministic based on
        the state, so the only case where it's different is when an illegal action is taken.
        """
        if action not in self.graph[state]:
            return state, -float("inf")
        successors = self.graph[state][action]
        next_state = np.random.choice(list(successors.keys()), p=list(successors.values()))
        reward = self.rewards[state]
        return next_state, reward

    def get_state_index(self, state):
        return self.state_index[state]

    def get_action_index(self, action):
        return self.action_index[action]

    # TODO: Very sloppy code duplication with DeterministicMDP.rollout_with_policy. These should inherit from the same
    #       base class.
    def rollout_with_policy(self, policy, fixed_horizon=None, epsilon=None):
        if fixed_horizon is None:
            fixed_horizon = self.horizon

        state = np.random.choice(self.start_states)
        state_indices = [self.state_index[state]]
        action_indices = []
        rewards = []

        while len(action_indices) < fixed_horizon:
            if epsilon is not None and np.random.random() < epsilon:
                available_actions = list(self.graph[state].keys())
                action = np.random.choice(available_actions)
            else:
                action = self.actions[policy.predict(state)]
            next_state, reward = self.transition(state, action)
            state_indices.append(self.state_index[next_state])
            action_indices.append(self.action_index[action])
            rewards.append(reward)
            state = next_state
        return TrajectoryWithRew(
            obs=np.array(state_indices, dtype=np.int16),
            acts=np.array(action_indices, dtype=np.int16),
            rews=np.array(rewards, dtype=float),
            terminal=True,
            infos=None,
        )


class GraphObservationFunction(ObservationFunction):
    """
    An observation function that maps states to observations according to a dictionary.
    """

    def __init__(self, graph_mdp: GraphMDP, observation_fn_dict: dict):
        self.graph_mdp = graph_mdp
        self.observation_fn_dict = observation_fn_dict

    def __call__(self, trajectory):
        state_sequence = trajectory.obs
        observations = np.array([self.observation_fn_dict[self.graph_mdp.states[s_idx]] for s_idx in state_sequence])
        return TrajectoryWithObs(
            state=state_sequence,
            obs=observations,
            acts=trajectory.acts,
            infos=trajectory.infos,
            terminal=trajectory.terminal,
        )


class MatrixBeliefFunction(BeliefFunction):
    """
    A belief function that maps observation sequences to distributions over state sequences according to a dictionary.

    Any observation sequence not in the dictionary is assumed to have no ambiguity; that is, it can only be produced by
    a single state sequence, which is assigned probability 1. If it can be produced by multiple state sequences, the
    code will raise an error.

    Note that this is a non-trivial assumption that is not justified in general. It assumes that a human has a
    sufficiently correct model of the environment that they recognize unambiguous observations and correctly infer the
    state sequence that produced them.

    To avoid making this assumption, use the default value of None for observation_fn_dict in the constructor.
    """

    def __init__(self, graph_mdp: GraphMDP, belief_fn_dict, observation_fn=None):
        """
        observation_fn_dict is used to handle unambiguous observations. If it is not provided and an observation
        sequence appears that is not explicitly accounted for in belief_fn_dict, an error will be raised.
        """
        self.graph_mdp = graph_mdp
        self.belief_fn_dict = belief_fn_dict

        self.unambiguous_obs_seq = {}
        if observation_fn is not None:
            # TODO: This can probably be cleaned up by actually using the ObservationFunction interface.
            observation_fn_dict = observation_fn.observation_fn_dict
            all_state_sequences = [traj.obs for traj in graph_mdp.enumerate_trajectories()]
            all_obs_seq = {}
            for state_seq in all_state_sequences:
                obs_seq = " ".join([observation_fn_dict[self.graph_mdp.states[state_idx]] for state_idx in state_seq])
                if obs_seq in all_obs_seq:
                    all_obs_seq[obs_seq].append(state_seq)
                else:
                    all_obs_seq[obs_seq] = [state_seq]
            for obs_seq, state_seqs in all_obs_seq.items():
                if len(state_seqs) == 1:
                    self.unambiguous_obs_seq[obs_seq] = state_seqs[0]

    def __call__(self, obs_seq: TrajectoryWithRewAndObs) -> BeliefDistribution:
        probs = []
        obs = obs_seq.obs
        obs_seq_key = " ".join(obs)
        if obs_seq_key in self.belief_fn_dict:
            for state_seq_key, prob in self.belief_fn_dict[obs_seq_key].items():
                state_seq = state_seq_key.split()
                # TODO: Not sure yet if we need to use state (string) or state index (int) here.
                #       Would need to do the same in the unambiguous case.
                state_seq_idx = [self.graph_mdp.state_index[state] for state in state_seq]

                # Construct a TrajectoryWithRewAndObs object with the state sequence.
                # Must compute rews based on the inferred state sequence, rather than the ground truth.
                rews = np.array([self.graph_mdp.rewards[self.graph_mdp.states[s_idx]] for s_idx in state_seq_idx[:-1]])
                traj = TrajectoryWithRewAndObs(
                    state=np.array(state_seq_idx),
                    obs=obs_seq.obs,
                    acts=obs_seq.acts,
                    rews=rews,
                    infos=obs_seq.infos,
                    terminal=obs_seq.terminal,
                )
                probs.append((traj, prob))
        elif obs_seq_key in self.unambiguous_obs_seq:
            state_seq_idx = self.unambiguous_obs_seq[obs_seq_key]
            rews = np.array([self.graph_mdp.rewards[self.graph_mdp.states[s_idx]] for s_idx in state_seq_idx[:-1]])
            traj = TrajectoryWithRewAndObs(
                state=np.array(state_seq_idx),
                obs=obs_seq.obs,
                acts=obs_seq.acts,
                rews=rews,
                infos=obs_seq.infos,
                terminal=obs_seq.terminal,
            )
            probs.append((traj, 1.0))
        else:
            raise ValueError(f"Observation sequence {obs} not in belief function dictionary.")
        return BeliefDistribution(probs)
