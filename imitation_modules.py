from typing import Any, Dict, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from imitation.algorithms import preference_comparisons
from imitation.data.types import TrajectoryWithRew
from imitation.rewards.reward_nets import RewardNet
from imitation.util import networks, util

import value_iteration


class DeterministicMDPTrajGenerator(preference_comparisons.TrajectoryGenerator):
    """
    A trajectory generator for a deterministic MDP that can be solved exactly using value iteration.
    """

    def __init__(self, reward_fn, env, rng, vi_gamma=0.99, max_vi_steps=None, epsilon=None, custom_logger=None):
        super().__init__(custom_logger=custom_logger)

        self.reward_fn = reward_fn
        self.env = env
        self.rng = rng
        self.vi_gamma = vi_gamma
        self.epsilon = epsilon

        if max_vi_steps is None:
            if hasattr(self.env, "max_steps"):
                max_vi_steps = self.env.max_steps
            else:
                raise ValueError("max_vi_steps must be specified if env does not have a max_steps attribute")
        self.max_vi_steps = max_vi_steps

        # TODO: Can I just pass `rng` to np.random.seed like this?
        self.policy = value_iteration.RandomPolicy(self.env, self.rng)

    def sample(self, steps):
        """
        Generate trajectories with total number of steps equal to `steps`.
        """
        trajectories = []
        total_steps = 0
        while total_steps < steps:
            trajectory = self.env.rollout_with_policy(
                self.policy,
                fixed_horizon=self.max_vi_steps,
                epsilon=self.epsilon,
            )
            trajectories.append(trajectory)
            total_steps += len(trajectory)
        return trajectories

    def train(self, steps):
        """
        Find the optimal policy using value iteration under the given reward function.
        Overrides the train method as required for imitation.preference_comparisons.
        """
        vi_steps = min(steps, self.max_vi_steps)
        self.policy = value_iteration.get_optimal_policy(
            self.env, gamma=self.vi_gamma, horizon=vi_steps, alt_reward_fn=self.reward_fn
        )


class NonImageCnnRewardNet(RewardNet):
    """
    A CNN reward network that does not make assumptions about the input being an image. In particular, it does not
    apply standard image preprocessing (e.g. normalization) to the input.

    Because the code that requires the input to be an image occurs in the __init__ method of CnnRewardNet (which is a
    more natural choice for superclass), we actually need to only subclass RewardNet and reimplement some functionality
    from CnnRewardNet.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, normalize_images=False)

        input_size = observation_space.shape[0]
        output_size = action_space.n

        full_build_cnn_kwargs: Dict[str, Any] = {
            "hid_channels": (32, 32),
            **kwargs,
            # we do not want the values below to be overridden
            "in_channels": input_size,
            "out_size": output_size,
            "squeeze_output": output_size == 1,
        }

        self.cnn = networks.build_cnn(**full_build_cnn_kwargs)

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Override standard input preprocess to bypass image preprocessing. Only lifts inputs to tensors.
        """
        state_th = util.safe_to_tensor(state).to(self.device).float()
        action_th = util.safe_to_tensor(action).to(self.device)
        next_state_th = util.safe_to_tensor(next_state).to(self.device)
        done_th = util.safe_to_tensor(done).to(self.device)

        return state_th, action_th, next_state_th, done_th

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Computes rewardNet value on input state and action. Ignores next_state, and done flag.

        Args:
            state: current state.
            action: current action.
            next_state: next state.
            done: flag for whether the episode is over.

        Returns:
            th.Tensor: reward of the transition.
        """
        outputs = self.cnn(state)
        # for discrete action spaces, action should be passed to forward as a one-hot vector.
        # If action is not 1-hot vector, then we need to convert it to one-hot vector
        # TODO: Chase down where this should actually be happening upstream of here
        if action.ndim == 1:
            rewards = outputs[th.arange(action.shape[0]), action.int()]
        else:
            rewards = th.sum(outputs * action, dim=1)

        return rewards


class SyntheticValueGatherer(preference_comparisons.SyntheticGatherer):
    """
    Computes synthetic preferences by a weighted combination of ground-truth environment rewards (present in the
    trajectory fragment) and ground-truth optimal value at the end of the trajectory fragment (computed using value
    iteration).
    """

    def __init__(
        self,
        env,
        temperature=1.0,
        rlhf_gamma=1.0,
        sample=True,
        rng=None,
        threshold=50,
        vi_horizon=None,
        vi_gamma=0.99,
        value_coeff=0.1,  # weight of value in synthetic reward
        custom_logger=None,
    ):
        super().__init__(temperature, rlhf_gamma, sample, rng, threshold, custom_logger)

        self.env = env
        self.vi_horizon = vi_horizon
        self.vi_gamma = vi_gamma

        self.value_coeff = value_coeff

        _, self.values = value_iteration.get_optimal_policy_and_values(
            self.env, gamma=self.vi_gamma, horizon=self.vi_horizon
        )

    def _get_value(self, state):
        return self.values[self.env.get_state_index(state)]

    def _augment_fragment_pair_with_value(self, fragment_pair):
        new_fragments = []
        for fragment in fragment_pair:
            final_state = fragment.obs[-1]
            value = self._get_value(final_state)
            new_rews = np.copy(fragment.rews)
            new_rews[-1] += self.value_coeff * value
            new_fragments.append(
                TrajectoryWithRew(fragment.obs, fragment.acts, fragment.infos, fragment.terminal, new_rews)
            )
        return tuple(new_fragments)

    def __call__(self, fragment_pairs):
        fragment_pairs = [self._augment_fragment_pair_with_value(fp) for fp in fragment_pairs]
        return super().__call__(fragment_pairs)
