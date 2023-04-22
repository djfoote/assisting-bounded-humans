from typing import Iterable, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import RewardNet
from torch import nn

import value_iteration
from utils import concatenate_categorical_data_to_images


class DeterministicMDPTrajGenerator(preference_comparisons.TrajectoryGenerator):
    """
    A trajectory generator for a deterministic MDP that can be solved exactly using value iteration.
    """

    def __init__(self, reward_fn, env, rng, vi_gamma=0.99, max_vi_steps=None, custom_logger=None):
        super().__init__(custom_logger=custom_logger)

        self.reward_fn = reward_fn
        self.env = env
        self.rng = rng
        self.vi_gamma = vi_gamma

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
            trajectory = self.env.rollout_with_policy(self.policy, fixed_horizon=self.max_vi_steps)
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


class ImageAndCategoricalRewardNet(RewardNet):
    """Reward network for state spaces with both image components and categorical components.
    For now, will only support functionality necessary for the `StealingGridworld` experiments, and will make
    assumptions that aren't true more generally.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        categorical_spaces: Iterable[gym.spaces.Discrete],
        action_space: gym.spaces.Discrete,
        **net_kwargs,
    ):
        """
        Args:
            obervation_space: The observation space. Observations are images where the last `n_categorical_elements`
                channels are categorical, with the value of the variable smeared across each pixel.
            categorical_spaces: The categorical spaces. Note that the spaces are already contained in the observation
                space, but we need them here to compute the size of the input to the fully-connected layers. The
                categorical spaces are assumed to be in the same order as the categorical channels in the observation
                space.
            action_space: The action space.
            net_kwargs: Keyword arguments to pass to `self._build_network`.
        """
        super().__init__(observation_space, action_space)
        self.categorical_spaces = categorical_spaces
        self.image_channels = observation_space.shape[1] - len(categorical_spaces)
        # TODO: validate that the spaces are what we expect.

        self._build_network(**net_kwargs)

    def _build_network(
        self,
        cnn_hidden_channels: Iterable[int],
        fc_hidden_sizes: Iterable[int],
    ):
        categorical_data_size = sum(space.n for space in self.categorical_spaces)

        # Build the image CNN.
        cnn_layers = []
        prev_channels = self.image_channels
        for out_channels in cnn_hidden_channels:
            cnn_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=prev_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                    ),
                    nn.ReLU(),
                )
            )
            prev_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers, nn.Flatten())

        # Build the fully-connected layers.
        fc_layers = []
        prev_size = out_channels + categorical_data_size
        for out_size in fc_hidden_sizes:
            fc_layers.append(nn.Linear(prev_size, out_size))
            fc_layers.append(nn.ReLU())
            prev_size = out_size
        self.fc = nn.Sequential(*fc_layers)

        # Build the output layer.
        self.final_layer = nn.Linear(prev_size, self.action_space.n)

    def preprocess(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, done: np.ndarray
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Preprocess the state, action, next_state, and done arrays.
        `state` is assumed to have shape (batch_size, n+1) where n is the number of categorical components of the
        state. For each batch element, the first element is the image (of shape CxHxW), and the remaining elements are
        categorical data represented as integers.
        Args:
            state (np.ndarray): A batch of states.
            action (np.ndarray): A batch of actions.
            next_state (np.ndarray): A batch of next states.
            done (np.ndarray): A batch of done indicators.
        Returns:
            A tuple of (state, action, next_state, done), where each element is a tensor. For states, the categorical
            components are concatenated to the image as extra channels. These extra channels are just the number of the
            category at each pixel, so they are not one-hot encoded.
        """
        state_np = concatenate_categorical_data_to_images(state[:, 0], state[:, 1:])
        next_state_np = concatenate_categorical_data_to_images(next_state[:, 0], next_state[:, 1:])

        state_th = th.as_tensor(state_np, dtype=th.int16)
        action_th = th.as_tensor(action, dtype=th.int16)
        next_state_th = th.as_tensor(next_state_np, dtype=th.int16)
        done_th = th.as_tensor(done, dtype=th.int16)

        return state_th, action_th, next_state_th, done_th

    def _separate_image_and_categorical_state(self, state: th.Tensor) -> Tuple[th.Tensor, Iterable[th.Tensor]]:
        """
        Separate the image and categorical components of the state.
        Args:
            state (th.Tensor): A preprocessed batch of states.
        Returns:
            A tuple of (image, categorical), where `image` is a batch of images and `categorical` is a list of batches
            of categorical data, one-hot encoded.
        """
        image = state[:, : self.image_channels]
        categorical = []
        for i, space in enumerate(self.categorical_spaces):
            category_values = state[:, self.image_channels + i, 0, 0]  # Smeared across all pixels; just take one.
            categorical.append(th.nn.functional.one_hot(category_values, space.n))
        return image, categorical

    def forward(self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor) -> th.Tensor:
        image, categoricals = self._separate_image_and_categorical_state(state)
        image_features = self.cnn(image)
        categorical_features = th.cat(categoricals, dim=1)
        features = th.cat([image_features, categorical_features], dim=1)
        net_output = self.final_layer(self.fc(features))
        rewards = th.sum(net_output * action, dim=1)
        return rewards
