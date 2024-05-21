import abc
import itertools
import pickle
import re
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from imitation.algorithms import base, preference_comparisons
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew
from imitation.rewards.reward_nets import RewardNet
from imitation.util import logger as imit_logger
from imitation.util import networks, util
from torch import nn
from torch.utils import data as data_th
from tqdm.auto import tqdm

import value_iteration
from sbx import PPO
from gymnasium.wrappers import FlattenObservation
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_checker import check_env



class DeterministicMDPTrajGenerator(preference_comparisons.TrajectoryGenerator):
    """
    A trajectory generator for a deterministic MDP that can be solved exactly using value iteration.
    """

    def __init__(self, reward_fn, env, rng, vi_gamma=0.99, max_vi_steps=None, epsilon=None, custom_logger=None, wandb_run=None):
        super().__init__(custom_logger=custom_logger)

        self.reward_fn = reward_fn
        self.env = env
        self.rng = rng
        self.vi_gamma = vi_gamma
        self.epsilon = epsilon
        self.run = wandb_run

        if max_vi_steps is None:
            if hasattr(self.env, "max_steps"):
                max_vi_steps = self.env.max_steps
            else:
                raise ValueError("max_vi_steps must be specified if env does not have a max_steps attribute")
        self.max_vi_steps = max_vi_steps

        # TODO: Can I just pass `rng` to np.random.seed like this?
        env = FlattenObservation(env)
        print('Checking environment...')
        check_env(env)
        self.policy = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[128, 128, 128]), verbose=2)

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
                render=False,
            )
            trajectories.append(trajectory)
            total_steps += len(trajectory)
        return trajectories

    def train(self, steps):
        """
        Find the optimal policy using value iteration under the given reward function.
        Overrides the train method as required for imitation.preference_comparisons.
        """
        
        # replace value iteration with PPO training

        self.policy.learn(total_timesteps=steps*100, progress_bar=True, callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{self.run.id}",
        verbose=2,
    ),)

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
        # if state input is a flattened array, reshape into an 'image' 
        batch = state.shape[0]
        grid_size = np.sqrt(state.shape[1] // 5).astype(int)
        state = state.reshape(batch, 5, grid_size, grid_size)
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
            action = th.nn.functional.one_hot(action.long(), num_classes=outputs.shape[1])
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


class ScalarFeedbackDataset(data_th.Dataset):
    """A PyTorch Dataset for scalar reward feedback.

    Each item is a tuple consisting of a trajectory fragment and a scalar reward (given by a FeedbackGatherer; not
    necessarily the ground truth environment rewards).

    This dataset is meant to be generated piece by piece during the training process, which is why data can be added
    via the .push() method.
    """

    def __init__(self, max_size=None):
        self.fragments = []
        self.max_size = max_size
        self.reward_labels = np.array([])

    def push(self, fragments, reward_labels):
        self.fragments.extend(fragments)
        self.reward_labels = np.concatenate((self.reward_labels, reward_labels))

        # Evict old samples if the dataset is at max capacity
        if self.max_size is not None:
            extra = len(self.reward_labels) - self.max_size
            if extra > 0:
                self.fragments = self.fragments[extra:]
                self.reward_labels = self.reward_labels[extra:]

    def __getitem__(self, index):
        return self.fragments[index], self.reward_labels[index]

    def __len__(self):
        assert len(self.fragments) == len(self.reward_labels)
        return len(self.reward_labels)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)


class RandomSingleFragmenter(preference_comparisons.RandomFragmenter):
    """Fragmenter that samples single fragments rather than fragment pairs.

    Intended to be used for non-comparison-based feedback, such as scalar reward feedback.
    """

    def __call__(self, trajectories, fragment_length, num_fragments):
        fragment_pairs = super().__call__(trajectories, fragment_length, int(np.ceil(num_fragments // 2)))
        # fragment_pairs is a list of (fragment, fragment) tuples. We want to flatten this into a list of fragments.
        return list(itertools.chain.from_iterable(fragment_pairs))


class ScalarFeedbackModel(nn.Module):
    """Class to convert a fragment's reward into a scalar feedback label."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, fragments):
        """Computes scalar feedback labels for the given fragments."""
        reward_predictions = []
        for fragment in fragments:
            transitions = rollout.flatten_trajectories([fragment])
            preprocessed = self.model.preprocess(
                transitions.obs,
                transitions.acts,
                transitions.next_obs,
                transitions.dones,
            )
            reward_prediction_per_step = self.model(*preprocessed)
            assert reward_prediction_per_step.shape == (len(transitions.obs),)
            reward_prediction = th.sum(reward_prediction_per_step, dim=0)
            reward_predictions.append(reward_prediction)
        return th.stack(reward_predictions)


class ScalarFeedbackGatherer(abc.ABC):
    """Base class for gathering scalar feedback for a trajectory fragment."""

    def __init__(self, rng=None, custom_logger=None):
        del rng  # unused
        self.logger = custom_logger or imit_logger.configure()

    @abc.abstractmethod
    def __call__(self, fragments):
        """Gathers the scalar feedback for the given fragments.

        See preference_comparisons.PreferenceGatherer for more details.
        """


class SyntheticScalarFeedbackGatherer(ScalarFeedbackGatherer):
    """Computes synthetic scalar feedback using ground-truth environment rewards."""

    # TODO: This is a placeholder for a more sophisticated synthetic feedback gatherer.

    def __call__(self, fragments):
        return [np.sum(fragment.rews) for fragment in fragments]


class NoisyObservationGathererWrapper(ScalarFeedbackGatherer):
    """Wraps a scalar feedback gatherer to handle the feedback giver seeing a noisy observation of state, rather than
    the true environment state.

    Current implementation only supports deterministic observation noise (such as occlusion). Later implementations
    will pass a random seed to the observation function to support stochastic observation noise. For now, a stochastic
    observation function will not fail, but will not be seed-able, so results will not be reproducible.
    """

    def __init__(self, gatherer: ScalarFeedbackGatherer, observe_fn):
        self.wrapped_gatherer = gatherer
        self.observe_fn = observe_fn

    def __getattr__(self, name):
        return getattr(self.wrapped_gatherer, name)

    def __call__(self, fragments):
        noisy_fragments = [self.observe_fn(fragment) for fragment in fragments]
        return self.wrapped_gatherer(noisy_fragments)


class ObservationFunction(abc.ABC):
    """Abstract class for functions that take an observation and return a new observation."""

    @abc.abstractmethod
    def __call__(self, fragment):
        """Returns a new fragment with observations, actions, and rewards filtered through an observation function.

        Args:
            fragment: a TrajectoryWithRew object.

        Returns:
            A new TrajectoryWithRew object with the same infos and terminal flag, but with the observations, actions,
            and rewards filtered through the observation function.
        """


class ScalarFeedbackRewardTrainer(abc.ABC):
    """Base class for training a reward model using scalar feedback."""

    def __init__(self, feedback_model, custom_logger=None):
        self._feedback_model = feedback_model
        self._logger = custom_logger or imit_logger.configure()

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, custom_logger):
        self._logger = custom_logger

    def train(self, dataset, epoch_multiplier=1.0):
        """Trains the reward model using the given dataset (a batch of fragments and feedback).

        Args:
            dataset: a Dataset object containing the feedback data.
            epoch_multiplier: a multiplier for the number of epochs to train for.
        """
        with networks.training(self._feedback_model.model):
            self._train(dataset, epoch_multiplier)

    @abc.abstractmethod
    def _train(self, dataset, epoch_multiplier):
        """Train the reward model; see ``train`` for details."""


class MSERewardLoss(preference_comparisons.RewardLoss):
    """Compute the MSE between the given rewards and the feedback labels."""

    def forward(self, fragments, feedback_labels, feedback_model):
        """Computes the MSE between the given rewards and the feedback labels."""
        reward_predictions = feedback_model(fragments)
        feedback_th = th.as_tensor(feedback_labels, dtype=th.float32, device=reward_predictions.device)
        return th.mean((reward_predictions - feedback_th) ** 2)


class BasicScalarFeedbackRewardTrainer(ScalarFeedbackRewardTrainer):
    """Train a basic reward model from scalar feedback."""

    def __init__(
        self,
        feedback_model,
        loss,
        rng,
        batch_size=32,
        minibatch_size=None,
        epochs=1,
        lr=1e-3,
        custom_logger=None,
    ):
        super().__init__(feedback_model, custom_logger=custom_logger)
        self.loss = loss
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size or batch_size
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("batch_size must be divisible by minibatch_size")
        self.epochs = epochs
        self.optim = th.optim.AdamW(self._feedback_model.parameters(), lr=lr)
        self.rng = rng
        self.lr = lr

    def _make_data_loader(self, dataset):
        return data_th.DataLoader(
            dataset,
            batch_size=self.minibatch_size,
            shuffle=True,
            collate_fn=lambda batch: tuple(zip(*batch)),
        )

    def _train(self, dataset, epoch_multiplier=1.0):
        dataloader = self._make_data_loader(dataset)
        epochs = np.round(self.epochs * epoch_multiplier).astype(int)
        assert epochs > 0, "Must train for at least one epoch."
        with self.logger.accumulate_means("reward"):
            for epoch_num in tqdm(range(epochs), desc="Training reward model"):
                with self.logger.add_key_prefix(f"epoch-{epoch_num}"):
                    train_loss = 0.0
                    accumulated_size = 0
                    self.optim.zero_grad()
                    for fragments, feedback in dataloader:
                        with self.logger.add_key_prefix("train"):
                            loss = self._training_inner_loop(fragments, np.array(feedback))
                            loss *= len(fragments) / self.batch_size  # rescale loss to account for minibatching
                        train_loss += loss.item()
                        loss.backward()
                        accumulated_size += len(fragments)
                        if accumulated_size >= self.batch_size:
                            self.optim.step()
                            self.optim.zero_grad()
                            accumulated_size = 0
                    if accumulated_size > 0:
                        self.optim.step()  # if there remains an incomplete batch

        # after training all the epochs,
        # record also the final value in a separate key for easy access.
        keys = list(self.logger.name_to_value.keys())
        outer_prefix = self.logger.get_accumulate_prefixes()
        for key in keys:
            base_path = f"{outer_prefix}reward/"  # existing prefix + accum_means ctx
            epoch_path = f"mean/{base_path}epoch-{epoch_num}/"  # mean for last epoch
            final_path = f"{base_path}final/"  # path to record last epoch
            pattern = rf"{epoch_path}(.+)"
            if regex_match := re.match(pattern, key):
                (key_name,) = regex_match.groups()
                val = self.logger.name_to_value[key]
                new_key = f"{final_path}{key_name}"
                self.logger.record(new_key, val)

    def _training_inner_loop(self, fragments, feedback):
        """Inner loop of training, for a single minibatch."""
        # The imitation implementation returns a NamedTuple where `loss` has to be unpacked. This is to pass accuracy
        # through in addition to loss for logging. I've decided to skip all that for now.
        loss = self.loss.forward(fragments, feedback, self._feedback_model)
        self.logger.record("loss", loss)
        return loss


class ScalarRewardLearner(base.BaseImitationAlgorithm):
    """Main interface for reward learning using scalar reward feedback.

    Largely mimicking PreferenceComparisons class from imitation.algorithms.preference_comparisons. If this code ever
    sees the light of day, this will first have been refactored to avoid code duplication.
    """

    def __init__(
        self,
        trajectory_generator,
        reward_model,
        num_iterations,
        fragmenter,
        feedback_gatherer,
        reward_trainer,
        feedback_queue_size=None,
        fragment_length=100,
        transition_oversampling=1,
        initial_feedback_frac=0.1,
        initial_epoch_multiplier=200.0,
        custom_logger=None,
        query_schedule="hyperbolic",
        policy_evaluator=None,
        callback=None,
    ):
        super().__init__(custom_logger=custom_logger, allow_variable_horizon=False)

        # For keeping track of the global iteration, in case train() is called multiple times
        self._iteration = 0

        self.num_iterations = num_iterations

        self.model = reward_model

        self.trajectory_generator = trajectory_generator
        self.trajectory_generator.logger = self.logger

        self.fragmenter = fragmenter
        self.fragmenter.logger = self.logger

        self.feedback_gatherer = feedback_gatherer
        self.feedback_gatherer.logger = self.logger

        self.reward_trainer = reward_trainer
        self.reward_trainer.logger = self.logger

        self.feedback_queue_size = feedback_queue_size
        self.fragment_length = fragment_length
        self.transition_oversampling = transition_oversampling
        self.initial_feedback_frac = initial_feedback_frac
        self.initial_epoch_multiplier = initial_epoch_multiplier

        if query_schedule not in preference_comparisons.QUERY_SCHEDULES:
            raise NotImplementedError(f"Callable query schedules not implemented.")
        self.query_schedule = preference_comparisons.QUERY_SCHEDULES[query_schedule]

        self.dataset = ScalarFeedbackDataset(max_size=feedback_queue_size)

        self.policy_evaluator = policy_evaluator
        self.callback = callback

    def train(self, total_timesteps, total_queries):
        initial_queries = int(self.initial_feedback_frac * total_queries)
        total_queries -= initial_queries

        # Compute the number of feedback queries to request at each iteration in advance.
        vec_schedule = np.vectorize(self.query_schedule)
        unnormalized_probs = vec_schedule(np.linspace(0, 1, self.num_iterations))
        probs = unnormalized_probs / np.sum(unnormalized_probs)
        shares = util.oric(probs * total_queries)
        schedule = [initial_queries] + shares.tolist()
        print(f"Query schedule: {schedule}")

        timesteps_per_iteration, extra_timesteps = divmod(total_timesteps, self.num_iterations)
        reward_loss = None

        for i, num_queries in enumerate(schedule):
            iter_log_str = f"Beginning iteration {i} of {self.num_iterations}"
            if self._iteration != i:
                iter_log_str += f" (global iteration {self._iteration})"
            self.logger.log(iter_log_str)

            #######################
            # Gather new feedback #
            #######################
            num_steps = np.ceil(self.transition_oversampling * num_queries * self.fragment_length).astype(int)
            self.logger.log(f"Collecting {num_queries} feedback queries ({num_steps} transitions)")
            trajectories = self.trajectory_generator.sample(num_steps)
            #  This assumes there are no fragments missing initial timesteps
            # (but allows for fragments missing terminal timesteps).
            horizons = (len(traj) for traj in trajectories if traj.terminal)
            self._check_fixed_horizon(horizons)

            self.logger.log("Fragmenting trajectories")
            fragments = self.fragmenter(trajectories, self.fragment_length, num_queries)
            self.logger.log("Gathering feedback")
            feedback = self.feedback_gatherer(fragments)
            self.dataset.push(fragments, feedback)
            self.logger.log(f"Dataset now contains {len(self.dataset.reward_labels)} feedback queries")
            self.logger.record(f"dataset_size", len(self.dataset.reward_labels))

            ######################
            # Train reward model #
            ######################

            # On the first iteration, we train the reward model for longer, as specified by initial_epoch_multiplier.
            epoch_multiplier = self.initial_epoch_multiplier if i == 0 else 1.0

            start_time = time.time()
            self.reward_trainer.train(self.dataset, epoch_multiplier=epoch_multiplier)
            self.logger.record("reward_train_time", time.time() - start_time)

            base_key = self.logger.get_accumulate_prefixes() + "reward/final/train"
            assert f"{base_key}/loss" in self.logger.name_to_value
            reward_loss = self.logger.name_to_value[f"{base_key}/loss"]
            self.logger.record("reward_loss", reward_loss)

            ###################
            # Train the agent #
            ###################

            num_steps = timesteps_per_iteration
            # If the number of timesteps per iteration doesn't exactly divide the desired total number of timesteps,
            # we train the agent a bit longer at the end of training (where the reward model is presumably best).
            if i == self.num_iterations - 1:
                num_steps += extra_timesteps

            self.logger.log(f"Training agent for {num_steps} timesteps")
            self.trajectory_generator.train(steps=num_steps)

            ###################
            # Log information #
            ###################

            if self.policy_evaluator is not None:
                with networks.evaluating(self.model):
                    prop_bad, prop_bad_per_condition = self.policy_evaluator.evaluate(
                        policy=self.trajectory_generator.policy,
                        env=self.trajectory_generator.env,
                        num_trajs=1000,
                    )
                    self.logger.record("policy_behavior/prop_bad_rollouts", prop_bad)
                    for condition, prop in prop_bad_per_condition.items():
                        self.logger.record(f"policy_behavior/prop_bad_rollouts_{condition}", prop)

            self.logger.dump(self._iteration)

            if self.callback is not None:
                self.callback(self)

            self._iteration += 1

        return {"reward_loss": reward_loss}
