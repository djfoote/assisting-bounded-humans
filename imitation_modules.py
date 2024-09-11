import time
from typing import Any, Dict, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from imitation.algorithms import base, preference_comparisons
from imitation.rewards.reward_nets import RewardNet
from imitation.util import networks, util
from tqdm.auto import tqdm

import value_iteration
from human_feedback_model import FeedbackType


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


class RewardLearner(base.BaseImitationAlgorithm):
    """Main interface for reward learning using either scalar or preference reward feedback.

    Largely mimicking PreferenceComparisons class from imitation.algorithms.preference_comparisons. If this code ever
    sees the light of day, this will need to be refactored to avoid code duplication.
    """

    def __init__(
        self,
        trajectory_generator,
        reward_model,
        num_iterations,
        fragmenter,
        human_feedback_model,
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

        self.human_feedback_model = human_feedback_model
        self.human_feedback_model.logger = self.logger
        self.feedback_type = self.human_feedback_model.feedback_type

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

        self.dataset = self.human_feedback_model.dataset_class(max_size=feedback_queue_size)

        # if self.feedback_type == FeedbackType.SCALAR:
        #     self.dataset = ScalarFeedbackDataset(max_size=feedback_queue_size)
        # elif self.feedback_type == FeedbackType.PREFERENCE:
        #     self.dataset = preference_comparisons.PreferenceDataset(max_size=feedback_queue_size)
        # else:
        #     raise ValueError(f"Unsupported feedback type: {self.feedback_type} (from {self.trajectory_generator})")

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
            if self.feedback_type == "scalar":
                num_steps *= 2
            what_we_collect = "fragments" if self.feedback_type == FeedbackType.SCALAR else "fragment pairs"
            self.logger.log(f"Collecting {num_queries} {what_we_collect} " f"({num_steps} transitions)")
            trajectories = self.trajectory_generator.sample(num_steps)
            #  This assumes there are no fragments missing initial timesteps
            # (but allows for fragments missing terminal timesteps).
            horizons = (len(traj) for traj in trajectories if traj.terminal)
            self._check_fixed_horizon(horizons)

            self.logger.log("Fragmenting trajectories")
            fragments = self.fragmenter(trajectories, self.fragment_length, num_queries)
            self.logger.log("Gathering feedback")
            feedback = self.human_feedback_model(fragments)
            self.dataset.push(fragments, feedback)
            self.logger.log(f"Dataset now contains {len(self.dataset)} feedback queries")
            self.logger.record(f"dataset_size", len(self.dataset))

            ######################
            # Train reward model #
            ######################

            # On the first iteration, we train the reward model for longer, as specified by initial_epoch_multiplier.
            epoch_multiplier = self.initial_epoch_multiplier if i == 0 else 1.0

            start_time = time.time()
            self.reward_trainer.train(self.dataset, epoch_multiplier=epoch_multiplier)
            self.logger.record("reward_model_train_time_elapsed", time.time() - start_time)

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
