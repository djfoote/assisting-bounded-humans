import abc
import dataclasses
from typing import Sequence, Tuple, Union

import numpy as np
from imitation.data.types import Pair, TrajectoryWithRew, TrajectoryWithRewPair

from human_feedback_model import FeedbackType, HumanFeedbackModel, SyntheticPreferenceHumanFeedbackModel


@dataclasses.dataclass(frozen=True)
class TrajectoryWithRewAndObs(TrajectoryWithRew):
    """A trajectory fragment with rewards and observations."""

    state: np.ndarray
    """State, shape (trajectory_len + 1, ) + state_shape.
    
    The state is defined in this class rather than observation because the imitation library uses "obs" to refer to
    what we call state. When using this class, `obs` refers to observations, rather than to states, as it does in the
    Trajectory base class in the imitation library.
    """

    def __post_init__(self):
        super().__post_init__()
        assert (
            self.state.shape[0] == self.obs.shape[0]
        ), f"Mismatching state and observation lengths: {self.state.shape[0]} != {self.obs.shape[0]}"

    def drop_obs(self) -> TrajectoryWithRew:
        return TrajectoryWithRew(
            obs=self.state,
            acts=self.acts,
            rews=self.rews,
            infos=self.infos,
            terminal=self.terminal,
        )


TrajectoryWithRewAndObsPair = Pair[TrajectoryWithRewAndObs]


# To match the theory, this should be forced to happen per state, not per fragment.
# However, this per-fragment implementation allows more efficient vectorized implementations.
# TODO: Implement per-state version, allowing subclass to also implement per-fragment version, with some kind of loose
# verification that they're equivalent (raise a warning if not).
class ObservationFunction(abc.ABC):
    """Abstract class for functions that take a state sequence and return an observation sequence."""

    @abc.abstractmethod
    def __call__(self, fragment: TrajectoryWithRew) -> TrajectoryWithRewAndObs:
        """Returns a new fragment with observations, actions, and rewards filtered through an observation function.

        Args:
            fragment: a TrajectoryWithRew object.

        Returns:
            A new TrajectoryWithRew object with the same infos and terminal flag, but with the observations, actions,
            and rewards filtered through the observation function.
        """


@dataclasses.dataclass
class BeliefDistribution:
    """A probability distribution over state sequences."""

    probs: Sequence[Tuple[TrajectoryWithRew, float]]
    """A list of (state sequence, probability) pairs."""

    def __post_init__(self):
        assert np.isclose(sum(prob for _, prob in self.probs), 1.0), "Probabilities do not sum to 1."

    @property
    def expected_total_reward(self) -> float:
        """Computes the expected total reward of the distribution."""
        return np.sum(prob * np.sum(traj.rews) for traj, prob in self.probs)

    def __str__(self):
        string = ""
        for traj, p in self.probs:
            string += f"{traj.state}: {p}, "
        return "{" + string[:-2] + "}"


class BeliefFunction(abc.ABC):
    """Abstract class for functions that take an observation sequence and return a distribution over state sequences."""

    @abc.abstractmethod
    def __call__(self, obs_seq: TrajectoryWithRewAndObs) -> BeliefDistribution:
        """Returns a distribution over state sequences given an observation sequence.

        Args:
            obs_seq: a TrajectoryWithRewAndObs object. In the scenario being modeled, the human sees only the
                observations, not the states.

        Returns:
            A distribution over state sequences, shape (num_states, trajectory_len + 1) + state_shape.
        """


# TODO: Really this should take in a sequence of TrajectoryWithRewAndObs objects, with the fragmenter handling the
# TrajectoryWithRew -> TrajectoryWithRewAndObs conversion. Getting it done quick and dirty for now.
class PORLHFHumanFeedbackModel(SyntheticPreferenceHumanFeedbackModel):
    """A synthetic preference human feedback model (gatherer) that takes state observations and returns preferences
    according to the PORLHF model described in "When Your AIs Deceive You".

    arXiv link: https://arxiv.org/abs/2402.17747
    """

    def __init__(self, observation_function: ObservationFunction, belief_function: BeliefFunction, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_function = observation_function
        self.belief_function = belief_function

    def __call__(self, fragment_pairs: Sequence[TrajectoryWithRewPair]) -> np.ndarray:
        """Computes preferences for the given fragments. For each of the two trajectories, the observation function is
        called to generate an observation sequence. Then, the belief function is called on the observation sequence to
        generate a distribution over fragments. The two distributions are scored by the expected total reward, and then
        the probability fragment 1 is preferred over fragment 2 is calculated using the standard Boltzmann preference
        model.

        This is super slow, and only reasonably usable for small toy experiments. A much faster implementation is
        possible, but not all that well-motivated: this algorithm's explicit calculation of an expectation over state
        sequences makes it too slow for real problems, even if the implementation were efficient.

        Args:
            fragment_pairs: a list of pairs of TrajectoryWithRew objects.

        Returns:
            A list of preferences, where each preference is the probability that the first fragment in the pair is
            preferred over the second fragment in the pair.
        """
        preferences = []
        for fragment_pair in fragment_pairs:
            obs_seq1 = self.observation_function(fragment_pair[0])
            obs_seq2 = self.observation_function(fragment_pair[1])
            belief_dist1 = self.belief_function(obs_seq1)
            belief_dist2 = self.belief_function(obs_seq2)
            score1 = belief_dist1.expected_total_reward
            score2 = belief_dist2.expected_total_reward
            score1 /= self.temperature
            score2 /= self.temperature

            # clip the returns to avoid overflow in the softmax below
            scores_diff = np.clip(score2 - score1, -self.threshold, self.threshold)
            model_probs = 1 / (1 + np.exp(scores_diff))
            if self.sample:
                assert self.rng is not None, "If `sample` is True, then `rng` must be provided."
                preference = np.float32(self.rng.binomial(1, model_probs))
            else:
                preference = model_probs
            preferences.append(preference)
        return np.array(preferences)


class StatesSameAsObsHumanModelWrapper(HumanFeedbackModel):
    """Wraps a human feedback model to handle the feedback giver seeing a noisy observation of state, rather than the
    true environment state. This wrapper works in cases where the observation and state spaces are the same; that is, a
    state is observed which may or may not be the actual underlying state.

    Current implementation only supports deterministic observation noise (such as occlusion). Later implementations
    will pass a random seed to the observation function to support stochastic observation noise. For now, a stochastic
    observation function will not fail, but will not be seed-able, so results will not be reproducible.
    """

    def __init__(self, gatherer, observe_fn):
        self.wrapped_gatherer = gatherer
        self.feedback_type = gatherer.feedback_type
        self.observe_fn = observe_fn

    def __getattr__(self, name):
        return getattr(self.wrapped_gatherer, name)

    def __call__(self, fragments: Union[Sequence[TrajectoryWithRew], Sequence[TrajectoryWithRewPair]]) -> np.ndarray:
        if self.feedback_type == FeedbackType.SCALAR:
            noisy_fragments = [self.observe_fn(fragment) for fragment in fragments]
        elif self.feedback_type == FeedbackType.PREFERENCE:
            noisy_fragments = [(self.observe_fn(fragment[0]), self.observe_fn(fragment[1])) for fragment in fragments]
        else:
            raise ValueError(f"Unsupported feedback type: {self.feedback_type}")
        return self.wrapped_gatherer(noisy_fragments)
