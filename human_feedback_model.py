import abc
from enum import Enum, auto

import numpy as np
import scipy.special as special
from imitation.algorithms.preference_comparisons import PreferenceDataset
from imitation.data import rollout
from imitation.util import logger as imit_logger


class FeedbackType(Enum):
    SCALAR = auto()
    PREFERENCE = auto()


class HumanFeedbackModel(abc.ABC):
    """
    Base class for a human feedback model. Called a gatherer in the imitation codebase.

    Can be subclassed to handle different types of feedback.
    """

    feedback_type: FeedbackType

    def __init__(
        self,
        rng=None,
        custom_logger=None,
    ) -> None:
        """Initializes the human feedback model.

        Args:
            rng: random number generator, if applicable.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        del rng  # unused.
        self.logger = custom_logger or imit_logger.configure()

    @abc.abstractmethod
    def __call__(self, fragments):
        pass


class PreferenceHumanFeedbackModel(HumanFeedbackModel):
    """
    Abstract base class for a human feedback model that takes preference feedback.

    Replaces imitation.algorithms.preference_comparisons.PreferenceGatherer.
    """

    feedback_type = FeedbackType.PREFERENCE
    dataset_class = PreferenceDataset
    # TODO: Add a __call__ abstract method that indicates the type of fragments.


class SyntheticPreferenceHumanFeedbackModel(PreferenceHumanFeedbackModel):
    """Computes synthetic preferences using ground-truth environment rewards.

    Copied from imitation.algorithms.preference_comparisons.SyntheticGatherer to properly inherit from
    PreferenceHumanFeedbackModel.
    """

    def __init__(
        self,
        temperature=1,
        discount_factor=1,
        sample=True,
        rng=None,
        threshold=50,
        custom_logger=None,
    ) -> None:
        """Initialize the synthetic preference gatherer.

        Args:
            temperature: the preferences are sampled from a softmax, this is
                the temperature used for sampling. temperature=0 leads to deterministic
                results (for equal rewards, 0.5 will be returned).
            discount_factor: discount factor that is used to compute
                how good a fragment is. Default is to use undiscounted
                sums of rewards (as in the DRLHP paper).
            sample: if True (default), the preferences are 0 or 1, sampled from
                a Bernoulli distribution (or 0.5 in the case of ties with zero
                temperature). If False, then the underlying Bernoulli probabilities
                are returned instead.
            rng: random number generator, only used if
                ``temperature > 0`` and ``sample=True``
            threshold: preferences are sampled from a softmax of returns.
                To avoid overflows, we clip differences in returns that are
                above this threshold (after multiplying with temperature).
                This threshold is therefore in logspace. The default value
                of 50 means that probabilities below 2e-22 are rounded up to 2e-22.
            custom_logger: Where to log to; if None (default), creates a new logger.

        Raises:
            ValueError: if `sample` is true and no random state is provided.
        """
        super().__init__(custom_logger=custom_logger)
        self.temperature = temperature
        self.discount_factor = discount_factor
        self.sample = sample
        self.rng = rng
        self.threshold = threshold

        if self.sample and self.rng is None:
            raise ValueError("If `sample` is True, then `rng` must be provided.")

    def __call__(self, fragment_pairs):
        """Computes probability fragment 1 is preferred over fragment 2."""
        returns1, returns2 = self._reward_sums(fragment_pairs)
        if self.temperature == 0:
            return (np.sign(returns1 - returns2) + 1) / 2

        returns1 /= self.temperature
        returns2 /= self.temperature

        # clip the returns to avoid overflows in the softmax below
        returns_diff = np.clip(returns2 - returns1, -self.threshold, self.threshold)
        # Instead of computing exp(rews1) / (exp(rews1) + exp(rews2)) directly,
        # we divide enumerator and denominator by exp(rews1) to prevent overflows:
        model_probs = 1 / (1 + np.exp(returns_diff))
        # Compute the mean binary entropy. This metric helps estimate
        # how good we can expect the performance of the learned reward
        # model to be at predicting preferences.
        entropy = -(special.xlogy(model_probs, model_probs) + special.xlogy(1 - model_probs, 1 - model_probs)).mean()
        self.logger.record("entropy", entropy)

        if self.sample:
            assert self.rng is not None
            return self.rng.binomial(n=1, p=model_probs).astype(np.float32)
        return model_probs

    def _reward_sums(self, fragment_pairs):
        rews1, rews2 = zip(
            *[
                (
                    rollout.discounted_sum(f1.rews, self.discount_factor),
                    rollout.discounted_sum(f2.rews, self.discount_factor),
                )
                for f1, f2 in fragment_pairs
            ],
        )
        return np.array(rews1, dtype=np.float32), np.array(rews2, dtype=np.float32)
