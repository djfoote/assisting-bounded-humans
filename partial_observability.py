import abc

from human_feedback_model import FeedbackType


# TODO: This should be agnostic to whether it's scalar or preference feedback
class PartialObservabilityHumanFeedbackModelWrapper:
    """Wraps a human feedback model to handle the feedback giver seeing a noisy observation of state, rather than the
    true environment state.

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

    def __call__(self, fragments):
        if self.feedback_type == FeedbackType.SCALAR:
            noisy_fragments = [self.observe_fn(fragment) for fragment in fragments]
        elif self.feedback_type == FeedbackType.PREFERENCE:
            noisy_fragments = [(self.observe_fn(fragment[0]), self.observe_fn(fragment[1])) for fragment in fragments]
        else:
            raise ValueError(f"Unsupported feedback type: {self.feedback_type}")
        return self.wrapped_gatherer(noisy_fragments)


# TODO: To match the theory, this should be forced to happen per state, not per fragment. Could replace or subclass.
class ObservationFunction(abc.ABC):
    """Abstract class for functions that take a state sequence and return an observation sequence."""

    @abc.abstractmethod
    def __call__(self, fragment):
        """Returns a new fragment with observations, actions, and rewards filtered through an observation function.

        Args:
            fragment: a TrajectoryWithRew object.

        Returns:
            A new TrajectoryWithRew object with the same infos and terminal flag, but with the observations, actions,
            and rewards filtered through the observation function.
        """
