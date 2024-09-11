import abc
import itertools
import pickle
import re

import numpy as np
import torch as th
from imitation.algorithms import preference_comparisons
from imitation.data import rollout
from imitation.util import logger as imit_logger
from imitation.util import networks
from torch import nn
from torch.utils import data as data_th
from tqdm.auto import tqdm

from human_feedback_model import FeedbackType, HumanFeedbackModel


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


class ScalarHumanFeedbackModel(HumanFeedbackModel):
    """
    Abstract base class for a human feedback model that takes scalar feedback.
    """

    feedback_type = FeedbackType.SCALAR
    dataset_class = ScalarFeedbackDataset
    # TODO: Add a __call__ abstract method that indicates the type of fragments it takes as input.


class GroundTruthScalarHumanFeedbackModel(ScalarHumanFeedbackModel):
    """Computes synthetic scalar feedback using ground-truth environment rewards."""

    def __call__(self, fragments):
        # NOTE: This ignores discounting.
        return [np.sum(fragment.rews) for fragment in fragments]


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
