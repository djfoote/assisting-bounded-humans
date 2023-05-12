# Copy of code from experiment ipython notebook

import os
import pickle
import time

import numpy as np
import torch as th
from imitation.algorithms import preference_comparisons
from imitation.util import logger as imit_logger

import wandb
from evaluate_reward_model import full_visibility_evaluator_factory, partial_visibility_evaluator_factory
from imitation_modules import (
    BasicScalarFeedbackRewardTrainer,
    DeterministicMDPTrajGenerator,
    MSERewardLoss,
    NoisyObservationGathererWrapper,
    NonImageCnnRewardNet,
    RandomSingleFragmenter,
    ScalarFeedbackModel,
    ScalarRewardLearner,
    SyntheticScalarFeedbackGatherer,
)
from stealing_gridworld import PartialGridVisibility, StealingGridworld

#######################################################################################################################
##################################################### Run params ######################################################
#######################################################################################################################


GPU_NUMBER = None
N_ITER = 20
N_COMPARISONS = 10_000


#######################################################################################################################
##################################################### Task params #####################################################
#######################################################################################################################


config = {
    "environment": {
        "name": "StealingGridworld",
        "grid_size": 3,
        "horizon": 30,
        "reward_for_depositing": 100,
        "reward_for_picking_up": 1,
        "reward_for_stealing": -200,
    },
    "reward_model": {
        "type": "NonImageCnnRewardNet",
        "hid_channels": [32, 32],
        "kernel_size": 3,
    },
    "seed": 0,
    "dataset_max_size": 10_000,
    "fragment_length": 3,
    "feedback": {
        "type": "scalar",
    },
    "visibility": {
        "visibility": "full",
        # Available visibility mask keys:
        # "full": All of the grid is visible. Not actually used, but should be set for easier comparison.
        # "(n-1)x(n-1)": All but the outermost ring of the grid is visible.
        "visibility_mask_key": "full",
    },
}

wandb.login()
run = wandb.init(
    project="assisting-bounded-humans",
    notes="trying to log more stuff including the config",
    name="setup_debug_3",
    tags=[
        "debug",
    ],
    config=config,
)


if wandb.config["feedback"]["type"] != "scalar":
    raise NotImplementedError("Only scalar feedback is supported at the moment.")


if wandb.config["visibility"]["visibility"] == "full" and wandb.config["visibility"]["visibility_mask_key"] != "full":
    raise ValueError(
        f'If visibility is "full", then visibility mask key must be "full".'
        f'Instead, it is {wandb.config["visibility"]["visibility_mask_key"]}.'
    )


def construct_visibility_mask(grid_size, visibility_mask_key):
    # Any other visibility mask keys should be added here.
    if visibility_mask_key == "(n-1)x(n-1)":
        visibility_mask = np.zeros((grid_size, grid_size), dtype=np.bool_)
        visibility_mask[1:-1, 1:-1] = True
        return visibility_mask
    else:
        raise ValueError(f"Unknown visibility mask key {visibility_mask_key}.")


#######################################################################################################################
################################################## Create everything ##################################################
#######################################################################################################################


env = StealingGridworld(
    grid_size=wandb.config["environment"]["grid_size"],
    max_steps=wandb.config["environment"]["horizon"],
    reward_for_depositing=wandb.config["environment"]["reward_for_depositing"],
    reward_for_picking_up=wandb.config["environment"]["reward_for_picking_up"],
    reward_for_stealing=wandb.config["environment"]["reward_for_stealing"],
)


if wandb.config["reward_model"]["type"] != "NonImageCnnRewardNet":
    raise ValueError(f'Unknown reward model type {wandb.config["reward_model"]["type"]}.')


reward_net = NonImageCnnRewardNet(
    env.observation_space,
    env.action_space,
    hid_channels=wandb.config["reward_model"]["hid_channels"],
    kernel_size=wandb.config["reward_model"]["kernel_size"],
)

rng = np.random.default_rng(wandb.config["seed"])

if GPU_NUMBER is not None:
    device = th.device(f"cuda:{GPU_NUMBER}" if th.cuda.is_available() else "cpu")
    reward_net.to(device)
    print(f"Reward net on {device}.")

fragmenter = RandomSingleFragmenter(rng=rng)
gatherer = SyntheticScalarFeedbackGatherer(rng=rng)


if wandb.config["visibility"]["visibility"] != "full":
    visibility_mask = construct_visibility_mask(
        wandb.config["environment"]["grid_size"],
        wandb.config["visibility"]["visibility_mask_key"],
    )
    observation_function = PartialGridVisibility(env, visibility_mask=visibility_mask)
    gatherer = NoisyObservationGathererWrapper(gatherer, observation_function)
    policy_evaluator = partial_visibility_evaluator_factory(visibility_mask)
elif wandb.config["visibility"]["visibility"] == "full":
    policy_evaluator = full_visibility_evaluator_factory()
else:
    raise ValueError(f'Unknown visibility {wandb.config["visibility"]}.')


feedback_model = ScalarFeedbackModel(model=reward_net)
reward_trainer = BasicScalarFeedbackRewardTrainer(
    feedback_model=feedback_model,
    loss=MSERewardLoss(),
    rng=rng,
    epochs=3,
)
trajectory_generator = DeterministicMDPTrajGenerator(
    reward_fn=reward_net,
    env=env,
    rng=None,  # This doesn't work yet
    epsilon=0.1,
)
logger = imit_logger.configure(format_strs=["stdout", "wandb"])


def save_model_params_and_dataset_callback(reward_learner):
    data_dir = os.path.join(wandb.run.dir, "saved_reward_models")
    latest_checkpoint_path = os.path.join(data_dir, "latest_checkpoint.pt")
    latest_dataset_path = os.path.join(data_dir, "latest_dataset.pkl")
    checkpoints_dir = os.path.join(data_dir, "checkpoints")
    checkpoint_iter_path = os.path.join(checkpoints_dir, f"model_weights_iter{reward_learner._iteration}.pt")
    dataset_iter_path = os.path.join(checkpoints_dir, f"dataset_iter{reward_learner._iteration}.pkl")

    os.makedirs(checkpoints_dir, exist_ok=True)
    th.save(reward_learner.model.state_dict(), latest_checkpoint_path)
    th.save(reward_learner.model.state_dict(), checkpoint_iter_path)
    reward_learner.dataset.save(latest_dataset_path)
    reward_learner.dataset.save(dataset_iter_path)


reward_learner = ScalarRewardLearner(
    trajectory_generator=trajectory_generator,
    reward_model=reward_net,
    num_iterations=N_ITER,
    fragmenter=fragmenter,
    feedback_gatherer=gatherer,
    feedback_queue_size=wandb.config["dataset_max_size"],
    reward_trainer=reward_trainer,
    fragment_length=wandb.config["fragment_length"],
    transition_oversampling=5,
    initial_epoch_multiplier=1,
    policy_evaluator=policy_evaluator,
    custom_logger=logger,
    callback=save_model_params_and_dataset_callback,
)


#######################################################################################################################
####################################################### Training ######################################################
#######################################################################################################################


result = reward_learner.train(
    # Just needs to be bigger then N_ITER * HORIZON. Value iteration doesn't really use this.
    total_timesteps=10 * N_ITER * wandb.config["environment"]["horizon"],
    total_queries=N_COMPARISONS,
)