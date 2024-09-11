# Copy of code from experiment ipython notebook

import os

import numpy as np
import torch as th
from imitation.algorithms.preference_comparisons import (
    BasicRewardTrainer,
    CrossEntropyRewardLoss,
    PreferenceModel,
    RandomFragmenter,
)
from imitation.util import logger as imit_logger

import wandb
from evaluate_reward_model import full_visibility_evaluator_factory, partial_visibility_evaluator_factory
from human_feedback_model import SyntheticPreferenceHumanFeedbackModel
from imitation_modules import DeterministicMDPTrajGenerator, NonImageCnnRewardNet, RewardLearner
from partial_observability import PartialObservabilityHumanFeedbackModelWrapper
from scalar_feedback import (
    BasicScalarFeedbackRewardTrainer,
    GroundTruthScalarHumanFeedbackModel,
    MSERewardLoss,
    RandomSingleFragmenter,
    ScalarFeedbackModel,
)
from stealing_gridworld import PartialGridVisibility, StealingGridworld

#######################################################################################################################
##################################################### Run params ######################################################
#######################################################################################################################


GPU_NUMBER = None
N_ITER = 5
N_COMPARISONS = 1_000


#######################################################################################################################
##################################################### Expt params #####################################################
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
    # If fragment_length is None, then the whole trajectory is used as a single fragment.
    "fragment_length": None,
    "transition_oversampling": 10,
    "initial_epoch_multiplier": 1,
    "feedback": {
        "type": "preference",
    },
    "trajectory_generator": {
        "epsilon": 0.1,
    },
    "visibility": {
        "visibility": "full",
        # Available visibility mask keys:
        # "full": All of the grid is visible. Not actually used, but should be set for easier comparison.
        # "(n-1)x(n-1)": All but the outermost ring of the grid is visible.
        "visibility_mask_key": "full",
    },
    "reward_trainer": {
        "num_epochs": 3,
    },
}


# Some validation

if config["feedback"]["type"] not in ["scalar", "preference"]:
    raise NotImplementedError("Only scalar and preference feedback are supported at the moment.")

if config["visibility"]["visibility"] == "full" and config["visibility"]["visibility_mask_key"] != "full":
    raise ValueError(
        f'If visibility is "full", then visibility mask key must be "full".'
        f'Instead, it is {wandb.config["visibility"]["visibility_mask_key"]}.'
    )

if config["visibility"]["visibility"] not in ["full", "partial"]:
    raise ValueError(
        f'Unknown visibility {wandb.config["visibility"]["visibility"]}.' f'Visibility must be "full" or "partial".'
    )

if config["reward_model"]["type"] != "NonImageCnnRewardNet":
    raise ValueError(f'Unknown reward model type {wandb.config["reward_model"]["type"]}.')

available_visibility_mask_keys = ["full", "(n-1)x(n-1)"]
if config["visibility"]["visibility_mask_key"] not in available_visibility_mask_keys:
    raise ValueError(
        f'Unknown visibility mask key {wandb.config["visibility"]["visibility_mask_key"]}.'
        f"Available visibility mask keys are {available_visibility_mask_keys}."
    )

if config["fragment_length"] == None:
    config["fragment_length"] = config["environment"]["horizon"]

wandb.login()
run = wandb.init(
    project="assisting-bounded-humans",
    notes="finalizing logging pipeline for now",
    name="setup_debug_4",
    tags=[
        "debug",
    ],
    config=config,
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
human_feedback_model = GroundTruthScalarHumanFeedbackModel(rng=rng)


if wandb.config["feedback"]["type"] == "scalar":
    fragmenter = RandomSingleFragmenter(rng=rng)
    human_feedback_model = GroundTruthScalarHumanFeedbackModel(rng=rng)
else:
    fragmenter = RandomFragmenter(rng=rng)
    human_feedback_model = SyntheticPreferenceHumanFeedbackModel(rng=rng)


if wandb.config["visibility"]["visibility"] == "partial":
    visibility_mask = construct_visibility_mask(
        wandb.config["environment"]["grid_size"],
        wandb.config["visibility"]["visibility_mask_key"],
    )
    observation_function = PartialGridVisibility(env, visibility_mask=visibility_mask)
    human_feedback_model = PartialObservabilityHumanFeedbackModelWrapper(human_feedback_model, observation_function)
    policy_evaluator = partial_visibility_evaluator_factory(visibility_mask)
elif wandb.config["visibility"]["visibility"] == "full":
    policy_evaluator = full_visibility_evaluator_factory()


if wandb.config["feedback"]["type"] == "scalar":
    feedback_model = ScalarFeedbackModel(model=reward_net)
    reward_trainer = BasicScalarFeedbackRewardTrainer(
        feedback_model=feedback_model,
        loss=MSERewardLoss(),  # Will need to change this for preference learning
        rng=rng,
        epochs=config["reward_trainer"]["num_epochs"],
    )
else:
    feedback_model = PreferenceModel(model=reward_net)
    reward_trainer = BasicRewardTrainer(
        preference_model=feedback_model,
        loss=CrossEntropyRewardLoss(),
        rng=rng,
        epochs=config["reward_trainer"]["num_epochs"],
    )


trajectory_generator = DeterministicMDPTrajGenerator(
    reward_fn=reward_net,
    env=env,
    rng=None,  # This doesn't work yet
    epsilon=wandb.config["trajectory_generator"]["epsilon"],
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


reward_learner = RewardLearner(
    trajectory_generator=trajectory_generator,
    reward_model=reward_net,
    num_iterations=N_ITER,
    fragmenter=fragmenter,
    human_feedback_model=human_feedback_model,
    feedback_queue_size=wandb.config["dataset_max_size"],
    reward_trainer=reward_trainer,
    fragment_length=wandb.config["fragment_length"],
    transition_oversampling=wandb.config["transition_oversampling"],
    initial_epoch_multiplier=wandb.config["initial_epoch_multiplier"],
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
