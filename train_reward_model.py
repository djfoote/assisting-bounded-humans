# Copy of code from experiment ipython notebook

import time
import os
import numpy as np
import pickle
import torch as th
from imitation.algorithms import preference_comparisons

from imitation_modules import (
    BasicScalarFeedbackRewardTrainer,
    DeterministicMDPTrajGenerator,
    MSERewardLoss,
    NonImageCnnRewardNet,
    RandomSingleFragmenter,
    ScalarFeedbackModel,
    ScalarRewardLearner,
    SyntheticScalarFeedbackGatherer,
    NoisyObservationGathererWrapper,
)
from stealing_gridworld import StealingGridworld, PartialGridVisibility
from evaluate_reward_model import (
    full_visibility_evaluator_factory,
    partial_visibility_evaluator_factory,
)


#######################################################################################################################
##################################################### Run params ######################################################
#######################################################################################################################


CONTINUE_TRAINING_MODEL_NAME = "partial-vis_scalar_reward_model_5_32,32_3_20230424_210742"
GPU_NUMBER = 7
N_ITER = 20
N_COMPARISONS = 10_000


#######################################################################################################################
##################################################### Task params #####################################################
#######################################################################################################################


GRID_SIZE = 5
HORIZON = 30

HID_CHANNELS = (32, 32)
KERNEL_SIZE = 3

SEED = 0

DATASET_MAX_SIZE = 10_000

VISIBILITY = "partial"
visibility_mask = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.bool_)

if VISIBILITY == "full":
    # Try to catch mistakes mixing up full and partial visibility setup.
    del visibility_mask
    policy_evaluator = full_visibility_evaluator_factory()
elif VISIBILITY == "partial":
    policy_evaluator = partial_visibility_evaluator_factory(visibility_mask)
else:
    raise ValueError(f"Unknown visibility type {VISIBILITY}.")


#######################################################################################################################
################################################## Create everything ##################################################
#######################################################################################################################


env = StealingGridworld(
    grid_size=GRID_SIZE,
    max_steps=HORIZON,
    reward_for_depositing=100,
    reward_for_picking_up=1,
    reward_for_stealing=-200,
)
reward_net = NonImageCnnRewardNet(
    env.observation_space,
    env.action_space,
    hid_channels=HID_CHANNELS,
    kernel_size=KERNEL_SIZE,
)

rng = np.random.default_rng(SEED)

if GPU_NUMBER is not None:
    device = th.device(f"cuda:{GPU_NUMBER}" if th.cuda.is_available() else "cpu")
    reward_net.to(device)
    print(f"Reward net on {device}.")

fragmenter = RandomSingleFragmenter(rng=rng)
gatherer = SyntheticScalarFeedbackGatherer(rng=rng)

if VISIBILITY == "partial":
    observation_function = PartialGridVisibility(
        env,
        visibility_mask=visibility_mask,
    )    
    gatherer = NoisyObservationGathererWrapper(
        gatherer,
        observation_function,
    )

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


def save_model_params_and_dataset_callback(reward_learner):
    th.save(reward_learner.model.state_dict(), f"saved_reward_models/{model_name}/latest_checkpoint.pt")
    th.save(
        reward_learner.model.state_dict(),
        f"saved_reward_models/{model_name}/checkpoints/model_weights_iter{reward_learner._iteration}.pt"
    )
    reward_learner.dataset.save(
        f"saved_reward_models/{model_name}/checkpoints/dataset_iter{reward_learner._iteration}.pkl")
    reward_learner.dataset.save(f"saved_reward_models/{model_name}/latest_dataset.pkl")


reward_learner = ScalarRewardLearner(
    trajectory_generator=trajectory_generator,
    reward_model=reward_net,
    num_iterations=N_ITER,
    fragmenter=fragmenter,
    feedback_gatherer=gatherer,
    feedback_queue_size=DATASET_MAX_SIZE,
    reward_trainer=reward_trainer,
    fragment_length=3,
    transition_oversampling=5,
    initial_epoch_multiplier=1,
    policy_evaluator=policy_evaluator,
    callback=save_model_params_and_dataset_callback,
)


#######################################################################################################################
############################################## Set up saving and loading ##############################################
#######################################################################################################################


if CONTINUE_TRAINING_MODEL_NAME is None:
    hid_channels_str = ",".join([str(x) for x in HID_CHANNELS])
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    model_name = f"{VISIBILITY}-vis_scalar_reward_model_{GRID_SIZE}_{hid_channels_str}_{KERNEL_SIZE}_{timestamp}"
    print(f"Model name: {model_name}")
    os.makedirs(f"saved_reward_models/{model_name}/checkpoints", exist_ok=True)
else:
    model_name = CONTINUE_TRAINING_MODEL_NAME
    # Get the most recent checkpoint and dataset
    checkpoint_files = os.listdir(f"saved_reward_models/{model_name}/checkpoints")
    checkpoint_files = [x for x in checkpoint_files if x.endswith(".pt")]
    get_checkpoint = lambda x: int(x.split("_")[-1].split(".")[0][len("iter"):])
    checkpoint_files = sorted(checkpoint_files, key=get_checkpoint)
    checkpoint_file = checkpoint_files[-1]

    reward_learner.model.load_state_dict(th.load(f"saved_reward_models/{model_name}/checkpoints/{checkpoint_file}"))
    reward_learner._iteration = get_checkpoint(checkpoint_file)

    dataset_filename = f"saved_reward_models/{model_name}/checkpoints/dataset_iter{reward_learner._iteration}.pkl"
    with open(dataset_filename, 'rb') as f:
        reward_learner.dataset = pickle.load(f)
    
    reward_learner.trajectory_generator.train(HORIZON)

    print(f"Continuing training model {model_name} from checkpoint {checkpoint_file}")


#######################################################################################################################
####################################################### Training ######################################################
#######################################################################################################################


result = reward_learner.train(
    # Just needs to be bigger then N_ITER * HORIZON. Value iteration doesn't really use this.
    total_timesteps=10 * N_ITER * HORIZON,  
    total_queries=N_COMPARISONS,
)
