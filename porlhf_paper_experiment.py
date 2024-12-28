import numpy as np
from imitation.algorithms.preference_comparisons import BasicRewardTrainer, CrossEntropyRewardLoss, PreferenceModel
from imitation.util import logger as imit_logger

import porlhf_envs
import wandb
from imitation_modules import ExhaustiveFragmenter, ExhaustiveTrajGenerator, RewardLearner, TabularStatewiseRewardNet
from partial_observability import PORLHFHumanFeedbackModel, PORLHFPreferenceModel
from wandb_utils import WandbLogPORLHFMetricsCallback, WandbLogTabularRewardAndPolicyCallback, WandbSaveDatasetCallback


def run_condition():
    run = wandb.init()

    # Seed the run
    rng = np.random.default_rng(wandb.config.seed)

    # Initialize the environment
    env_config = wandb.config.env
    env, obs_fn, belief_fn = load_environment(env_config)

    # Initialize the reward and likelihood models
    reward_net = TabularStatewiseRewardNet(len(env.states))
    likelihood_model_type = run.config.likelihood_model_type
    if likelihood_model_type == "naive":
        learned_feedback_model = PreferenceModel(model=reward_net)
    elif likelihood_model_type == "po-aware":
        learned_feedback_model = PORLHFPreferenceModel(
            model=reward_net,
            observation_function=obs_fn,
            belief_function=belief_fn,
        )
    else:
        raise ValueError(f"Unknown likelihood model type: {likelihood_model_type}")

    # Initialize the feedback process
    trajectory_generator = ExhaustiveTrajGenerator(reward_fn=reward_net, env=env, rng=rng)
    fragmenter = ExhaustiveFragmenter(rng)
    human_feedback_model = PORLHFHumanFeedbackModel(observation_function=obs_fn, belief_function=belief_fn, rng=rng)

    # Initialize the learning algorithm and logging
    train_hyperparams = wandb.config.train_hyperparams
    reward_trainer = BasicRewardTrainer(
        preference_model=learned_feedback_model,
        loss=CrossEntropyRewardLoss(),
        rng=rng,
        epochs=train_hyperparams["num_epochs"],
    )
    logger = imit_logger.configure(format_strs=["stdout", "wandb"])
    callbacks = [WandbLogTabularRewardAndPolicyCallback(), WandbSaveDatasetCallback(), WandbLogPORLHFMetricsCallback()]
    reward_learner = RewardLearner(
        trajectory_generator=trajectory_generator,
        reward_model=reward_net,
        num_iterations=0,  # Learn offline from exhaustive trajectories
        fragmenter=fragmenter,
        human_feedback_model=human_feedback_model,
        reward_trainer=reward_trainer,
        feedback_queue_size=train_hyperparams["total_queries"],
        fragment_length=env.horizon,
        initial_epoch_multiplier=1,  # Don't need this for offline learning
        custom_logger=logger,
        callback=callbacks,
    )

    # Run training
    reward_learner.train(5, train_hyperparams["total_queries"])


def load_environment(env_config):
    env_name = env_config["env_name"]
    env_hyperparams = env_config["hyperparams"]
    if env_name == "hiding_env":
        return porlhf_envs.get_hiding_env_obs_and_belief(**env_hyperparams)
    elif env_name == "verbose_env":
        return porlhf_envs.get_verbose_env_obs_and_belief(**env_hyperparams)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")
