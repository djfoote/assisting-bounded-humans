import numpy as np
import torch as th
from imitation.algorithms.preference_comparisons import BasicRewardTrainer, CrossEntropyRewardLoss, PreferenceModel
from imitation.util import logger as imit_logger

import porlhf_envs
import wandb
from imitation_modules import ExhaustiveFragmenter, ExhaustiveTrajGenerator, RewardLearner, TabularStatewiseRewardNet
from partial_observability import PORLHFHumanFeedbackModel, PORLHFPreferenceModel
from wandb_utils import WandbLogPORLHFMetricsCallback, WandbLogTabularRewardAndPolicyCallback, WandbSaveDatasetCallback


class QValuesCallbackHiding:
    # Not exactly Q-values; I'm excluding the immediate reward

    def __init__(self, env, p):
        self.env = env
        self.p = p

    def __call__(self, model, epoch_number, train_loss):
        reward_net = model.model
        reward_values = reward_net.rewards_per_state(self.env)
        q_I_aC = (1 - self.p) * reward_values["L"] + self.p * reward_values["W"]
        q_I_aT = reward_values["T"]
        q_I_aH = (1 - self.p) * reward_values["LH"] + self.p * reward_values["WH"]
        wandb.log({"aC": q_I_aC, "aT": q_I_aT, "aH": q_I_aH}, commit=False)
        wandb.log({f"R({s})": r for s, r in reward_values.items()}, commit=False)


class QValuesCallbackVerbose:
    # Not exactly Q-values; I'm excluding the immediate reward

    def __init__(self, env, p):
        self.env = env
        self.p = p

    def __call__(self, model, epoch_number, train_loss):
        reward_net = model.model
        reward_values = reward_net.rewards_per_state(self.env)
        q_I_aD = (1 - self.p) * reward_values["L"] + self.p * reward_values["W"]
        q_I_aT = reward_values["T"]
        q_I_aV = (1 - self.p) * reward_values["LV"] + self.p * reward_values["WV"]
        wandb.log({"aD": q_I_aD, "aT": q_I_aT, "aV": q_I_aV}, commit=False)
        wandb.log({f"R({s})": r for s, r in reward_values.items()}, commit=False)


def run_condition():
    run = wandb.init()

    # Seed the run
    rng = np.random.default_rng(wandb.config.seed)

    # Initialize the environment
    env_config = wandb.config.env
    env, obs_fn, belief_fn = load_environment(env_config)
    if env_config["env_name"] == "hiding_env":
        q_values_callback = QValuesCallbackHiding(env, env_config["hyperparams"]["p"])
    elif env_config["env_name"] == "verbose_env":
        q_values_callback = QValuesCallbackVerbose(env, env_config["hyperparams"]["p"])
    else:
        raise ValueError(f"Unknown environment name: {env_config['env_name']}")

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
    human_feedback_model = PORLHFHumanFeedbackModel(
        observation_function=obs_fn,
        belief_function=belief_fn,
        sample=False,
    )

    # Initialize the learning algorithm and logging
    train_hyperparams = wandb.config.train_hyperparams
    reward_trainer = BasicRewardTrainer(
        preference_model=learned_feedback_model,
        loss=CrossEntropyRewardLoss(),
        rng=rng,
        epochs=train_hyperparams["num_epochs"],
        lr=train_hyperparams["lr"],
        callbacks=[q_values_callback],
    )
    logger = imit_logger.configure(format_strs=["stdout", "wandb"])
    callbacks = [WandbLogTabularRewardAndPolicyCallback(), WandbLogPORLHFMetricsCallback()]
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
