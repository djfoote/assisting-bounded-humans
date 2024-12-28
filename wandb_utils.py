from pathlib import Path

import wandb
from graph_mdp import MDPWithObservations
from imitation_modules import TabularStatewiseRewardNet
from partial_observability import PORLHFHumanFeedbackModel
from value_iteration import TabularPolicy, get_optimal_policy


class WandbSaveDatasetCallback:
    def __init__(self, save_dir: str = "datasets"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def __call__(self, reward_learner):
        dataset = reward_learner.dataset
        step = reward_learner._iteration

        # Save locally first (temporary)
        filename = f"dataset_{step}.pkl" if step is not None else "dataset.pkl"
        local_path = self.save_dir / filename
        dataset.save(local_path)

        # Create and log artifact
        description = f"Training dataset at step {step}" if step is not None else "Training dataset"
        artifact = wandb.Artifact(name="training_dataset", type="dataset", description=description)
        artifact.add_file(str(local_path))
        wandb.log_artifact(artifact)

        # Remove local file after uploading
        local_path.unlink()


class WandbLogTabularRewardAndPolicyCallback:
    def __call__(self, reward_learner):
        env = reward_learner.trajectory_generator.env

        reward_net = reward_learner.model
        assert isinstance(reward_net, TabularStatewiseRewardNet)
        reward_table = wandb.Table(columns=["state", "reward"])
        for state, reward in reward_net.rewards_per_state(env).items():
            reward_table.add_data(state, reward)

        policy = reward_learner.trajectory_generator.policy
        assert isinstance(policy, TabularPolicy)
        policy_table = wandb.Table(columns=["state", "action"])
        for state, action in policy.actions_per_state().items():
            policy_table.add_data(state, action)

        wandb.log(
            {
                "learned_reward_function_table": reward_table,
                "learned_policy_table": policy_table,
                "learned_reward_function": reward_net.rewards_per_state(env),
                "learned_policy": policy.actions_per_state(),
            }
        )


class WandbLogPORLHFMetricsCallback:
    def __call__(self, reward_learner):
        env = reward_learner.trajectory_generator.env
        human_feedback_model = reward_learner.human_feedback_model
        assert isinstance(human_feedback_model, PORLHFHumanFeedbackModel)
        obs_fn = human_feedback_model.observation_function
        belief_fn = human_feedback_model.belief_function
        env_with_obs = MDPWithObservations(env, obs_fn, belief_fn)

        policy = reward_learner.trajectory_generator.policy
        optimal_policy = get_optimal_policy(env, horizon=env.horizon)

        J = env_with_obs.J(policy)
        J_obs = env_with_obs.J_obs(policy)
        overestimation_error = env_with_obs.overestimation_error(policy)
        underestimation_error = env_with_obs.underestimation_error(policy)

        deceptive_inflation = env_with_obs.is_deceptive_inflation(policy, ref_policy=optimal_policy)
        overjustification = env_with_obs.is_overjustification(policy, ref_policy=optimal_policy)

        wandb.log(
            {
                "J": J,
                "J_obs": J_obs,
                "overestimation_error": overestimation_error,
                "underestimation_error": underestimation_error,
                "deceptive_inflation": deceptive_inflation,
                "overjustification": overjustification,
            }
        )
